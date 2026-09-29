"""Event-driven controller startup, Pico modes, and explicit Sonic stop packets."""
import os
import select
import sys
import termios
import tty
from enum import IntEnum
import uuid
import numpy as np

from ..core.runtime import Module
from ..core.protocol import RobotCommand, StopCommand
from ..core.streams import FieldReader, FieldWriter, command_metadata, field_ports


def stop_command(session, sequence, now_ns):
    """Sonic's explicit stop packet, carried through the ordinary command stream."""
    return StopCommand(session, sequence, "sonic.control.v1", now_ns,
        {"start": np.array([0], np.uint8), "stop": np.array([1], np.uint8),
         "planner": np.array([1], np.uint8)}, "operator", sequence)


class StreamMode(IntEnum):
    OFF = 0
    POSE = 1
    PLANNER = 2
    PLANNER_FROZEN_UPPER_BODY = 3
    POSE_PAUSE = 4
    PLANNER_VR_3PT = 5


def next_mode(current, parent, controls, previous):
    """Pure state transition matching run_pico_manager's edge/priority rules."""
    a, b, x, y = controls["buttons"]
    flags = {"start": bool(a and b and x and y), "ax": bool(a and x),
             "by": bool(b and y), "click": bool(controls["clicks"][0])}
    rising = {key: value and not previous.get(key, False) for key, value in flags.items()}
    target = current
    if current == StreamMode.OFF:
        if rising["start"]:
            target = StreamMode.PLANNER
    elif rising["start"]:
        target = StreamMode.OFF
    elif current == StreamMode.PLANNER:
        if rising["ax"]:
            target = StreamMode.POSE
        elif rising["click"]:
            target = StreamMode.PLANNER_VR_3PT
    elif current == StreamMode.POSE:
        if rising["ax"]:
            target = StreamMode.PLANNER
        elif rising["by"]:
            target = StreamMode.PLANNER_FROZEN_UPPER_BODY
        elif controls["menus"][0]:
            target = StreamMode.POSE_PAUSE
    elif current == StreamMode.PLANNER_FROZEN_UPPER_BODY:
        if rising["by"]:
            target = StreamMode.POSE
        elif rising["click"]:
            target = StreamMode.PLANNER_VR_3PT
    elif current == StreamMode.POSE_PAUSE:
        if not controls["menus"][0]:
            target = StreamMode.POSE
    elif current == StreamMode.PLANNER_VR_3PT:
        if rising["click"]:
            target = parent
        elif rising["ax"] or rising["by"]:
            target = StreamMode.POSE
    if target == StreamMode.PLANNER_VR_3PT and current != target:
        parent = current
    return target, parent, flags


class Lifecycle(Module):
    """Single-key input, Pico modes, controller startup, and stop commands.

    Bound Pico controls select teleoperation. With command outputs alone, Enter
    advances staged controller startup. Without command outputs, this module
    only emits keyboard events (for example, to start a replay).
    """
    event_actions = {"stop"}

    def bind(self, ctx):
        removed = {"startup", "startup_wait_seconds", "start_controller", "planner"} & self.options.keys()
        if removed:
            raise ValueError(f"Lifecycle starts manually from keyboard/Pico input; remove {sorted(removed)}")
        self.controls_in = FieldReader(ctx, "controls")
        self.pico = bool(self.controls_in)
        self.command_out = FieldWriter(ctx, "command")
        self.mode_out = FieldWriter(ctx, "mode")
        self.enabled_output = ctx.bind_write(["enabled"] if "enabled" in ctx.outputs else [])
        self.body_input = ctx.bind_read(["body"] if "body" in ctx.inputs else [])
        self.trigger_input = ctx.bind_read(["triggers"] if "triggers" in ctx.inputs else [])
        if self.command_out and set(self.command_out.fields) != {"start", "stop", "planner"}:
            raise ValueError("Lifecycle needs command.start, command.stop and command.planner outputs")
        if self.pico:
            if not {"buttons", "grips", "axes", "menus", "clicks"} <= set(self.controls_in.fields):
                raise ValueError("Pico controls need buttons, grips, axes, menus and clicks")
            if not self.command_out or "value" not in self.mode_out.fields:
                raise ValueError("Pico lifecycle needs command outputs and mode.value")
        self.planner_seconds = float(self.options.get("planner_seconds", 2.5))
        if not np.isfinite(self.planner_seconds) or self.planner_seconds < 0:
            raise ValueError("planner_seconds must be finite and nonnegative")

    def start(self, ctx):
        self.terminal_state, self.fd = None, None
        if self.pico and ctx.rate_hz <= 0:
            raise ValueError("Pico lifecycle requires a positive rate_hz")
        self.session, self.sequence = str(uuid.uuid4()), 0
        self.phase = "off"
        self.mode, self.parent = StreamMode.OFF, StreamMode.PLANNER
        self.button_previous, self.record_previous = {}, (False, False)
        self.pose_yaw = 0.
        self.pending_control = None
        self.stop_published = self.stop_pending = False
        self.mode_fields = {}
        self.command_out.invalidate()
        self._enabled(False)
        self.stream = sys.stdin
        self.keyboard_active = True
        if self.stream.isatty():
            self.fd = self.stream.fileno()
            self.terminal_state = termios.tcgetattr(self.fd)
            # Keep ISIG so Ctrl+C still reaches the runtime's signal handler.
            tty.setcbreak(self.fd)
        ctx.logger.info("Single-key controls ready; Enter starts/advances, Ctrl+C exits")
        if self.pico:
            ctx.report("IDLE", "OFF: press ABXY to start/calibrate, then AX for pose teleoperation")
            ctx.logger.info("Pico: ABXY=start/stop, AX=pose/planner, BY=freeze, left menu=pause")
        elif self.command_out:
            ctx.report("IDLE", "Ready; press Enter to start the controller in planner mode")
        else:
            ctx.report("READY", "Keyboard ready; actions follow configured event bindings")

    def _enabled(self, value):
        if self.enabled_output:
            self.enabled_output.write((np.asarray(value),))

    def _control(self, planner):
        self.pending_control = {"start": np.array([1], np.uint8), "stop": np.array([0], np.uint8),
                                "planner": np.array([planner], np.uint8)}

    def _stop_command(self, ctx):
        if not getattr(self, "stop_published", True):
            self.sequence += 1
            command = stop_command(self.session, self.sequence, ctx.now_ns)
            self.command_out.write(command.values, spec_id=command.spec_id, command=command_metadata(command))
            self.stop_published = True
        self.phase, self.pending_control = "stopped", None
        self.stop_pending = False
        self.mode, self.mode_fields = StreamMode.OFF, {}
        self.mode_out.invalidate()
        self._enabled(False)

    def stop(self, ctx):
        try:
            # A queued ESTOP must still be honored if the worker already exited.
            explicit_stop = any(event.kind in {"estop", "stop"} for event in ctx.events())
            if self.options.get("stop_on_exit", True) or explicit_stop or getattr(self, "stop_pending", False):
                self._stop_command(ctx)
        finally:
            if self.terminal_state is not None:
                termios.tcsetattr(self.fd, termios.TCSANOW, self.terminal_state)
                self.terminal_state = None

    def on_event(self, ctx, event):
        if event.kind in {"estop", "stop"}:
            self._stop_command(ctx)
        elif event.kind == "shutdown":
            if self.options.get("stop_on_exit", True):
                self._stop_command(ctx)
            else:
                self.phase, self.pending_control = "stopped", None
                self.mode_out.invalidate()
                self._enabled(False)
        elif event.kind in {"reset", "discontinuity"}:
            self._enabled(False)
            if self.pico:
                self.mode, self.pending_control, self.mode_fields = StreamMode.OFF, None, {}
                self.button_previous = {}
                self.mode_out.invalidate()
                self.command_out.invalidate()
            elif self.phase not in {"off", "stopped"}:
                self.phase = "confirmation"
                self._control(False)

    def _advance(self, ctx):
        if self.stop_published or self.phase == "stopped":
            return
        if self.phase == "off":
            self.phase = "planner"
            self._control(True)
            self.deadline = ctx.now_ns + int(self.planner_seconds * 1e9)
            ctx.report("IDLE", "Controller started; planner mode settling")
        elif self.phase == "confirmation":
            self.phase = "active"
            self._enabled(True)
            ctx.emit("started")
            ctx.report("OK", "Model control enabled")

    def process(self, ctx):
        self._read_key(ctx)
        if ctx.write_generation != ctx.generation or ctx.stopping:
            return
        if self.stop_pending:
            self._stop_command(ctx)
        if self.stop_published or self.phase == "stopped":
            return
        if self.pico:
            if not self._prepare_pico(ctx):
                return
            if self.mode == StreamMode.POSE:
                body = self.body_input.read()[0] if self.body_input else None
                triggers = self.trigger_input.read()[0] if self.trigger_input else None
                if (body is not None and triggers is not None
                        and body.fresh(ctx.now_ns, 100_000_000) and triggers.fresh(ctx.now_ns, 100_000_000)
                        and body.value.shape == (24, 7) and triggers.value.shape == (2,)
                        and np.isfinite(body.value).all() and np.isfinite(triggers.value).all()
                        and np.all(np.linalg.norm(body.value[:, 3:], axis=1) >= 1e-8)):
                    self._record_controls(ctx, self.current_controls)
        elif self.phase == "planner" and ctx.now_ns >= self.deadline:
            self.phase = "confirmation"
            self._control(False)
            ctx.emit("ready")
            ctx.report("IDLE", "Pose mode ready; press Enter to start model commands/replay")
        if self.pending_control is not None:
            self.sequence += 1
            command = RobotCommand(self.session, self.sequence, "sonic.control.v1", ctx.now_ns + (200_000_000 if self.pico else 500_000_000),
                                   self.pending_control, "operator", self.sequence)
            self.command_out.write(command.values, spec_id=command.spec_id, command=command_metadata(command))
            self.pending_control = None
        self._enabled(self.mode == StreamMode.POSE if self.pico else self.phase == "active")

    def _read_key(self, ctx):
        if not self.keyboard_active:
            return
        try:
            readable = select.select([self.stream], [], [], 0)[0]
        except (ValueError, OSError):
            return
        if not readable:
            return
        key = os.read(self.fd, 1).decode('latin1') if self.fd is not None else self.stream.read(1)
        if not key:
            self.keyboard_active = False
            return
        if key in {'\r', '\n'}:
            if self.command_out and not self.pico:
                self._advance(ctx)
            else:
                ctx.emit("enter")
        elif key.isascii() and key.isprintable():
            ctx.emit(key.lower())

    def _mode_status(self):
        hints = {StreamMode.OFF: "press ABXY to start/calibrate", StreamMode.PLANNER: "joystick control; AX switches to pose",
                 StreamMode.POSE: "pose teleoperation", StreamMode.POSE_PAUSE: "release left menu to resume",
                 StreamMode.PLANNER_FROZEN_UPPER_BODY: "upper body frozen; BY returns to pose",
                 StreamMode.PLANNER_VR_3PT: "VR three-point control; left stick click returns"}
        return f"{self.mode.name}: {hints[self.mode]}"

    def _transition(self, ctx, target):
        previous = self.mode
        if previous == StreamMode.POSE:
            ctx.emit("record_stop")
        if previous == StreamMode.OFF and target != StreamMode.OFF:
            ctx.emit("calibrate", {"reference": "zero"})
        if target == StreamMode.PLANNER_VR_3PT:
            ctx.emit("calibrate", {"reference": "measured"})
        if target == StreamMode.POSE:
            self.pose_yaw = 0.
        self.mode = target
        ctx.logger.info("Mode %s -> %s", previous.name, self._mode_status())
        ctx.report("IDLE" if target == StreamMode.OFF else "OK", self._mode_status())
        ctx.emit("sonic_mode", {"previous": previous.name, "mode": target.name})
        if target == StreamMode.OFF:
            self.pending_control = None
            self.stop_pending = True
            # Stop hooks publish Sonic stop data after the reset, before the
            # command sender drains its queue and closes.
            ctx.emit("stopped", {"reason": "Pico ABXY stop"})
            return
        if target != StreamMode.POSE_PAUSE:
            self._control(target != StreamMode.POSE)

    def _prepare_pico(self, ctx):
        controls = self.controls_in.read(max_age_ns=200_000_000)
        if not controls:
            ctx.report("WAITING", f"{self.mode.name}: waiting for fresh Pico controls")
            self.mode_out.invalidate()
            self.mode_fields = {}
            return False
        self.current_controls = controls
        target, self.parent, self.button_previous = next_mode(self.mode, self.parent, controls, self.button_previous)
        if target != self.mode:
            self._transition(ctx, target)
            if target == StreamMode.OFF:
                if ctx.write_generation == ctx.generation and not ctx.stopping:
                    self._stop_command(ctx)
                return False
        increment = 1.5 * -float(controls["axes"][2]) * (.95 / ctx.rate_hz)
        if self.mode == StreamMode.POSE and abs(controls["axes"][2]) >= .15:
            self.pose_yaw += increment
        self.mode_fields = {"value": int(self.mode), "heading": self.pose_yaw,
                             "heading_increment": increment, "toggle_record": False, "abort_record": False}
        self.mode_out.write(self.mode_fields)
        if not controls.get("tracking", True):
            ctx.report("WAITING", f"{self.mode.name}: Pico body tracking unavailable")
            self.mode_out.invalidate()
            return False
        if self.mode != StreamMode.POSE:
            ctx.report("IDLE" if self.mode == StreamMode.OFF else "OK", self._mode_status())
        return True

    def _record_controls(self, ctx, controls):
        # Like PoseStreamer.run_once(), consume recording edges only while
        # processing a valid pose. Presses in planner/OFF do not consume them.
        a, b, _, _ = controls["buttons"]
        toggle, abort = bool(a and controls["grips"][0] > .5), bool(b and controls["grips"][0] > .5)
        toggle_edge, abort_edge = toggle and not self.record_previous[0], abort and not self.record_previous[1]
        self.record_previous = toggle, abort
        self.mode_fields.update(toggle_record=toggle_edge, abort_record=abort_edge)
        self.mode_out.write(self.mode_fields)
        if abort_edge:
            ctx.emit("record_abort")
        elif toggle_edge:
            ctx.emit("record_toggle")
