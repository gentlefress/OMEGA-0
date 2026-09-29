"""Sonic pose retargeting and hand command preparation."""
from __future__ import annotations

import uuid

import numpy as np

from ..core.streams import FieldReader, FieldWriter, command_metadata
from ..core.protocol import RobotCommand, StopCommand
from ..core.runtime import Module
from ..core.streams import field_ports
from ..transforms.geometry import forward_kinematics, wrist_targets
from .lifecycle import StreamMode
from .geometry import G1Kinematics, ThreePointCalibration, load_skeleton, neck_targets, pose_lerp, quat_lerp


# Pico's first 22 body joints follow SMPL order. Quaternions use XYZW order.
SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
PICO_WORLD_TO_SMPL_XYZW = (0.7071067811865476, 0., 0., 0.7071067811865476)
PICO_JOINT_OFFSETS_XYZW = ((0., 1., 0., 0.),) * 22


def hand_targets(trigger, calibration):
    """Interpolate configured channels; each hand owns its array length."""
    opened, closed = np.asarray(calibration["open"], np.float32), np.asarray(calibration["closed"], np.float32)
    if opened.shape != closed.shape or opened.ndim != 1:
        raise ValueError("Hand calibration dimensions differ")
    result = opened + (closed - opened) * float(np.clip(trigger, 0, 1))
    curve = calibration.get("curve", "linear")
    if curve == "inspire_legacy":
        if result.size < (6 if calibration.get("thumb_bend", False) else 5):
            raise ValueError("Hand curve requires the configured thumb channels")
        result[4] = np.sin(np.sqrt(result[4] + 1e-8) * np.pi / 2)
        if calibration.get("thumb_bend", False):
            result[5] = 0
    elif curve != "linear":
        raise ValueError("Unknown hand calibration curve")
    for index, value in calibration.get("fixed", {}).items():
        result[int(index)] = float(value)
    return result


class SonicCommandAdapter(Module):
    """Prepare Sonic commands from named model fields or raw XR input.

    Lifecycle mode and calibration arrive through named inputs and events.
    Optional reference/details/vr outputs expose data for recording.
    """

    event_actions = {"sonic_mode", "calibrate"}

    def bind(self, ctx):
        self.body_input = ctx.bind_read(['body'] if 'body' in ctx.inputs else [])
        self.joints_input = ctx.bind_read(['joints'] if 'joints' in ctx.inputs else [])
        self.command_in = FieldReader(ctx, 'command')
        self.command_out = FieldWriter(ctx, 'command')
        self.controls_in = FieldReader(ctx, 'controls')
        self.details_out = FieldWriter(ctx, 'details')
        self.mode_in = FieldReader(ctx, 'mode')
        self.reference_out = FieldWriter(ctx, 'reference')
        self.timing_fields = tuple(field_ports(ctx.inputs, 'timing'))
        self.vr_out = FieldWriter(ctx, 'vr')
        self.xr_input = ctx.bind_read(('body', 'triggers', *('timing.' + name for name in self.timing_fields))
                                      if {'body', 'triggers'} <= ctx.inputs.keys() else ())

        self.planner_out = FieldWriter(ctx, "planner")
        if self.planner_out and ("value" not in self.mode_in.fields or
                                 not {"buttons", "axes"} <= set(self.controls_in.fields)):
            raise ValueError("Planner outputs need mode.value and controls.buttons/axes")
        self.freeze_fields = tuple((port, field) for port, field in (
            ("joints", "upper_body_position"), ("left_hand", "left_hand_joints"),
            ("right_hand", "right_hand_joints")) if port in ctx.inputs)
        self.freeze_input = ctx.bind_read(port for port, _ in self.freeze_fields)

    def start(self, ctx):
        groups = {"command", "reference", "details", "vr", "planner"}
        if any("." not in port or port.split(".", 1)[0] not in groups for port in ctx.outputs):
            raise ValueError("Sonic outputs must name tensor fields, such as command.token_state")
        self.retargeting = "body" in ctx.inputs or "triggers" in ctx.inputs
        if self.retargeting:
            if not {"body", "triggers"} <= ctx.inputs.keys():
                raise ValueError("XR inputs need both body and triggers")
            if not any(field_ports(ctx.outputs, group) for group in ("command", "reference", "details")):
                raise ValueError("XR-driven Sonic needs a command, reference or details output")
        elif not field_ports(ctx.inputs, "command") or not field_ports(ctx.outputs, "command"):
            raise ValueError("Model-driven Sonic needs command input and output")
        if "buttons" in self.controls_in.fields and "value" not in self.mode_in.fields:
            raise ValueError("Pico controls require mode.value from Lifecycle")
        if "stop_on_exit" in self.options:
            raise ValueError("stop_on_exit belongs to the lifecycle module")
        if "action_spec" in self.options or "layout" in self.options:
            raise ValueError("Sonic accepts named inputs; model action shape and timing come from the inference server")
        self.last_sequence = None
        self.vr_fields = None
        self.skeleton = None
        if self.options.get("skeleton"):
            self.skeleton = load_skeleton(self.options["skeleton"])
        if self.retargeting and self.command_out and self.skeleton is None:
            raise ValueError("XR pose commands require a calibrated skeleton")
        self.history = []
        self.action_period_ns = int(self.options.get("action_period_ns", 20_000_000))
        if self.action_period_ns <= 0:
            raise ValueError("Action period must be positive")
        self.reference_spec_id = "g1.sonic.smpl71.v1"
        self.hands = self.options.get("hands", {})
        # Pose histories and individual frames have different shapes. Other
        # fields are shared by sender and recorder through the same stream.
        distinct = {"smpl_pose", "smpl_joints", "body_quat_w", "joint_pos"}
        if self.options.get("hand_batch_dim", False):
            distinct |= {"left_hand_joints", "right_hand_joints"}
        duplicate = set(self.command_out.fields) & set(self.details_out.fields) - distinct
        if duplicate:
            raise ValueError(f"Bind shared Sonic fields once for sender and recorder: {sorted(duplicate)}")
        self.model_fields = set(self.command_in.fields)
        pose_fields = {"smpl_pose", "smpl_joints", "body_quat_w", "joint_pos", "joint_vel"}
        allowed = pose_fields | {"token_state", "frame_index", "left_trigger", "right_trigger"}
        if self.model_fields - allowed:
            raise ValueError(f"Unknown Sonic model inputs: {sorted(self.model_fields - allowed)}")
        self.model_kind = "latent" if "token_state" in self.model_fields else "pose" if self.model_fields & pose_fields else "hands"
        if self.command_in:
            required = {side + "_trigger" for side in self.hands if side + "_hand_joints" in self.command_out.fields}
            if not required <= self.model_fields:
                raise ValueError(f"Sonic hand outputs need trigger inputs: {sorted(required)}")
        self.source_priority = self.options.get("source_priority")
        if self.source_priority not in {None, "model", "xr"}:
            raise ValueError("source_priority must be model or xr")
        if self.command_out:
            expected = set()
            if self.command_in:
                expected |= set(self.command_in.fields) - {"left_trigger", "right_trigger"}
            if self.retargeting:
                expected |= {"smpl_pose", "smpl_joints", "body_quat_w", "joint_pos", "joint_vel", "frame_index",
                    "neck_pose", "left_trigger", "right_trigger", "left_grip", "right_grip", "pico_dt", "pico_fps",
                    "timestamp_realtime", "timestamp_monotonic", "heading_increment", "toggle_data_collection",
                    "toggle_data_abort", "vr_position", "vr_orientation"}
            expected |= {side + "_hand_joints" for side in self.hands}
            if not set(self.command_out.fields) <= expected:
                raise ValueError(f"Sonic outputs are unavailable from the bound inputs: {sorted(set(self.command_out.fields) - expected)}")
        if self.retargeting:
            self._start_retargeting(ctx)
        if self.planner_out:
            self._start_planner(ctx)

    def _hands(self, fields, left, right):
        for side, trigger in (("left", left), ("right", right)):
            if side in self.hands:
                targets = hand_targets(trigger, self.hands[side])
                fields[side + "_hand_joints"] = targets[None] if self.options.get("hand_batch_dim", False) else targets
        return fields

    def _model_fields(self, command):
        values = command.values
        for name, value in values.items():
            value = np.asarray(value)
            dtype = np.int64 if name == "frame_index" else np.float32
            if value.dtype != dtype or not np.isfinite(value).all():
                raise ValueError(f"Invalid Sonic {name} dtype or values")
        shapes = {"left_trigger": (1,), "right_trigger": (1,), "token_state": (1,64)}
        tails = {"smpl_pose": (21,3), "smpl_joints": (24,3), "body_quat_w": (4,), "joint_pos": (29,), "joint_vel": (29,)}
        sizes = set()
        for name, value in values.items():
            if name in shapes and value.shape != shapes[name]:
                raise ValueError(f"Invalid Sonic {name} shape")
            if name in tails:
                if value.ndim != len(tails[name]) + 1 or value.shape[1:] != tails[name] or value.shape[0] < 1:
                    raise ValueError(f"Invalid Sonic {name} shape")
                sizes.add(value.shape[0])
        if len(sizes) > 1:
            raise ValueError("Sonic pose inputs must have matching history lengths")
        if "frame_index" in values and values["frame_index"].shape not in {(1,)} | {(size,) for size in sizes}:
            raise ValueError("Invalid Sonic frame_index shape")
        fields = {name: value for name, value in values.items() if name in self.command_out.fields}
        for side in self.hands:
            name = side + "_hand_joints"
            if name in self.command_out.fields:
                targets = hand_targets(values[side + "_trigger"][0], self.hands[side])
                fields[name] = targets[None] if self.options.get("hand_batch_dim", False) else targets
        return fields

    def _xr_fields(self, command):
        frame = self.reference_frame
        joints = (self.reference_details['joint_pos'] if self.reference_details is not None
                  and 'joint_pos' in self.reference_details else wrist_targets(frame['pose']))
        current = {"smpl_pose": frame["pose"], "smpl_joints": frame["joints"], "body_quat_w": frame["quat"],
                   "joint_pos": joints, "joint_vel": np.zeros(29), "frame_index": command.step}
        size = int(self.options.get("history_size", 1))
        if size < 1:
            raise ValueError("History size must be positive")
        self.history.append(current)
        self.history = self.history[-size:]
        if self.options.get("history_start", "pad") == "wait" and len(self.history) < size:
            return None
        padded = [self.history[0]] * (size - len(self.history)) + self.history
        fields = {name: np.stack([item[name] for item in padded]).astype(np.float32) for name in current if name != "frame_index"}
        fields["frame_index"] = np.asarray([item["frame_index"] for item in padded] if self.options.get("history_frame_indices", False) else [command.step], np.int64)
        fields.update({key: np.asarray(value) for key, value in (self.reference_details or {}).items() if key not in {"smpl_pose", "smpl_joints", "body_quat_w", "joint_pos", "left_hand_joints", "right_hand_joints"}})
        for side in self.hands:
            name = side + "_hand_joints"
            if name in self.command_out.fields:
                targets = self.reference_details[name]
                fields[name] = targets[None] if self.options.get("hand_batch_dim", False) else targets
        return fields

    def _invalidate_command(self):
        self.last_sequence = None
        self.history = []
        self.command_out.invalidate()

    def process(self, ctx):
        self._prepare_commands(ctx)
        if self.planner_out:
            self._prepare_planner(ctx)

    def _prepare_commands(self, ctx):
        model_samples = self.command_in.samples()
        model = None
        if model_samples and all(s is not None and s.valid and s.command is not None
                                 for s in model_samples.values()):
            first = next(iter(model_samples.values()))
            if all((s.spec_id, s.command, s.batch_id) == (first.spec_id, first.command, first.batch_id)
                   for s in model_samples.values()):
                info = first.command
                command_type = StopCommand if info.stop else RobotCommand
                model = command_type(info.session_id, info.sequence, first.spec_id, info.valid_until_ns,
                                     {name: sample.value for name, sample in model_samples.items()}, info.source, info.step)
        if model is not None and model.valid_until_ns <= ctx.now_ns:
            model = None
        model_fields = None
        if model is not None:
            try:
                model_fields = self._model_fields(model)
            except (ValueError, TypeError, IndexError) as error:
                ctx.report("DEGRADED", str(error))
                model = None
        xr = None
        if self.retargeting:
            self._retarget(ctx)
            xr = self.reference
            if xr is not None and xr.valid_until_ns <= ctx.now_ns:
                xr = None
        if not self.command_out:
            return
        if model is not None and xr is not None and self.source_priority is None:
            self._invalidate_command()
            ctx.report("DEGRADED", "Both model and XR inputs are valid; configure source_priority")
            return
        source = "model" if model is not None and (xr is None or self.source_priority == "model") else "xr"
        command = model if source == "model" else xr
        if command is None:
            self._invalidate_command()
            return
        metadata = {"batch_id": next(iter(model_samples.values())).batch_id} if source == "model" else self.reference_metadata
        identity = (source, command.session_id, command.sequence, metadata["batch_id"])
        if identity == self.last_sequence:
            return
        for name, bounds in self.options.get("limits", {}).items():
            values = command.values[name]
            if ((values < bounds[0]) | (values > bounds[1])).any():
                self._invalidate_command()
                ctx.report("DEGRADED", f"Command {name} outside limits")
                return
        if self.last_sequence is not None and identity[:2] != self.last_sequence[:2]:
            self.history = []
        fields = model_fields if source == "model" else self._xr_fields(command)
        self.last_sequence = identity
        if fields is None:
            self.command_out.invalidate()
            ctx.report("WAITING", f"Building pose history ({len(self.history)}/{self.options.get('history_size', 1)} frames)")
            return
        prepared = RobotCommand(command.session_id, command.sequence, command.spec_id,
                                command.valid_until_ns, fields, command.source, command.step)
        self.command_out.write(prepared.values, spec_id=prepared.spec_id,
                               command=command_metadata(prepared), **metadata)
        ctx.report("OK", "POSE: publishing teleoperation commands" if source == "xr" else "Preparing model commands")

    def on_event(self, ctx, event):
        if self.planner_out and event.kind in {"reset", "discontinuity", "estop", "shutdown"}:
            self._reset_planner(ctx)
        if event.kind in {"reset", "discontinuity", "estop", "shutdown"} or (event.kind == "sonic_mode" and (event.payload["mode"] == "POSE" or event.payload["previous"] == "POSE")):
            self.last_sequence = None
            self.history = []
            if self.command_out:
                self.command_out.invalidate()
            if self.retargeting:
                self._reset_retargeting(ctx)
        elif self.retargeting and event.kind == "calibrate":
            self.pending_calibration = event.payload.get("reference", "zero")

    def _start_retargeting(self, ctx):
        self.interpolate = bool(self.options.get("interpolate", False))
        if self.interpolate and self.skeleton is None:
            raise ValueError("Pico interpolation needs the matching SMPL rest skeleton")
        self.vr = ThreePointCalibration()
        self.kinematics = G1Kinematics(self.options["robot_urdf"], self.options.get("robot_joint_names")) if self.options.get("robot_urdf") else None
        self.hand_position = np.zeros(2)
        self.pending_calibration = None
        self._reset_retargeting(ctx)

    def _reset_retargeting(self, ctx):
        self.session, self.sequence, self.last = str(uuid.uuid4()), 0, None
        self.previous = self.next_target_ns = None
        self.vr_fields = None
        if self.vr_out:
            self.vr_out.invalidate()
        self._invalidate_reference(ctx)

    def _invalidate_reference(self, ctx):
        self.reference = self.reference_details = None
        self.reference_metadata = {}
        for output in (self.reference_out, self.details_out):
            if output:
                output.invalidate()

    def _vr_targets(self, ctx, body, triggers):
        if not (self.vr_out or any(name.startswith("vr_") for output in
                (self.command_out, self.details_out, self.vr_out, self.planner_out) for name in output.fields)):
            self.pending_calibration = None
            return None
        if self.pending_calibration is not None:
            if self.kinematics is None:
                raise ValueError("Sonic VR calibration requires robot_urdf")
            joints = np.zeros(29)
            measured = self.pending_calibration == "measured"
            reference = "zero"
            if measured and "joints" in ctx.inputs:
                sample = self.joints_input.read()[0]
                if sample is not None and sample.fresh(ctx.now_ns, 200_000_000):
                    joints = np.asarray(sample.value).reshape(-1)
                    reference = "measured"
            self.vr.capture(body.value, self.kinematics.wrists(joints), preserve_neck=measured)
            ctx.logger.info("VR calibration complete (%s joint reference)", reference)
            self.pending_calibration = None
        if not any((self.command_out, self.details_out, self.vr_out, self.planner_out)):
            return None
        pose = self.vr.apply(body.value)
        fields = {"vr_position": pose[:, :3].reshape(-1), "vr_orientation": pose[:, 3:].reshape(-1)}
        for i, side in enumerate(("left", "right")):
            if side in self.hands:
                fields[f"{side}_hand_joints"] = hand_targets(triggers[i], self.hands[side])
        if self.vr_out:
            self.vr_out.write(fields, source_ns=body.source_ns, clock=body.clock)
        return fields

    def _frame(self, poses):
        from scipy.spatial.transform import Rotation as R
        basis = R.from_quat(PICO_WORLD_TO_SMPL_XYZW)
        offsets = R.from_quat(PICO_JOINT_OFFSETS_XYZW)
        global_rot = basis * R.from_quat(poses[:22, 3:]) * offsets
        local = (global_rot[list(SMPL_PARENTS[1:])].inv() * global_rot[1:]).as_rotvec()
        orientation = global_rot[0]
        body = orientation * R.from_quat([.5, .5, .5, .5]).inv()
        frame = {"pose": local.astype(np.float32), "quat": body.as_quat()[[3, 0, 1, 2]].astype(np.float32)}
        if self.skeleton:
            angles = np.zeros((len(self.skeleton["parents"]), 3))
            angles[0], angles[1:22] = orientation.as_rotvec(), local
            world = forward_kinematics(self.skeleton["rest_joints"], self.skeleton["parents"], angles)[self.skeleton["output_indices"]]
            frame["joints"] = body.inv().apply(world).astype(np.float32)
        return frame

    def _retarget(self, ctx):
        from scipy.spatial.transform import Rotation as R
        self.vr_fields = None
        body, triggers, *timing_samples = self.xr_input.read()
        age = int(self.options.get("max_age_ms", 100) * 1e6)
        if body is None or triggers is None or not body.fresh(ctx.now_ns, age) or not triggers.fresh(ctx.now_ns, age):
            ctx.report("WAITING", "XR: waiting for fresh body and trigger data")
            self._invalidate_reference(ctx)
            if self.vr_out:
                self.vr_out.invalidate()
            return
        if any(sample is None or not sample.fresh(ctx.now_ns, age)
               or (sample.batch_id, sample.source_ns, sample.clock) != (body.batch_id, body.source_ns, body.clock)
               for sample in timing_samples):
            self._invalidate_reference(ctx)
            if self.vr_out:
                self.vr_out.invalidate()
            ctx.report("DEGRADED", "XR body and timing must come from one frame")
            return
        timing = {name: sample.value for name, sample in zip(self.timing_fields, timing_samples)}
        poses = np.asarray(body.value)
        if poses.shape != (24, 7) or not np.isfinite(poses).all() or np.any(np.linalg.norm(poses[:, 3:], axis=1) < 1e-8) or np.asarray(triggers.value).shape != (2,) or not np.isfinite(triggers.value).all():
            self._invalidate_reference(ctx)
            if self.vr_out:
                self.vr_out.invalidate()
            ctx.report("DEGRADED", "Invalid XR body or trigger tensors")
            return
        controls = self.controls_in.read(max_age_ns=age)
        values = np.clip(triggers.value, 0, 1)
        vr = self.vr_fields = self._vr_targets(ctx, body, values)
        mode = self.mode_in.read(max_age_ns=age)
        if "value" in mode:
            mode["name"] = StreamMode(int(mode["value"])).name
        if self.mode_in and mode.get("name") != "POSE":
            self._invalidate_reference(ctx)
            return
        if body.sequence == self.last:
            return
        self.last = body.sequence
        frame = self._frame(poses)
        stamp = body.source_ns if body.source_ns is not None else body.received_ns
        if self.interpolate:
            if self.previous is None:
                self.previous, self.next_target_ns = (stamp, frame), stamp
                return
            previous_stamp, previous = self.previous
            if stamp <= previous_stamp:
                return
            self.next_target_ns = max(self.next_target_ns, previous_stamp)
            if self.next_target_ns > stamp:
                return
            alpha = float(np.clip((self.next_target_ns - previous_stamp) / (stamp - previous_stamp), 0, 1))
            current = frame
            frame = {"pose": pose_lerp(previous["pose"], current["pose"], alpha).astype(np.float32),
                     "quat": quat_lerp(previous["quat"], current["quat"], alpha).astype(np.float32),
                     "joints": ((1 - alpha) * previous["joints"] + alpha * current["joints"]).astype(np.float32)}
            self.previous = stamp, current
            self.next_target_ns += self.action_period_ns
        if self.options.get("hand_delta_control", False):
            grips = controls.get("grips", np.zeros(2))
            increment = np.where(values > 0, 1., np.where(np.asarray(grips) > 0, -1., 0.)) * self.options.get("hand_movement_step", .05)
            self.hand_position = np.clip(self.hand_position + increment, 0, 1)
            hand_values = self.hand_position
        else:
            hand_values = values
        orientation = R.from_quat(frame["quat"][[1, 2, 3, 0]]) * R.from_quat([.5, .5, .5, .5])
        action = np.concatenate([orientation.as_matrix()[:, :2].reshape(-1), frame["pose"].reshape(-1), hand_values]).astype(np.float32)
        command_values = {self.options.get("action_field", "action"): action}
        self.reference_frame, self.reference_hands = frame, hand_values.copy()
        step = self.sequence
        self.sequence += 1
        deadline = min(body.received_ns, triggers.received_ns) + age
        self.reference_details = None
        self.reference_metadata = {"batch_id": str(uuid.uuid4()), "source_ns": stamp, "clock": body.clock}
        if self.details_out or self.command_out:
            details = {"neck_pose": neck_targets(poses), "left_trigger": np.asarray([values[0]], np.float32),
                       "right_trigger": np.asarray([values[1]], np.float32),
                       "left_grip": np.asarray([controls.get("grips", [0, 0])[0]], np.float32),
                       "right_grip": np.asarray([controls.get("grips", [0, 0])[1]], np.float32),
                       "pico_dt": np.asarray([timing.get("dt", 0.)], np.float32),
                       "pico_fps": np.asarray([timing.get("fps", 0.)], np.float32),
                       "timestamp_realtime": np.asarray([timing.get("realtime", 0.)], np.float64),
                       "timestamp_monotonic": np.asarray([timing.get("monotonic", ctx.now_ns / 1e9)], np.float64),
                       "heading_increment": np.asarray([mode.get("heading_increment", 0.)], np.float32),
                       "toggle_data_collection": np.asarray([mode.get("toggle_record", False)], bool),
                       "toggle_data_abort": np.asarray([mode.get("abort_record", False)], bool)}
            if self.details_out:
                details.update({"body_poses_np": poses[:, :3].copy(),
                                "smpl_pose": frame["pose"], "body_quat_w": frame["quat"],
                                "joint_pos": wrist_targets(frame["pose"])})
            for side, trigger in zip(("left", "right"), hand_values):
                name = side + "_hand_joints"
                if side in self.hands and (name in self.details_out.fields or name in self.command_out.fields):
                    details[name] = hand_targets(trigger, self.hands[side])
            if "joints" in frame:
                details["smpl_joints"] = frame["joints"]
            if vr:
                details.update({key: value for key, value in vr.items() if key.startswith("vr_")})
            self.reference_details = details
            if self.details_out:
                command = RobotCommand(self.session, self.sequence, self.reference_spec_id, deadline, details, "teleop", step)
                self.details_out.write(command.values, spec_id=command.spec_id,
                                       command=command_metadata(command), **self.reference_metadata)
        self.reference = RobotCommand(self.session, self.sequence, self.reference_spec_id, deadline, command_values, "teleop", step)
        if self.reference_out:
            self.reference_out.write(self.reference.values, spec_id=self.reference.spec_id,
                                     command=command_metadata(self.reference), **self.reference_metadata)

    def _start_planner(self, ctx):
        planner_hz = float(self.options.get("planner_hz", 20))
        if not np.isfinite(planner_hz) or planner_hz <= 0 or round(1e9 / planner_hz) < 1:
            raise ValueError("Planner rate must be positive and representable in nanoseconds")
        self.planner_period_ns = round(1e9 / planner_hz)
        self.planner_session, self.planner_sequence = str(uuid.uuid4()), 0
        self._reset_planner(ctx)

    def _reset_planner(self, ctx):
        self.planner_mode = StreamMode.OFF
        self.yaw = 0.
        self.locomotion = 0
        self.ab_previous = self.xy_previous = False
        self.next_planner_ns, self.last_body = ctx.now_ns, None
        self.frozen = {}
        self.planner_out.invalidate()

    def _prepare_planner(self, ctx):
        mode = self.mode_in.read(max_age_ns=100_000_000)
        controls = self.controls_in.read(max_age_ns=200_000_000)
        if not mode or not controls:
            self.planner_out.invalidate()
            return
        target = StreamMode(int(mode["value"]))
        if target != self.planner_mode:
            if target == StreamMode.PLANNER_FROZEN_UPPER_BODY:
                self._freeze(ctx)
            if target in {StreamMode.PLANNER, StreamMode.PLANNER_FROZEN_UPPER_BODY} and self.planner_mode != StreamMode.PLANNER_VR_3PT:
                self.yaw = 0.
            self.next_planner_ns, self.last_body = ctx.now_ns, None
            self.planner_mode = target
        self.current_controls = controls
        self._write_planner(ctx)

    def _planner_command(self, ctx, output, kind, values):
        self.planner_sequence += 1
        command = RobotCommand(self.planner_session, self.planner_sequence, f"sonic.{kind}.v1",
                               ctx.now_ns + 200_000_000, values, "operator", self.planner_sequence)
        output.write(values, spec_id=command.spec_id, command=command_metadata(command))
        ctx.report("OK", f"{self.planner_mode.name}: publishing planner commands")

    def _freeze(self, ctx):
        self.frozen = {}
        for (port, field), sample in zip(self.freeze_fields, self.freeze_input.read()):
            if sample is not None and sample.fresh(ctx.now_ns, 200_000_000):
                values = np.asarray(sample.value, np.float32).reshape(-1)
                if port == "joints":
                    indices = self.options.get("upper_body_indices", [12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28])
                    values = values[indices]
                self.frozen[field] = values.copy()

    def _write_planner(self, ctx):
        controls = self.current_controls
        a, b, x, y = controls["buttons"]
        planner_mode = self.planner_mode in {StreamMode.PLANNER, StreamMode.PLANNER_FROZEN_UPPER_BODY, StreamMode.PLANNER_VR_3PT}
        body = self.body_input.read()[0] if "body" in ctx.inputs else None
        fresh = controls.get("tracking", True) and ("body" not in ctx.inputs or (body is not None and body.fresh(ctx.now_ns, 100_000_000)))
        if self.planner_out and planner_mode and fresh and ctx.now_ns >= self.next_planner_ns and (body is None or body.sequence != self.last_body):
            ab, xy = bool(a and b), bool(x and y)
            if ab and not self.ab_previous:
                self.locomotion = min(19, self.locomotion + 1)
            if xy and not self.xy_previous:
                self.locomotion = max(0, self.locomotion - 1)
            self.ab_previous, self.xy_previous = ab, xy
            dt = 1. / self.options.get("planner_hz", 20)
            fields, self.yaw = planner_fields(controls["axes"], self.locomotion, self.yaw, dt=dt)
            if self.planner_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY:
                fields.update(self.frozen)
            if self.planner_mode == StreamMode.PLANNER_VR_3PT:
                if self.vr_fields is None:
                    self.planner_out.invalidate()
                    ctx.report("DEGRADED", "Waiting for calibrated VR targets")
                    return
                fields.update(self.vr_fields)
            self._planner_command(ctx, self.planner_out, "planner", fields)
            self.next_planner_ns += self.planner_period_ns
            if self.next_planner_ns <= ctx.now_ns:
                # Skip missed updates after a stall; do not burst commands or
                # integrate the current stick over the missing time.
                self.next_planner_ns = ctx.now_ns + self.planner_period_ns
            self.last_body = body.sequence if body else None
        elif not planner_mode or not fresh:
            self.planner_out.invalidate()


def planner_fields(axes, mode, yaw, *, dt=.05, yaw_gain=1.5, deadzone=.15):
    """Original joystick-to-Sonic planner mapping, without a socket or SDK."""
    lx, ly, rx, _ = np.asarray(axes, np.float64)
    if abs(rx) >= deadzone:
        yaw += yaw_gain * -rx * dt
    facing = np.array([np.cos(yaw), np.sin(yaw), 0.])
    raw = float(np.clip(np.hypot(lx, ly), 0., 1.))
    if raw < deadzone:
        magnitude, speed, selected_mode = 0., -1., 0
    else:
        magnitude = min(1., (raw - deadzone) / (1. - deadzone))
        selected_mode = mode
        speed = .1 + .5 * magnitude if mode == 1 else -1. if mode == 2 else 1.5 + 3 * magnitude if mode == 3 else magnitude
    movement_local = np.array([-lx, ly]) * magnitude / (raw or 1.)
    xy = np.array([[-facing[1], facing[0]], [facing[0], facing[1]]]) @ movement_local
    return {"mode": np.array([selected_mode], np.int32), "movement": np.asarray([*xy, 0.], np.float32),
            "facing": facing.astype(np.float32), "speed": np.array([speed], np.float32),
            "height": np.array([-1.], np.float32)}, yaw
