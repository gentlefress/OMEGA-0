"""Raw XR acquisition; retargeting and recording are separate modules."""
import numpy as np
import time

from ..core.streams import FieldWriter
from ..core.runtime import Module


class XRSource(Module):
    required_outputs = ("body",)

    def bind(self, ctx):
        self.frame_fields = tuple(port for port in ctx.outputs if port == 'body' or port.startswith('timing.'))
        self.frame_output = ctx.bind_write(self.frame_fields)
        self.triggers_output = ctx.bind_write(['triggers'] if 'triggers' in ctx.outputs else [])
        self.controls_out = FieldWriter(ctx, 'controls')

    def start(self, ctx):
        allowed = {"body", "triggers"} | {"controls." + name for name in
            ("buttons", "grips", "axes", "menus", "clicks", "tracking")} | {"timing." + name for name in
            ("dt", "fps", "realtime", "monotonic")}
        if not ctx.outputs.keys() <= allowed:
            raise ValueError("XR outputs must name individual controls/timing tensor fields")
        import xrobotoolkit_sdk
        self.sdk = xrobotoolkit_sdk
        self.sdk.init()
        self.last_timestamp = None
        self.last_frame_ns = ctx.now_ns
        self.fps = 0.
        ctx.report("WAITING", "XR SDK initialized; waiting for Pico body tracking")

    def _read(self, name, default):
        try:
            method = getattr(self.sdk, name, None)
            return method() if method is not None else default
        except (AttributeError, NotImplementedError, RuntimeError):
            # SDK builds expose optional buttons/axes with unsupported stubs.
            return default

    def process(self, ctx):
        available = bool(self.sdk.is_body_data_available())
        triggers = np.asarray([self.sdk.get_left_trigger(), self.sdk.get_right_trigger()], dtype=np.float32)
        # Controls must remain available for stop/recording even when body tracking
        # stalls or the body timestamp does not advance.
        if self.controls_out:
            controls = {
                "buttons": np.asarray([self._read(f"get_{key}_button", False) for key in "ABXY"], bool),
                "grips": np.asarray([self._read(f"get_{side}_grip", 0.) for side in ("left", "right")], np.float32),
                "axes": np.asarray([*self._read("get_left_axis", [0., 0.]), *self._read("get_right_axis", [0., 0.])], np.float32),
                "menus": np.asarray([self._read(f"get_{side}_menu_button", False) for side in ("left", "right")], bool),
                "clicks": np.asarray([self._read(f"get_{side}_axis_click", False) for side in ("left", "right")], bool),
                "tracking": available,
            }
            self.controls_out.write(controls, source_ns=ctx.now_ns, clock="host.receive")
        if "triggers" in ctx.outputs:
            self.triggers_output.write((triggers,), source_ns=ctx.now_ns, clock="host.receive")
        if not available:
            self.frame_output.write((None,) * len(self.frame_output))
            ctx.report("DEGRADED", "XR body tracking unavailable")
            return
        stamp = int(self.sdk.get_time_stamp_ns())
        if stamp == self.last_timestamp:
            if ctx.now_ns - self.last_frame_ns > 500_000_000:
                ctx.report("DEGRADED", "Pico body frame timestamp stopped advancing")
            return
        dt = (stamp - self.last_timestamp) * 1e-9 if self.last_timestamp is not None else 0.
        if dt > 0:
            self.fps = 1. / dt if self.fps == 0 else .9 * self.fps + .1 / dt
        self.last_timestamp = stamp
        self.last_frame_ns = ctx.now_ns
        frame = {"body": np.asarray(self.sdk.get_body_joints_pose(), dtype=np.float32),
                 "timing.dt": dt, "timing.fps": self.fps,
                 "timing.realtime": time.time(), "timing.monotonic": ctx.now_ns / 1e9}
        self.frame_output.write(tuple(np.asarray(frame[name]) for name in self.frame_fields), source_ns=stamp, clock="xr")
        ctx.report("OK", "Receiving Pico body tracking and controls")

    def log_status(self, ctx):
        if getattr(self, 'sdk', None) is not None and ctx.runtime.health[ctx.id].level == 'OK':
            ctx.log_status(f"Receiving Pico body tracking and controls ({self.fps:.1f} Hz)")
        else:
            super().log_status(ctx)

    def stop(self, ctx):
        sdk, self.sdk = getattr(self, "sdk", None), None
        if sdk is not None:
            close = getattr(sdk, "close", None) or getattr(sdk, "shutdown", None)
            if close is not None:
                close()
