"""Visualize live RGB inputs in Rerun without queuing old frames."""
from __future__ import annotations

from pathlib import Path
import os
import sysconfig
from uuid import uuid4

import numpy as np

from .core.runtime import Module


class RgbVisualizer(Module):
    """Display a uint8 HWC RGB stream; optionally connect or save an RRD file."""

    required_inputs = ("rgb",)

    def configure(self, options):
        super().configure(options)
        self.mode = self.options.get("mode", "spawn")
        if self.mode not in {"spawn", "connect", "save"}:
            raise ValueError("Visualizer mode must be spawn, connect, or save")
        if self.mode == "save" and not self.options.get("path"):
            raise ValueError("Visualizer save mode requires options.path")
        self.entity_path = self.options.get("entity_path", "inputs/rgb")
        self.recording = None

    def bind(self, ctx):
        self.input = ctx.bind_read(("rgb",))

    def start(self, ctx):
        try:
            import rerun as rr
        except ImportError as error:
            raise RuntimeError("RGB visualization requires rerun-sdk; install omega-zero[deploy]") from error
        self.rr = rr
        self.last_sample = None
        self.start_ns = ctx.now_ns
        self.recording = rr.RecordingStream(
            self.options.get("application_id", "omega-real"), recording_id=uuid4(),
        )
        try:
            if self.mode == "spawn":
                # Direct .venv/bin/python launches do not put the environment's
                # scripts on PATH. Pass its viewer explicitly to the same
                # launcher used by RecordingStream.spawn in our pinned SDK.
                viewer = Path(sysconfig.get_path("scripts")) / ("rerun.exe" if os.name == "nt" else "rerun")
                if viewer.is_file():
                    import rerun_bindings
                    rerun_bindings.spawn(executable_path=str(viewer), memory_limit="512MB")
                    self.recording.connect_grpc(flush_timeout_sec=1)
                else:
                    self.recording.spawn(memory_limit="512MB")
                destination = "Rerun viewer"
            elif self.mode == "connect":
                destination = self.options.get("url", "rerun+http://127.0.0.1:9876/proxy")
                self.recording.connect_grpc(destination, flush_timeout_sec=1)
            else:
                path = Path(self.options["path"]).expanduser()
                path.parent.mkdir(parents=True, exist_ok=True)
                self.recording.save(path)
                destination = str(path)
        except BaseException:
            self.stop(ctx)
            raise
        ctx.report("WAITING", f"RGB visualization ready: {destination}; waiting for {ctx.inputs['rgb']}")

    def process(self, ctx):
        sample, = self.input.read()
        if sample is None or not sample.valid:
            ctx.report("WAITING", "Waiting for valid RGB input")
            return
        if sample is self.last_sample:
            return
        self.last_sample = sample
        rgb = np.asarray(sample.value)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not all(rgb.shape):
            ctx.report("DEGRADED", f"Expected uint8 RGB (H, W, 3), got {rgb.dtype} {rgb.shape}")
            return
        self.recording.set_time("frame", sequence=sample.sequence)
        self.recording.set_time("elapsed", duration=(sample.received_ns - self.start_ns) / 1e9)
        self.recording.log(self.entity_path, self.rr.Image(rgb, color_model="RGB"), strict=True)
        ctx.report("OK", f"Visualizing {ctx.inputs['rgb']} ({rgb.shape[1]}x{rgb.shape[0]})")

    def stop(self, ctx):
        recording, self.recording = self.recording, None
        if recording is not None:
            try:
                recording.flush()
            finally:
                recording.disconnect()
