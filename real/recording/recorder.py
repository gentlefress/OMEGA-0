"""Record configured runtime inputs as episodes."""
from __future__ import annotations

from collections.abc import Mapping
import shutil

import numpy as np

from ..core.runtime import Module
from .episode import EpisodeWriter, hdf5_paths, video_files
from .images import image_directories
from .tensor_episode import EpisodeWriter as TensorEpisodeWriter


class Recorder(Module):
    queued_inputs = True

    def configure(self, options):
        super().configure(options)
        if {"hdf5", "video_inputs"} & self.options.keys():
            raise ValueError("Use options.layout.hdf5, options.layout.videos and options.layout.images for recording destinations")
        self.format = self.options.get("format", "state_action")
        if self.format not in {"state_action", "tensor"}:
            raise ValueError("Recording format must be state_action or tensor")
        self.sampling = self.options.get("sampling", "latest" if self.format == "state_action" else "all")
        if self.sampling not in {"all", "latest"}:
            raise ValueError("Recorder sampling must be 'all' or 'latest'")
        if self.format == "state_action" and self.sampling != "latest":
            raise ValueError("State/action recordings require sampling: latest")
        self.queued_inputs = self.sampling == "all"

    def bind(self, ctx):
        layout = self.options.get("layout", {})
        if not isinstance(layout, Mapping) or set(layout) - {"hdf5", "videos", "images"}:
            raise ValueError("Recorder layout supports only hdf5, videos and images mappings")
        hdf5 = hdf5_paths(layout.get("hdf5"))
        videos = video_files(layout.get("videos"))
        images = image_directories(layout.get("images"))
        unknown = (set(hdf5.values()) | set(videos.values()) | set(images.values())) - ctx.inputs.keys()
        if unknown:
            raise ValueError(f"Recording layout references unknown recorder inputs: {sorted(unknown)}")
        self.hdf5 = hdf5_paths({path: ctx.inputs[port] for path, port in hdf5.items()})
        self.video_files = video_files({name: ctx.inputs[port] for name, port in videos.items()})
        if set(self.hdf5.values()) & set(self.video_files.values()):
            raise ValueError("An input cannot appear in both layout.hdf5 and layout.videos")
        self.image_directories = image_directories({name: ctx.inputs[port] for name, port in images.items()})
        if set(self.image_directories.values()) & (set(self.hdf5.values()) | set(self.video_files.values())):
            raise ValueError('An input cannot appear in layout.images and another destination')
        self.status_output = ctx.bind_write(['status'] if 'status' in ctx.outputs else [])
        self.input = ctx.bind_read(ctx.inputs)

    def start(self, ctx):
        self.writer = None
        self.last_path = None
        self.drop_baseline = {}
        if self.options.get("auto_start", True):
            self._begin(ctx)
        else:
            ctx.report("IDLE", "Recorder ready; waiting for recording start")

    def _begin(self, ctx):
        if self.writer:
            return
        if self.queued_inputs:
            self.input.drain()
        else:
            # Recording has its own cache. Holding values here never makes an
            # invalid command/input valid again in the live runtime.
            self.latest_samples = [sample if sample is not None and sample.valid else None
                                   for sample in self.input.read()]
            self.recorded_samples = [None] * len(self.latest_samples)
            self.missing_ticks = {}
            self.missing_interval = None
        self.frame_index = 0
        self.drop_baseline = ctx.dropped_inputs
        self.event_drop_baseline = ctx.dropped_events
        metadata = dict(self.options.get("metadata", {}))
        configuration = ctx.configuration
        if configuration is not None:
            metadata["runtime_config"] = configuration
        metadata["recording_inputs"] = dict(ctx.inputs)
        writer_type = EpisodeWriter if self.format == "state_action" else TensorEpisodeWriter
        self.writer = writer_type(self.options.get("root", "artifacts/episodes"),
                                    metadata=metadata, videos=self.video_files,
                                    fps=self.options.get("video_fps", 30), hdf5=self.hdf5, images=self.image_directories)
        self.writer.manifest["sampling"] = self.sampling
        if not self.queued_inputs:
            self.writer.manifest["missing_input_policy"] = "hold_last_valid"
        ctx.report("WAITING", f"Recording started: {self.writer.path}; waiting for inputs")
        if "status" in ctx.outputs:
            self.status_output.write((np.asarray(True),))
        ctx.emit("recording_status", {"recording": True, "path": str(self.writer.path)})

    def _finish(self, ctx, *, complete=True, reason=""):
        if self.writer is None:
            return
        if self.queued_inputs:
            self.process(ctx)
        else:
            self._close_missing_interval(ctx)
        gaps = {ctx.inputs[p]: count - self.drop_baseline.get(p, 0) for p, count in ctx.dropped_inputs.items()}
        if not self.queued_inputs:
            gaps.update(self.missing_ticks)
        gaps["__events__"] = ctx.dropped_events - self.event_drop_baseline
        writer, self.writer = self.writer, None
        self.last_path = writer.close(complete=complete, gaps=gaps, reason=reason)
        ctx.report("IDLE", f"Recording {writer.manifest['status']}: {self.last_path}")
        if "status" in ctx.outputs:
            self.status_output.write((np.asarray(False),))
        ctx.emit("recording_status", {"recording": False, "path": str(self.last_path), "status": writer.manifest["status"]})

    def process(self, ctx):
        if not self.queued_inputs:
            if self.writer is None:
                return
            current = self.input.read()
            for index, sample in enumerate(current):
                if sample is not None and sample.valid:
                    self.latest_samples[index] = sample
            missing = tuple(stream for stream, sample in zip(ctx.inputs.values(), self.latest_samples) if sample is None)
            if missing:
                for stream in (*missing, "__samples__"):
                    self.missing_ticks[stream] = self.missing_ticks.get(stream, 0) + 1
                if self.missing_interval is None or self.missing_interval["streams"] != missing:
                    self._close_missing_interval(ctx)
                    self.missing_interval = {"start_ns": ctx.now_ns, "streams": missing, "ticks": 0}
                self.missing_interval["ticks"] += 1
                ctx.report("DEGRADED", "Recording waiting for initial values: " + ", ".join(missing))
                return
            self._close_missing_interval(ctx)
            if not self.latest_samples:
                return
            self.frame_index += 1
            timestamp = ctx.now_ns
            held = {sample.stream for sample, observed, previous in zip(self.latest_samples, current, self.recorded_samples)
                    if sample is previous or observed is None or not observed.valid}
            if self.format == "state_action":
                self.writer.append_row({sample.stream:sample for sample in self.latest_samples},
                                       timestamp_ns=timestamp, held=held)
            else:
                for sample in self.latest_samples:
                    self.writer.append(sample, recorded_ns=timestamp, recorded_sequence=self.frame_index,
                                       held=sample.stream in held)
            self.recorded_samples = list(self.latest_samples)
            ctx.report("OK", f"Recording to {self.writer.path}")
            return
        samples = []
        for batch in self.input.drain():
            samples.extend(batch)
        if self.writer is not None:
            for sample in sorted(samples, key=lambda x: (x.received_ns, x.stream, x.sequence)):
                self.writer.append(sample)
            dropped = sum(count - self.drop_baseline.get(p, 0) for p, count in ctx.dropped_inputs.items())
            if dropped:
                ctx.report("DEGRADED", f"Recording queue dropped {dropped} samples")
            elif samples:
                ctx.report("OK", f"Recording to {self.writer.path}")

    def log_status(self, ctx):
        writer = getattr(self, 'writer', None)
        if writer is None or ctx.runtime.health[ctx.id].level != 'OK':
            return super().log_status(ctx)
        if self.sampling == 'latest':
            progress = f"{self.frame_index} frames"
        else:
            progress = f"{sum(entry['count'] for entry in tuple(writer.manifest['streams'].values()))} samples"
        ctx.log_status(f"Recording {progress} to {writer.path}")

    def _close_missing_interval(self, ctx):
        if self.missing_interval is not None:
            self.writer.event("recording_gap", ctx.now_ns, {**self.missing_interval, "end_ns": ctx.now_ns})
            self.missing_interval = None

    def on_event(self, ctx, event):
        if self.writer is not None:
            self.writer.event(event.kind, ctx.now_ns, event.payload)
        if event.kind == "record_start":
            self._begin(ctx)
        elif event.kind == "record_toggle":
            self._finish(ctx) if self.writer else self._begin(ctx)
        elif event.kind == "record_stop":
            self._finish(ctx)
        elif event.kind == "record_abort":
            active = self.writer is not None
            self._finish(ctx, complete=False, reason=event.kind)
            if active and self.options.get("abort", "retain") == "discard":
                # Only delete the episode this recorder just finalized, never a
                # supplied path or a previous successfully recorded episode.
                shutil.rmtree(self.last_path)
                ctx.logger.info("Discarded aborted episode: %s", self.last_path)
                if "status" in ctx.outputs:
                    self.status_output.write((np.asarray(False),))
                ctx.emit("recording_status", {"recording": False, "status": "discarded"})
        elif event.kind == "estop":
            self._finish(ctx, complete=False, reason=event.kind)
        elif event.kind in {"reset", "discontinuity"} and self.writer is not None and not self.queued_inputs:
            # An explicit runtime/session boundary is not a missing sensor update.
            self.latest_samples = [None] * len(self.latest_samples)
            self.recorded_samples = [None] * len(self.recorded_samples)

    def stop(self, ctx):
        faults = ctx.faults
        self._finish(ctx, complete=not bool(faults), reason=str(faults) if faults else "")
