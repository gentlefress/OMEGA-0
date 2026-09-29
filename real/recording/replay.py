"""Replay saved episode streams into the runtime."""
from __future__ import annotations

import bisect
from dataclasses import replace
import uuid

from ..core.runtime import Module
from .episode import EpisodeReader


class Replay(Module):
    """Discover recorded fields under the module ID; start through an event."""
    event_actions = {"start", "pause", "resume", "seek"}

    def configure(self, options):
        super().configure(options)
        removed = {"namespace", "streams", "wait_for_start", "start_event", "stop_when_finished"} & options.keys()
        if removed:
            raise ValueError(f"Replay uses its module ID and event bindings; remove {sorted(removed)}")
        self.max_fields = int(self.options.get("max_fields", 4096))
        if self.max_fields <= 0:
            raise ValueError("Replay max_fields must be positive")

    def output_namespaces(self):
        return {self.id + ".": self.max_fields}

    def bind(self, ctx):
        if ctx.outputs:
            raise ValueError("Replay discovers episode fields; remove explicit outputs")

    def start(self, ctx):
        self.reader = EpisodeReader(self.options["episode"], allow_incomplete=self.options.get("allow_incomplete", False))
        self.streams = {}
        for name in self.reader.manifest["streams"]:
            port = name.replace("/", ".")
            if port in self.streams:
                raise ValueError(f"Recorded names collide after slash-to-dot conversion: {name}")
            self.streams[port] = name
        if len(self.streams) > self.max_fields:
            raise ValueError("Replay field limit exceeded")
        self.output = ctx.bind_write(self.streams, namespace=ctx.id)
        ctx.logger.info("Publishing %d recorded streams under %s.*", len(self.streams), ctx.id)
        self.selected_groups = {self.reader.manifest['streams'][stream].get('group', stream)
                                for stream in self.streams.values()}
        self.speed = float(self.options.get("speed", 1))
        if self.speed <= 0:
            raise ValueError("Replay speed must be positive")
        # File publication groups are discovered after opening the episode.
        self.publications = self.reader.publication_groups()
        self.timestamps = [self.reader.index[indices[0]][0] for indices in self.publications]
        self.publication_ports = {}
        self.index = 0
        self.recorded_start = self.timestamps[0] if self.timestamps else 0
        self.origin = ctx.now_ns
        self.offset = 0
        self.waiting_for_start = True
        self.paused_at = self.origin
        self.finished = False
        self.sessions = {}
        self.command_sequences = {}
        self.batches = {}
        ctx.report("READY", f"Replay loaded: {self.reader.path}; {len(self.publications)} publications at {self.speed:g}x")
        ctx.report("IDLE", "Episode loaded; waiting for a configured start event")

    def _metadata(self, sample, ctx):
        metadata = {"spec_id": sample.spec_id}
        if sample.batch_id:
            metadata["batch_id"] = self.batches.setdefault(sample.batch_id, str(uuid.uuid4()))
        if sample.command is not None:
            value = sample.command
            due_ns = self.origin + int((sample.received_ns - self.recorded_start - self.offset) / self.speed)
            lifetime = int((value.valid_until_ns - sample.received_ns) / self.speed)
            session = self.sessions.setdefault(value.session_id, str(uuid.uuid4()))
            identities = self.command_sequences.setdefault(session, {})
            sequence = identities.setdefault(value.sequence, len(identities) + 1)
            metadata["command"] = replace(value, session_id=session, sequence=sequence,
                                            valid_until_ns=max(0, due_ns + lifetime))
        if sample.prediction is not None:
            value = sample.prediction
            session = self.sessions.setdefault(value.session_id, str(uuid.uuid4()))
            request = self.sessions.setdefault((value.session_id, value.request_id), str(uuid.uuid4()))
            metadata["prediction"] = replace(value, session_id=session, request_id=request)
        return metadata

    def on_event(self, ctx, event):
        if event.source == ctx.id and event.kind == "discontinuity":
            return
        if self.waiting_for_start and event.kind == "start":
            self.origin += ctx.now_ns - self.paused_at
            self.paused_at = None
            self.waiting_for_start = False
            ctx.report("OK", "Replay started")
        elif event.kind in {"pause", "estop", "shutdown", "reset", "discontinuity"} and self.paused_at is None:
            self.paused_at = ctx.now_ns
            ctx.report("IDLE", "Replay paused")
        elif event.kind == "resume" and self.paused_at is not None:
            self.origin += ctx.now_ns - self.paused_at
            self.paused_at = None
            self.waiting_for_start = False
            ctx.report("OK", "Replay resumed")
        elif event.kind == "seek":
            self.offset = int(float(event.payload) * 1e9)
            if self.offset < 0:
                raise ValueError("Seek time cannot be negative")
            self.index = bisect.bisect_left(self.timestamps, self.recorded_start + self.offset)
            self.origin = ctx.now_ns
            if self.paused_at is not None:
                self.paused_at = self.origin
            self.finished = False
            self.sessions.clear()
            self.command_sequences.clear()
            self.batches.clear()
            self.output.write((None,) * len(self.output))
            ctx.emit("discontinuity", {"replay": ctx.id, "offset_ns": self.offset})

    def process(self, ctx):
        # A seek emits a discontinuity while handling this cycle's events. Wait
        # for the next cycle to capture its generation before consuming samples.
        if ctx.write_generation != ctx.generation or self.paused_at is not None or self.finished:
            return
        due = self.recorded_start + self.offset + int((ctx.now_ns - self.origin) * self.speed)
        for _ in range(int(self.options.get("max_batch", 128))):
            if self.index >= len(self.timestamps) or self.timestamps[self.index] > due:
                break
            samples = {sample.stream: sample for sample in
                       (self.reader.sample(index) for index in self.publications[self.index]
                        if self.reader.index[index][1] in self.selected_groups)}
            self.index += 1
            ports = tuple(port for port, stream in self.streams.items() if stream in samples)
            if not ports:
                continue
            output = self.publication_ports.get(ports)
            if output is None:
                output = self.publication_ports[ports] = ctx.bind_write(ports, namespace=ctx.id)
            selected = [samples[self.streams[port]] for port in ports]
            output.write(tuple(sample.value if sample.valid else None for sample in selected),
                         field_metadata=[dict(source_ns=sample.source_ns, clock=sample.clock,
                                              **self._metadata(sample, ctx)) for sample in selected])
        if self.index == len(self.timestamps):
            self.finished = True
            ctx.report("IDLE", "Replay finished")
            ctx.emit("finished")

    def stop(self, ctx):
        if getattr(self, "reader", None):
            self.reader.close()
