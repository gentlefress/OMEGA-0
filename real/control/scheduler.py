"""Chunk scheduling helpers owned by inference clients."""
from __future__ import annotations

import uuid
import numpy as np

from ..core.protocol import ActionChunk, RobotCommand


class ChunkSchedule:
    """Execution state owned by a chunk-based policy; no runtime dependencies."""

    def __init__(self, spec, now_ns):
        self.spec = spec
        self.session_id = str(uuid.uuid4())
        self.origin_ns = now_ns
        self.chunk = None
        self.last_step = -1
        self.command_sequence = 0

    def step(self, now_ns):
        return (now_ns - self.origin_ns) // self.spec.period_ns

    def accept(self, chunk, now_ns):
        if not isinstance(chunk, ActionChunk):
            raise TypeError("Chunk schedule expects ActionChunk")
        chunk.validate(self.spec)
        if chunk.session_id != self.session_id or chunk.start_step + chunk.horizon <= self.step(now_ns):
            return False
        if self.chunk is not None and chunk.start_step < self.chunk.start_step:
            return False
        self.chunk = chunk
        return True

    def remaining(self, now_ns):
        if self.chunk is None:
            return {}
        offset = max(0, self.step(now_ns) - self.chunk.start_step)
        if offset >= self.chunk.horizon:
            return {}
        return {k: v[offset:] for k, v in self.chunk.actions.items()}

    def available(self, now_ns):
        return self.chunk is not None and 0 <= self.step(now_ns) - self.chunk.start_step < self.chunk.horizon

    def command(self, now_ns):
        step = self.step(now_ns)
        if not self.available(now_ns) or step == self.last_step:
            return None
        self.last_step = step
        self.command_sequence += 1
        offset = step - self.chunk.start_step
        return RobotCommand(self.session_id, self.command_sequence, self.spec.id, self.origin_ns + (step + 1) * self.spec.period_ns,
                            {k: v[offset] for k, v in self.chunk.actions.items()}, "policy", step)


class RtcChunkSchedule:
    """Real-time chunk execution with RTC prefix preservation and overlap blending.

    Execution begins with action zero when the first response arrives. Inference
    does not advance the execution cursor; commands advance it at the control rate.
    The unblended prediction is retained separately for the next model's RTC input.
    """

    def __init__(self, spec, now_ns, *, interval=8, delay=8, overlap=9):
        self.spec, self.interval, self.delay, self.overlap = spec, int(interval), int(delay), int(overlap)
        if self.interval <= 0 or min(self.delay, self.overlap) < 0:
            raise ValueError("Invalid RTC interval/delay/overlap")
        self.session_id = str(uuid.uuid4())
        self.chunk = self.prediction = None
        self.action_index = self.command_sequence = 0
        self.next_due_ns = now_ns

    def step(self, now_ns):
        return self.command_sequence

    def ready_for_request(self):
        return self.chunk is None or self.action_index >= self.interval

    def remaining(self, now_ns):
        if self.prediction is None or self.prediction.horizon <= self.interval:
            return {}
        return {key: values[self.interval:] for key, values in self.prediction.actions.items()}

    def accept(self, chunk, now_ns):
        chunk.validate(self.spec)
        if chunk.session_id != self.session_id:
            return False
        # Keep an exhausted buffer until the next control tick. A response in
        # that interval still merges, as in the original locked control loop.
        if self.chunk is not None:
            required = self.interval + self.delay + self.overlap
            if self.chunk.horizon < required or chunk.horizon < self.delay + self.overlap:
                raise ValueError("RTC horizon is shorter than prefix plus overlap")
            actions = {}
            for key, incoming in chunk.actions.items():
                old = self.chunk.actions[key]
                prefix = old[self.interval:self.interval + self.delay]
                previous = old[self.interval + self.delay:required]
                next_overlap = incoming[self.delay:self.delay + self.overlap]
                shape = (self.overlap,) + (1,) * (incoming.ndim - 1)
                alpha = ((np.arange(self.overlap, dtype=np.float32) + 1) / (self.overlap + 1)).reshape(shape)
                blended = previous * (1 - alpha) + next_overlap * alpha
                actions[key] = np.concatenate([prefix, blended, incoming[self.delay + self.overlap:]], axis=0).astype(incoming.dtype)
            self.action_index = max(0, self.action_index - self.interval)
            self.chunk = ActionChunk(chunk.session_id, chunk.request_id, chunk.observation_id,
                                     chunk.spec_id, chunk.start_step, chunk.period_ns, actions)
        else:
            self.chunk, self.action_index = chunk, 0
            self.next_due_ns = now_ns
        self.prediction = chunk
        return True

    def available(self, now_ns):
        return self.chunk is not None and self.action_index < self.chunk.horizon

    def command(self, now_ns):
        if now_ns < self.next_due_ns:
            return None
        if self.chunk is not None and self.action_index >= self.chunk.horizon:
            self.chunk, self.action_index = None, 0
        if self.chunk is None:
            return None
        values = {key: value[self.action_index] for key, value in self.chunk.actions.items()}
        step = self.command_sequence
        self.action_index += 1
        self.command_sequence += 1
        # Keep the cadence when this module is polled faster than the action
        # rate (e.g. 100 Hz polling for 30 Hz actions); do not round every period
        # up to the next poll. Long stalls still produce only one command.
        self.next_due_ns = max(self.next_due_ns + self.spec.period_ns, now_ns + 1)
        return RobotCommand(self.session_id, self.command_sequence, self.spec.id, now_ns + self.spec.period_ns, values, "policy", step)
