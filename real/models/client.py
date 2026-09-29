"""Real-world inference clients: gather, prepare, request, convert and publish."""
from __future__ import annotations

import urllib.request
import urllib.error
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace

import numpy as np

from ..core.streams import FieldWriter, command_metadata
from ..core.protocol import ActionChunk, PolicyRequest, PredictionMetadata, action_spec, dumps, loads
from ..core.runtime import Module
from ..core.streams import field_ports
from ..transforms.observation import ObservationSelector
from ..control.scheduler import ChunkSchedule, RtcChunkSchedule


class HttpPolicyClient:
    def __init__(self, endpoint, timeout=2):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = float(timeout)

    def _request(self, path, value=None):
        request = urllib.request.Request(self.endpoint + path, data=None if value is None else dumps(value), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return loads(response.read(128 * 1024 * 1024 + 1))
        except urllib.error.HTTPError as error:
            # A rejected/busy model request is not loss of the transport connection.
            message = error.read(4096).decode("utf-8", errors="replace")
            raise ValueError(f"Model server rejected request ({error.code}): {message}") from error

    def health(self):
        return self._request("/health")

    def reset(self, session_id):
        return self._request("/reset", {"session_id": session_id})

    def predict(self, request):
        result = self._request("/act", request)
        if not isinstance(result, ActionChunk):
            raise TypeError("Policy server did not return an ActionChunk")
        return result


class InferenceClient(Module):
    """Client module with aligned inputs, asynchronous HTTP and action chunks.

    preprocess() prepares sensor observations for the server API; postprocess()
    converts its response; prepare_command() converts each scheduled action for
    downstream consumers. Weights and model tensor processing belong to the server.
    """

    event_actions = {"restart", "model_toggle_pause"}

    def bind(self, ctx):
        self.view = self.options.get("view", "ego")
        if self.view not in ("ego", "exo"):
            raise ValueError("Model view must be 'ego' or 'exo'")
        self.enabled_input = ctx.bind_read(['enabled'] if 'enabled' in ctx.inputs else [])
        self.chunk_out = FieldWriter(ctx, 'chunk')
        self.command_out = FieldWriter(ctx, 'command')
        self.reference_out = FieldWriter(ctx, 'reference')
        self.selector = ObservationSelector({key: value for key, value in ctx.inputs.items() if key != "enabled"}, self.options.get("observation", {}))
        self.selector.bind(ctx)
        self.preprocessing = self.options.get("preprocessing", {})
        if not isinstance(self.preprocessing, dict) or not self.preprocessing.keys() <= set(self.selector.ports):
            raise ValueError("Client preprocessing must name bound observation inputs")
        if any(not isinstance(options, dict) or set(options) - {"crop", "size"}
               for options in self.preprocessing.values()):
            raise ValueError("Client RGB preprocessing supports crop and size")

    def __init__(self, policy=None):
        self.policy = policy
        self.executor = None
        self.pending = None

    def start(self, ctx):
        if self.policy is None:
            self.policy = HttpPolicyClient(self.options["endpoint"], self.options.get("timeout_seconds", 2))
        # The server owns the action contract. Discover it once for scheduling
        # and output validation, then recheck it whenever a session starts.
        self.spec = action_spec(self.policy.health()["action_spec"])
        self.validate_spec()
        if not set(field_ports(ctx.outputs, "command")) <= set(self.command_fields()):
            raise ValueError("Unknown model command output field")
        if not set(field_ports(ctx.outputs, "chunk")) <= set(self.spec.fields):
            raise ValueError("Unknown model chunk output field")
        if not set(self.reference_out.fields) <= set(self.spec.fields):
            raise ValueError("Unknown model reference output field")
        self.paused = False
        self.inference_interval = int(self.options.get("inference_interval", 8))
        if self.inference_interval <= 0:
            raise ValueError("inference_interval must be positive")
        self.worker_session = None
        self._reset(ctx)
        # Only HTTP inference runs separately; the runtime owns the module loop.
        # Deterministic tick execution performs inference inline.
        if ctx.runtime.threaded:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=ctx.id + "-inference")
        ctx.logger.info("Inference endpoint: %s; actions %.1f Hz, scheduling=%s", self.options.get("endpoint", "provided policy"), 1e9 / self.spec.period_ns,
            self.options.get("scheduling", "absolute"))
        ctx.report("WAITING", "Waiting for controller enable and valid observations" if 'enabled' in ctx.inputs else "Waiting for valid observations")

    def validate_spec(self):
        """Validate the discovered server contract before starting scheduling."""

    def command_fields(self):
        return self.spec.fields

    def preprocess(self, observation):
        """Optional CPU RGB crop/resize; override for another server's input API."""
        if not self.preprocessing:
            return observation
        from ..sensors.camera import prepare_rgb
        samples = dict(observation.samples)
        for name, options in self.preprocessing.items():
            if name not in samples:
                continue  # Optional observations can be absent.
            sample = samples[name]
            value = sample.value
            if not sample.valid or value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
                raise ValueError(f"Client preprocessing expects RGB uint8 HWC input: {name}")
            samples[name] = replace(sample, value=prepare_rgb(value, options))
        return replace(observation, samples=samples)

    def postprocess(self, chunk):
        """Convert a server response to the configured action representation."""
        return chunk

    def prepare_command(self, command):
        return command

    def stop(self, ctx):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        close = getattr(self.policy, "close", None)
        if close is not None:
            close()

    def _invalidate(self, ctx):
        self.command_out.invalidate()
        if self.reference_out:
            self.reference_out.invalidate()
        if self.chunk_out:
            self.chunk_out.invalidate()

    def _reset(self, ctx):
        self.generation = ctx.write_generation
        scheduling = self.options.get("scheduling", "absolute")
        if scheduling == "rtc":
            self.schedule = RtcChunkSchedule(self.spec, ctx.now_ns, interval=self.inference_interval,
                delay=self.options.get("inference_delay", 8), overlap=self.options.get("overlap", 9))
        elif scheduling == "absolute":
            self.schedule = ChunkSchedule(self.spec, ctx.now_ns)
        else:
            raise ValueError(f"Unknown chunk scheduling: {scheduling}")
        self.last_observation = None
        self.next_request_step = 0
        self._invalidate(ctx)

    def on_event(self, ctx, event):
        if event.kind in {"reset", "discontinuity", "estop", "restart"}:
            self._reset(ctx)
        elif event.kind == "model_toggle_pause":
            self.paused = not self.paused
            self._reset(ctx)

    def _predict(self, request):
        # Called only by the inference worker (or synchronous deterministic tick).
        if self.worker_session != request.session_id:
            if isinstance(self.policy, HttpPolicyClient) and self.policy.health()["action_spec"] != self.spec:
                raise ValueError("Policy server action/controller specification differs")
            self.policy.reset(request.session_id)
            self.worker_session = request.session_id
        prepared = replace(request, observation=self.preprocess(request.observation))
        chunk = self.postprocess(self.policy.predict(prepared))
        if not isinstance(chunk, ActionChunk):
            raise TypeError("Chunk policy expects ActionChunk")
        chunk.validate(self.spec)
        if (chunk.session_id, chunk.request_id, chunk.observation_id, chunk.start_step) != (
                request.session_id, request.request_id, request.observation.observation_id, request.start_step):
            raise ValueError("Policy response identity mismatch")
        return chunk

    def _collect(self, ctx, observation):
        if self.pending is None or not self.pending[0].done():
            return
        future, request, generation = self.pending
        self.pending = None
        # An earlier session may still be completing; never reuse its result.
        if generation != ctx.generation or request.session_id != self.schedule.session_id:
            return
        try:
            chunk = future.result()
            max_age = int(self.options.get("max_response_age_ms", self.selector.max_age_ns / 1e6) * 1e6)
            if observation is None or not 0 <= ctx.now_ns - request.observation.timestamp_ns <= max_age:
                return
            if self.schedule.accept(chunk, ctx.now_ns):
                if self.chunk_out:
                    info = PredictionMetadata(chunk.session_id, chunk.request_id, chunk.observation_id,
                                              chunk.start_step, chunk.period_ns)
                    self.chunk_out.write(chunk.actions, spec_id=chunk.spec_id,
                                         prediction=info, generation=generation)
                ctx.report("OK", "Receiving inference responses")
        except (OSError, ValueError, TypeError) as error:
            self.schedule.chunk = None
            self._invalidate(ctx)
            ctx.report("DEGRADED", str(error))
            if isinstance(error, OSError):
                ctx.emit("estop", {"reason": "policy connection failed"})

    def process(self, ctx):
        if self.generation != ctx.write_generation:
            self._reset(ctx)
        enabled = self.enabled_input.read()[0] if "enabled" in ctx.inputs else None
        if self.paused or ("enabled" in ctx.inputs and (enabled is None or not enabled.fresh(ctx.now_ns, 200_000_000) or not enabled.value)):
            self.schedule.chunk = None
            self._invalidate(ctx)
            ctx.report("IDLE", "Model paused" if self.paused else "Waiting for controller enable")
            return
        observation = self.selector.select(ctx)
        self._collect(ctx, observation)
        if self.generation != ctx.generation:
            return
        if observation is None:
            self.schedule.chunk = None
            self._invalidate(ctx)
            ctx.report("WAITING", "Waiting for fresh, aligned observations: " + ", ".join(self.selector.ports))
            return
        step = self.schedule.step(ctx.now_ns)
        ready = self.schedule.ready_for_request() if isinstance(self.schedule, RtcChunkSchedule) else step >= self.next_request_step
        if (self.pending is None and ready
                and observation.observation_id != self.last_observation):
            request = PolicyRequest(self.schedule.session_id, str(uuid.uuid4()), observation, self.spec.id, step,
                                    {"previous_actions": self.schedule.remaining(ctx.now_ns),
                                     "inference_delay": int(self.options.get("inference_delay", 0))}, view=self.view)
            if isinstance(self.schedule, RtcChunkSchedule):
                # The server retains its normalized model output for RTC;
                # this index addresses the same unblended prediction as remaining().
                request = replace(request, context={"inference_start_idx": self.schedule.interval,
                    "inference_delay": self.schedule.delay})
            generation = ctx.write_generation
            if self.executor is None:
                # Runtime.tick() is deterministic and does not start background work.
                future = Future()
                try:
                    future.set_result(self._predict(request))
                except Exception as error:
                    future.set_exception(error)
            else:
                future = self.executor.submit(self._predict, request)
            self.pending = future, request, generation
            self.last_observation = observation.observation_id
            self.next_request_step = step + self.inference_interval
            self._collect(ctx, observation)
        now = ctx.now_ns
        command = self.schedule.command(now)
        if command is not None:
            if self.reference_out:
                self.reference_out.write(command.values, spec_id=command.spec_id,
                                         command=command_metadata(command), generation=self.generation)
            prepared = self.prepare_command(command)
            if prepared is not None:
                self.command_out.write(prepared.values, spec_id=prepared.spec_id,
                                       command=command_metadata(prepared), generation=self.generation)
            else:
                self.command_out.invalidate()
        elif not self.schedule.available(now):
            self.command_out.invalidate()
            if self.reference_out:
                self.reference_out.invalidate()
            # Preserve an inference error until a later request succeeds.
            if self.pending is not None:
                ctx.report("WAITING", "Waiting for inference response; no actions available")
