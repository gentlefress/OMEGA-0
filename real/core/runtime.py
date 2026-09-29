"""Module workers, shared storage, bounded queues, and output inhibition."""
from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Callable

from .protocol import Sample, TensorSpec, _tensor_layout
from .logging import ModuleLogger
from .events import EventBindings, SYSTEM_EVENTS

log = ModuleLogger(logging.getLogger(__name__), {'module_id': 'runtime'})


@dataclass(frozen=True)
class Event:
    kind: str
    source: str = "operator"
    payload: Any = None


@dataclass
class Health:
    level: str = "OK"
    message: str = ""
    processed: int = 0
    failures: int = 0
    event_drops: int = 0


class StreamQueue:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Queue capacity must be positive")
        self.items = deque(maxlen=capacity)
        self.dropped = 0
        self.lock = threading.Lock()

    def push(self, sample):
        with self.lock:
            if len(self.items) == self.items.maxlen:
                self.dropped += 1
            self.items.append(sample)

    def drain(self):
        with self.lock:
            result = list(self.items)
            self.items.clear()
            return result


@dataclass
class Stream:
    owner: str
    spec: TensorSpec | None
    history: deque
    sequence: int = 0
    subscribers: list[StreamQueue] = field(default_factory=list)
    wake_events: list[threading.Event] = field(default_factory=list, repr=False)
    lock: Any = field(default_factory=threading.Lock, repr=False)


class _LockGroup:
    """Reusable lock order; acquisition state stays local to each caller."""

    def __init__(self, locks):
        self.locks = locks
        self.reverse = tuple(reversed(locks))

    def __enter__(self):
        acquired = 0
        try:
            for lock in self.locks:
                lock.acquire()
                acquired += 1
        except BaseException:
            for lock in reversed(self.locks[:acquired]):
                lock.release()
            raise

    def __exit__(self, *_):
        for lock in self.reverse:
            lock.release()


class Port:
    """An ordered set of prebound streams with one access mode.

    Names and lock order are resolved at bind time. read/write/history/drain use
    direct stream references and positional tuples, without registry lookups or
    sorting. A one-field port follows the same API as a group.
    """

    def __init__(self, store, names, streams, *, writable=False, context=None, queues=None):
        self._store = store
        self.names = tuple(names)
        self._streams = tuple(streams)
        self._writable = writable
        self._context = context
        self._queues = queues
        unique = {id(stream): (name, stream.lock) for name, stream in zip(self.names, self._streams)}
        self._locks = tuple(lock for _, lock in sorted(unique.values(), key=lambda item: item[0]))
        self._lock = self._locks[0] if len(self._locks) == 1 else _LockGroup(self._locks)

    def __len__(self):
        return len(self._streams)

    def _readable(self):
        if self._writable:
            raise PermissionError("Cannot read through a write port")

    def read(self):
        """Return latest Sample-or-None entries in binding order."""
        self._readable()
        with self._lock:
            return tuple(stream.history[-1] if stream.history else None for stream in self._streams)

    def history(self):
        self._readable()
        with self._lock:
            return tuple(tuple(stream.history) for stream in self._streams)

    def drain(self):
        """Drain bound input queues atomically against field commits."""
        self._readable()
        if self._queues is None:
            raise RuntimeError("This port has no input queues")
        with self._lock:
            return tuple(queue.drain() for queue in self._queues)

    def write(self, values, *, generation=None, source_ns=None, clock="unknown",
              spec_id="", command=None, prediction=None, batch_id="", field_metadata=None):
        """Commit tensors/None in binding order; return samples or stale-work None.

        field_metadata, when supplied, is a same-length sequence of dictionaries
        overriding source_ns, clock, spec_id, command, prediction or batch_id.
        """
        if not self._writable:
            raise PermissionError("Cannot write through a read port")
        if not isinstance(values, (tuple, list)) or len(values) != len(self):
            raise ValueError("Values must be a sequence matching the bound field count")
        expected = (self._context.write_generation if self._context is not None else self._store.generation) if generation is None else generation
        if expected != self._store.generation:
            return None
        common = dict(source_ns=source_ns, clock=clock, spec_id=spec_id, command=command,
                      prediction=prediction, batch_id=batch_id or str(uuid.uuid4()))
        overrides = (None,) * len(self) if field_metadata is None else tuple(field_metadata)
        if len(overrides) != len(self) or any(item is not None and not item.keys() <= common.keys() for item in overrides):
            raise ValueError("Unknown per-field metadata or incorrect field count")
        samples = tuple(Sample(name, 0, 0, value, valid=value is not None,
                               **(common if item is None else {**common, **item}))
                        for name, value, item in zip(self.names, values, overrides))
        candidates = tuple(self._store._prepare(stream, sample) for stream, sample in zip(self._streams, samples))
        with self._lock:
            if expected != self._store.generation:
                return None
            # Validate the entire group before changing schemas or any samples.
            for stream, candidate in zip(self._streams, candidates):
                spec = stream.spec
                if candidate is not None and isinstance(spec, TensorSpec) and (spec.dtype, spec.shape) != (candidate.dtype, candidate.shape):
                    raise ValueError("A stream's tensor dtype and shape cannot change during a run")
            now = self._store.clock()
            if now < 0:
                raise ValueError("Invalid receive timestamp")
            for stream, sample, candidate in zip(self._streams, samples, candidates):
                self._store._check_shape(stream, candidate)
                self._store._commit(stream, sample, now)
        return samples


def _field_names(names):
    """Resolve a name or iterable once when binding; preserve caller order."""
    names = (names,) if isinstance(names, str) else tuple(names)
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Field names must be nonempty strings without duplicates")
    return names


class Store:
    """Atomic stream groups with independent progress outside each selection.

    Dynamic producers reserve a bounded namespace at configuration time. The
    registry lock protects discovery/reset; existing streams stay independent.
    """
    def __init__(self, clock: Callable[[], int], history_size=128):
        self.clock = clock
        self.history_size = history_size
        self._registry_lock = threading.Lock()
        self.streams: dict[str, Stream] = {}
        self.namespaces = {}
        self.generation = 0

    def reserve_namespace(self, prefix, owner, limit):
        """Reserve names for a producer whose output fields arrive at runtime."""
        with self._registry_lock:
            if not prefix or limit <= 0 or any(prefix.startswith(p) or p.startswith(prefix) for p in self.namespaces):
                raise ValueError("Dynamic output namespaces must be bounded and disjoint")
            if any(name.startswith(prefix) for name in self.streams):
                raise ValueError(f"Output namespace {prefix!r} already has a writer")
            self.namespaces[prefix] = (owner, limit)

    def _namespace(self, name):
        return next(((prefix, owner, limit) for prefix, (owner, limit) in self.namespaces.items()
                     if name.startswith(prefix) and name != prefix), None)

    def _declare(self, name, owner, spec=None):
        namespace = self._namespace(name)
        if namespace:
            prefix, reserved_owner, limit = namespace
            if owner != reserved_owner:
                raise ValueError(f"Output {name!r} belongs to {reserved_owner}")
            if sum(key.startswith(prefix) for key in self.streams) >= limit:
                raise ValueError(f"Output namespace {prefix!r} exceeds its field limit")
        if not name or name in self.streams:
            raise ValueError(f"Output {name!r} must have exactly one writer")
        self.streams[name] = Stream(owner, spec, deque(maxlen=self.history_size))

    def declare(self, name, owner, spec=None):
        with self._registry_lock:
            self._declare(name, owner, spec)

    def bind_input(self, name):
        """Allow a consumer to subscribe before its dynamic producer receives data."""
        with self._registry_lock:
            if name not in self.streams:
                namespace = self._namespace(name)
                if namespace is None:
                    raise ValueError(f"Unknown stream {name}")
                self._declare(name, namespace[1])

    def subscribe(self, name, capacity):
        if name not in self.streams:
            raise ValueError(f"Unknown stream {name}")
        stream = self.streams[name]
        subscription = StreamQueue(capacity)
        with stream.lock:
            stream.subscribers.append(subscription)
        return subscription

    def _commit(self, stream, sample, now=None):
        """Finalize a private prepared sample while holding its stream lock.

        No caller can see the sample until it enters history/subscriber queues.
        Assigning identity here avoids copying its immutable payload a second time.
        """
        now = self.clock() if now is None else now
        if now < 0:
            raise ValueError("Invalid receive timestamp")
        object.__setattr__(sample, "sequence", stream.sequence + 1)
        object.__setattr__(sample, "received_ns", now)
        stream.sequence = sample.sequence
        stream.history.append(sample)
        for subscription in stream.subscribers:
            subscription.push(sample)
        for wake in stream.wake_events:
            wake.set()
        return sample

    def _owned_stream(self, owner, name):
        stream = self.streams.get(name)
        if stream is None:
            with self._registry_lock:
                namespace = self._namespace(name)
                if namespace is None or namespace[1] != owner:
                    raise PermissionError(f"{owner} does not own {name}")
                if name not in self.streams:
                    self._declare(name, owner)
                stream = self.streams[name]
        if stream.owner != owner:
            raise PermissionError(f"{owner} does not own {name}")
        return stream

    @staticmethod
    def _prepare(stream, sample):
        # Copy and potentially expensive validation stay outside shared locks.
        if not sample.valid:
            return None
        candidate = _tensor_layout(sample.value.dtype, sample.value.shape)
        if isinstance(stream.spec, TensorSpec):
            stream.spec.validate(sample.value, finite=False)
        elif stream.spec is not None:
            stream.spec.validate(sample.value)
        return candidate

    @staticmethod
    def _check_shape(stream, candidate):
        if candidate is None:
            return
        if stream.spec is None:
            stream.spec = candidate
        elif isinstance(stream.spec, TensorSpec) and (stream.spec.dtype, stream.spec.shape) != (candidate.dtype, candidate.shape):
            raise ValueError("A stream's tensor dtype and shape cannot change during a run")

    def bind_read(self, names):
        names = _field_names(names)
        return Port(self, names, tuple(self.streams[name] for name in names))

    def bind_write(self, owner, names):
        names = _field_names(names)
        return Port(self, names, tuple(self._owned_stream(owner, name) for name in names), writable=True)

    def write(self, owner, values, *, metadata_by_field=None, **metadata):
        """Named convenience API; cache bind_write() for repeated I/O."""
        names = tuple(values)
        metadata_by_field = metadata_by_field or {}
        if not metadata_by_field.keys() <= values.keys():
            raise ValueError("Unknown per-field metadata")
        samples = self.bind_write(owner, names).write(tuple(values.values()),
            field_metadata=tuple(metadata_by_field.get(name) for name in names), **metadata)
        return None if samples is None else dict(zip(names, samples))

    def read(self, names):
        """Named latest-sample snapshot; cache bind_read() for repeated I/O."""
        port = self.bind_read(names)
        return dict(zip(port.names, port.read()))

    def history(self, names):
        port = self.bind_read(names)
        return dict(zip(port.names, port.history()))

    def sequence(self, name):
        stream = self.streams[name]
        with stream.lock:
            return stream.sequence

    def reset(self):
        """Invalidate all streams atomically against commits, including raw writes.

        Reset is a rare global transition. Field operations lock only their
        selected streams; reset takes every stream lock in name order.
        """
        with self._registry_lock, ExitStack() as locks:
            for name in sorted(self.streams):
                locks.enter_context(self.streams[name].lock)
            self.generation += 1
            for name, stream in self.streams.items():
                self._commit(stream, Sample(name, 0, 0, valid=False))
            return self.generation


class Module:
    """Functional module base. SDK and codec helpers need not subclass this."""
    required_inputs: tuple[str, ...] = ()
    required_outputs: tuple[str, ...] = ()
    queued_inputs = False
    event_actions = None

    def configure(self, options: dict):
        self.options = dict(options)

    def output_namespaces(self):
        """Optional dynamic output prefixes and their maximum field counts."""
        return {}

    def bind(self, ctx: Context):
        """Cache input/output handles before start(), without opening devices."""
        pass

    def start(self, ctx: Context):
        pass

    def process(self, ctx: Context):
        pass

    def log_status(self, ctx: Context):
        """Log a read-only progress snapshot; called by the runtime monitor.

        Override for module-specific counters. Do not perform device I/O or
        change module state here: the module worker may be running concurrently.
        """
        ctx.log_status()

    def on_event(self, ctx: Context, event: Event):
        pass

    def stop(self, ctx: Context):
        pass


class Context:
    def __init__(self, runtime, module_id, inputs, outputs, queue_size, queued):
        self.runtime = runtime
        self.id = module_id
        self.logger = ModuleLogger(logging.getLogger(f'omega_real.modules.{module_id}'), {'module_id': module_id})
        self.inputs = inputs
        self.outputs = outputs
        self.write_generation = runtime.generation
        self._read_ports, self._write_ports = {}, {}
        self._reported_status = None
        self._reported_at_ns = None
        self._queued = queued
        self.queues = {key: runtime.store.subscribe(name, queue_size) for key, name in inputs.items()} if queued else {}

    @property
    def runner(self):
        return next(r for r in self.runtime.runners if r.id == self.id)

    @property
    def rate_hz(self):
        """Configured rate for the module worker."""
        return self.runner.rate_hz

    @property
    def configuration(self):
        """Resolved workflow configuration, copied for module-owned metadata."""
        from copy import deepcopy
        return deepcopy(self.runtime.configuration)

    @property
    def stopping(self):
        return self.runtime.stop_requested.is_set()

    @property
    def queue_capacity(self):
        return self.runner.queue_size

    @property
    def dropped_inputs(self):
        """Snapshot of dropped publications, keyed by local input port."""
        return {port: queue.dropped for port, queue in self.queues.items()}

    @property
    def dropped_events(self):
        return self.runtime.health[self.id].event_drops

    @property
    def faults(self):
        """Immutable snapshot of workflow failures for resource finalization."""
        return tuple(self.runtime.faults)

    def events(self):
        """Begin a new cycle and drain subscribed events.

        Capture generation before blocking/asynchronous work and pass it to
        write(), so a reset cannot make old work current again.
        """
        self.write_generation = self.generation
        return self.runner.events.drain()

    def process(self):
        """One module step, shared by worker threads, groups and Runtime.tick()."""
        for event in self.events():
            self.runner.module.on_event(self, event)
        self.runner.module.process(self)
        self.runtime.health[self.id].processed += 1

    @property
    def now_ns(self):
        return self.runtime.clock()

    @property
    def generation(self):
        return self.runtime.generation

    def bind_read(self, ports):
        """Resolve configured input ports once; cache the returned ordered handle."""
        ports = _field_names(ports)
        if ports not in self._read_ports:
            names = tuple(self.inputs[port] for port in ports)
            queues = tuple(self.queues[port] for port in ports) if self._queued else None
            self._read_ports[ports] = Port(self.runtime.store, names,
                tuple(self.runtime.store.streams[name] for name in names), context=self, queues=queues)
        return self._read_ports[ports]

    def bind_write(self, ports, *, namespace=None):
        """Bind outputs, or discovered fields in an owned dynamic namespace."""
        ports = _field_names(ports)
        key = (namespace, ports)
        if key not in self._write_ports:
            if namespace is not None:
                declaration = self.runtime.store.namespaces.get(namespace + '.')
                if declaration is None or declaration[0] != self.id:
                    raise PermissionError(f"{self.id} does not own namespace {namespace!r}")
            names = tuple(self.outputs[port] if namespace is None else namespace + '.' + port for port in ports)
            streams = tuple(self.runtime.store._owned_stream(self.id, name) for name in names)
            self._write_ports[key] = Port(self.runtime.store, names, streams, writable=True, context=self)
        return self._write_ports[key]

    def read(self, ports):
        """Named convenience snapshot; use a bound port for lookup-free reads."""
        ports = _field_names(ports)
        return dict(zip(ports, self.bind_read(ports).read()))

    def write(self, values, *, namespace=None, metadata_by_field=None, **metadata):
        """Named convenience write; None invalidates a field.

        Keys name output ports or fields in an explicitly supplied namespace.
        Results keep those keys. Cache bind_write() for repeated operations.
        """
        names = tuple(values)
        metadata_by_field = metadata_by_field or {}
        if not metadata_by_field.keys() <= values.keys():
            raise ValueError("Unknown per-field metadata")
        samples = self.bind_write(names, namespace=namespace).write(tuple(values.values()),
            field_metadata=tuple(metadata_by_field.get(name) for name in names), **metadata)
        return None if samples is None else dict(zip(names, samples))

    def history(self, ports):
        ports = _field_names(ports)
        return dict(zip(ports, self.bind_read(ports).history()))

    def drain(self, ports):
        ports = _field_names(ports)
        return dict(zip(ports, self.bind_read(ports).drain()))

    def emit(self, kind, payload=None):
        self.runtime.emit(Event(kind, self.id, payload))

    def report(self, level, message=""):
        """Update health and log transitions; throttle changing progress text."""
        health = self.runtime.health[self.id]
        health.level, health.message = level, message
        status = (level, message)
        if status == self._reported_status:
            return
        now = self.now_ns
        level_changed = self._reported_status is None or level != self._reported_status[0]
        if level_changed or now - self._reported_at_ns >= 2_000_000_000:
            self.log_status()
            self._reported_status, self._reported_at_ns = status, now

    def log_status(self, message=None):
        """Emit this module's current health, optionally with a progress message."""
        health = self.runtime.health[self.id]
        level = health.level
        message = health.message if message is None else message
        priority = logging.ERROR if level == 'FAULT' else logging.WARNING if level == 'DEGRADED' else logging.INFO
        self.logger.log(priority, "%s%s", level, ": " + message if message else "")

    def write_output(self, callback, *, generation=None):
        """Serialize the last host-side send decision with output inhibition."""
        with self.runtime.output_lock:
            if not self.runtime.output_enabled or (generation is not None and generation != self.generation):
                return False
            callback()
            return True

    def write_shutdown(self, callback):
        """Run a transport's explicit stop operation after motion is inhibited.

        Controller modules own stop semantics. Transports send their explicit
        StopCommand values without adding robot-specific fields.
        This never enables a runtime that was launched without hardware arming.
        """
        with self.runtime.output_lock:
            if not self.runtime.hardware_armed:
                return False
            callback()
            return True


@dataclass
class Runner:
    id: str
    module: Module
    rate_hz: float
    inputs: dict
    outputs: dict
    queue_size: int
    events: StreamQueue = field(default_factory=lambda: StreamQueue(64))
    context: Context | None = None
    thread: threading.Thread | None = None
    ready: threading.Event = field(default_factory=threading.Event)
    drain_ready: threading.Event = field(default_factory=threading.Event)
    started: bool = False
    finished: threading.Event = field(default_factory=threading.Event)


@dataclass
class ExecutionGroup:
    id: str
    members: tuple[Runner, ...]
    rate_hz: float
    trigger: str | None = None
    wake: threading.Event = field(default_factory=threading.Event, repr=False)
    thread: threading.Thread | None = None
    ready: threading.Event = field(default_factory=threading.Event)


class Runtime:
    def __init__(self, *, clock=time.monotonic_ns, allow_hardware=False, history_size=128, configuration=None):
        from copy import deepcopy
        self.configuration = deepcopy(configuration) if configuration is not None else None
        self.clock = clock
        self.store = Store(clock, history_size)
        self.runners: list[Runner] = []
        self.execution_groups: list[ExecutionGroup] = []
        self.grouped_modules = set()
        self.execution_units = []
        self.tick_order = []
        self.health: dict[str, Health] = {}
        self.stop_requested = threading.Event()
        self.producers_stopped = threading.Event()
        self.output_lock = threading.RLock()
        self.hardware_armed = bool(allow_hardware)
        self.output_enabled = bool(allow_hardware)
        self.running = False
        self.threaded = False
        self.faults = []
        self.consumer_order = []
        self.event_bindings = {}
        self.runtime_events = None
        self.routes = {}
        self.runtime_routes = EventBindings([], sources=set())

    @property
    def generation(self):
        return self.store.generation

    def add(self, module_id, module, *, inputs=None, outputs=None, rate_hz=30, options=None, queue_size=256, specs=None, events=None):
        if isinstance(rate_hz, bool) or not isinstance(rate_hz, (int, float)) or not math.isfinite(rate_hz) or rate_hz < 0:
            raise ValueError("Module rate_hz must be finite and nonnegative")
        if type(queue_size) is not int or queue_size <= 0:
            raise ValueError("Module queue_size must be a positive integer")
        if not isinstance(module_id, str):
            raise ValueError("Module ID must be a nonempty string")
        if hasattr(module, "run"):
            raise ValueError("Modules use start/process/on_event/stop; custom run() loops are not supported")
        if self.running or module_id in self.health or any(g.id == module_id for g in self.execution_groups) or not module_id or rate_hz < 0:
            raise ValueError("Invalid or duplicate module configuration")
        inputs, outputs = dict(inputs or {}), dict(outputs or {})
        if not set(module.required_inputs) <= inputs.keys() or not set(module.required_outputs) <= outputs.keys():
            raise ValueError(f"Missing required ports for {module_id}")
        module.id = module_id
        module.configure(options or {})
        for prefix, limit in module.output_namespaces().items():
            self.store.reserve_namespace(prefix, module_id, limit)
        for port, name in outputs.items():
            spec = (specs or {}).get(port)
            self.store.declare(name, module_id, TensorSpec(**spec) if isinstance(spec, dict) else spec)
        self.runners.append(Runner(module_id, module, float(rate_hz), inputs, outputs, int(queue_size)))
        self.health[module_id] = Health()
        self.event_bindings[module_id] = [] if events is None else events
        return module

    def add_execution_group(self, group_id, modules, *, rate_hz, trigger=None):
        """Run process-based modules in order on one periodic worker.

        Configure after adding the member modules. The group rate replaces their
        individual rates. Inference I/O remains in its module-owned executor.
        With a trigger stream, wait for a fresh publication within each period,
        falling back to the timer. This keeps the nominal rate with variable
        intervals, bounded below by half a period; it is not a strict rate cap.
        """
        if self.running or not group_id or group_id in self.health or any(g.id == group_id for g in self.execution_groups):
            raise ValueError("Invalid or duplicate execution group ID")
        if not isinstance(modules, (list, tuple)) or not modules or len(set(modules)) != len(modules):
            raise ValueError("Execution group needs a nonempty list of distinct modules")
        rate_hz = float(rate_hz)
        if not math.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError("Execution group rate_hz must be finite and positive")
        if trigger is not None and (not isinstance(trigger, str) or not trigger):
            raise ValueError("Execution group trigger must be a nonempty stream name")
        runners = {runner.id: runner for runner in self.runners}
        if set(modules) - runners.keys():
            raise ValueError("Execution group contains unknown modules")
        if set(modules) & self.grouped_modules:
            raise ValueError("A module may belong to only one execution group")
        members = tuple(runners[name] for name in modules)
        group = ExecutionGroup(group_id, members, rate_hz, trigger)
        self.execution_groups.append(group)
        self.grouped_modules.update(modules)
        for runner in members:
            runner.rate_hz = rate_hz
        return group

    def _bind(self):
        sources = set(self.health) | {"runtime", "operator"}
        self.routes = {runner.id: EventBindings(self.event_bindings[runner.id], sources=sources,
                       actions=runner.module.event_actions)
                       for runner in self.runners}
        self.runtime_routes = EventBindings(self.runtime_events or [], sources=sources, actions=SYSTEM_EVENTS)
        groups = {runner.id: group for group in self.execution_groups for runner in group.members}
        seen = set()
        self.execution_units = []
        for runner in self.runners:
            group = groups.get(runner.id)
            if group is None:
                self.execution_units.append(runner)
            elif group.id not in seen:
                self.execution_units.append(group)
                seen.add(group.id)
        self.tick_order = [runner for unit in self.execution_units
                           for runner in (unit.members if isinstance(unit, ExecutionGroup) else (unit,))]
        for runner in self.runners:
            for name in runner.inputs.values():
                self.store.bind_input(name)
            runner.context = Context(self, runner.id, runner.inputs, runner.outputs, runner.queue_size, runner.module.queued_inputs)
        for runner in self.runners:
            runner.module.bind(runner.context)
        for group in self.execution_groups:
            if group.trigger is not None:
                self.store.bind_input(group.trigger)
                stream = self.store.streams[group.trigger]
                if stream.owner in {r.id for r in group.members}:
                    raise ValueError("Execution group trigger must have an independent producer")
                if not any(group.trigger in r.inputs.values() for r in group.members):
                    raise ValueError("Execution group trigger must be a member input")
                with stream.lock:
                    if group.wake not in stream.wake_events:
                        stream.wake_events.append(group.wake)
        # Queued consumers may themselves produce data (e.g. sender receipts).
        # Drain upstream consumers before closing their downstream recorders.
        remaining = {r.id: r for r in self.runners if r.module.queued_inputs}
        self.consumer_order = []
        while remaining:
            ready = [r for r in remaining.values()
                     if not {self.store.streams[name].owner for name in r.inputs.values()} & remaining.keys()]
            if not ready:
                raise ValueError("Queued consumer dependencies must not form a cycle")
            for runner in ready:
                self.consumer_order.append(runner)
                del remaining[runner.id]

    def emit(self, event: Event):
        self._deliver_event(event)
        # Runtime actions are terminal; configuration cannot create event loops.
        for action in self.runtime_routes.dispatch(event):
            self._deliver_event(action)

    def _deliver_event(self, event: Event):
        if event.kind == "estop":
            reason = event.payload.get("reason", "stop requested") if isinstance(event.payload, dict) else event.payload
            log.warning("Hardware output latched off by %s: %s; restart the runtime to rearm", event.source, reason)
        elif event.kind == "shutdown":
            log.info("Shutdown requested by %s", event.source)
        if event.kind in {"reset", "discontinuity", "estop"}:
            with self.output_lock:
                # Resetting history never implicitly rearms hardware.
                if event.kind == "estop":
                    self.output_enabled = False
                # Invalidate cached commands immediately, before workers receive the event.
                self.store.reset()
        if event.kind == "shutdown":
            self.request_stop()
        for runner in self.runners:
            # Safety and data invalidation always reach every module. Explicit
            # bindings can additionally request a different local action.
            routed = list(self.routes[runner.id].dispatch(event))
            if event.kind in SYSTEM_EVENTS:
                routed = [event] + [item for item in routed if item.kind != event.kind]
            for delivered in routed:
                before = runner.events.dropped
                runner.events.push(delivered)
                self.health[runner.id].event_drops += runner.events.dropped - before

    def _fault(self, runner, error):
        self.health[runner.id].level = "FAULT"
        self.health[runner.id].message = str(error)
        self.health[runner.id].failures += 1
        self.faults.append((runner.id, str(error)))
        self.request_stop()
        runner.context.logger.error("Failed: %s", error, exc_info=True)

    def _process(self, runner):
        runner.context.process()

    def _start_runner(self, runner):
        runner.context.logger.info("Starting %s (%.1f Hz)", type(runner.module).__name__, runner.rate_hz)
        self.health[runner.id].level = "STARTING"
        runner.started = True
        runner.module.start(runner.context)
        if self.health[runner.id].level == "STARTING":
            self.health[runner.id].level = "READY"
        runner.ready.set()
        runner.context.logger.info("Started")

    def _stop_runner(self, runner):
        try:
            if runner.started:
                runner.context.logger.info("Stopping")
                runner.context.write_generation = self.generation
                runner.module.stop(runner.context)
                if self.health[runner.id].level != "FAULT":
                    self.health[runner.id].level, self.health[runner.id].message = "STOPPED", ""
                runner.context.logger.info("Stopped")
        except Exception as error:
            self._fault(runner, error)
        finally:
            runner.started = False
            runner.finished.set()

    def _run(self, runner):
        try:
            # A partially started module is still stopped to release acquired resources.
            self._start_runner(runner)
            interval = 1 / runner.rate_hz if runner.rate_hz else 0.01
            next_tick = time.monotonic()
            last_sequences = None
            inputs = runner.context.bind_read(runner.inputs)
            while not self.stop_requested.is_set():
                sequences = tuple(sample.sequence if sample is not None else 0 for sample in inputs.read())
                if runner.rate_hz or sequences != last_sequences or runner.events.items:
                    self._process(runner)
                    last_sequences = sequences
                next_tick = max(next_tick + interval, time.monotonic())
                self.stop_requested.wait(max(0, next_tick - time.monotonic()))
        except Exception as error:
            self._fault(runner, error)
        finally:
            runner.ready.set()
            if runner.module.queued_inputs:
                # Runtime.stop releases consumers after their producers finish.
                runner.drain_ready.wait()
            self._stop_runner(runner)

    def _run_group(self, group):
        runner = group.members[0]
        try:
            for runner in group.members:
                self._start_runner(runner)
            group.ready.set()
            interval = 1 / group.rate_hz
            next_tick = time.monotonic()
            last_tick = next_tick - interval
            while not self.stop_requested.is_set():
                if group.trigger is not None:
                    # One cycle per fixed window. Discard notifications from
                    # before the window; members read the latest atomic snapshot,
                    # never a backlog. A half-period floor prevents bursts when
                    # input resumes just after a timer fallback.
                    earliest = max(next_tick, last_tick + interval / 2)
                    self.stop_requested.wait(max(0, earliest - time.monotonic()))
                    group.wake.clear()
                    if self.stop_requested.is_set():
                        break
                    group.wake.wait(max(0, next_tick + interval - time.monotonic()))
                    if self.stop_requested.is_set():
                        break
                last_tick = time.monotonic()
                for runner in group.members:
                    if self.stop_requested.is_set():
                        break
                    self._process(runner)
                if group.trigger is not None:
                    next_tick += interval
                    # A timeout runs at the end of its window. Keep the next
                    # window's deadline instead of adding processing time to it.
                    # Skip completely missed windows after a slow cycle.
                    now = time.monotonic()
                    if now >= next_tick + interval:
                        next_tick = now
                else:
                    next_tick = max(next_tick + interval, time.monotonic())
                    self.stop_requested.wait(max(0, next_tick - time.monotonic()))
        except Exception as error:
            self._fault(runner, error)
        finally:
            group.ready.set()
            for runner in reversed(group.members):
                if not runner.module.queued_inputs:
                    self._stop_runner(runner)
            # The runtime releases queued members individually in global consumer
            # order, so another worker can drain before this group's recorder.
            members = {runner.id for runner in group.members}
            for runner in self.consumer_order:
                if runner.id in members:
                    runner.drain_ready.wait()
                    self._stop_runner(runner)

    def start(self, *, threaded=True):
        if self.running:
            raise RuntimeError("Runtime already started")
        if self.stop_requested.is_set():
            raise RuntimeError("Construct a new runtime after shutdown")
        self._bind()
        log.info("Starting runtime: %d modules, %d execution groups; hardware output %s",
                 len(self.runners), len(self.execution_groups),
                 "ENABLED (--arm)" if self.hardware_armed else "DISABLED (no --arm)")
        self.running, self.threaded = True, threaded
        try:
            for unit in self.execution_units:
                if threaded:
                    target = self._run_group if isinstance(unit, ExecutionGroup) else self._run
                    unit.thread = threading.Thread(target=target, args=(unit,), name=unit.id, daemon=True)
                    unit.thread.start()
                    if not unit.ready.wait(10):
                        raise TimeoutError(f"{unit.id} startup timed out")
                    if self.faults:
                        raise RuntimeError(str(self.faults))
                else:
                    members = unit.members if isinstance(unit, ExecutionGroup) else (unit,)
                    for runner in members:
                        self._start_runner(runner)
            self.emit(Event("started", "runtime"))
        except BaseException:
            self.request_stop()
            self.stop()
            raise

    def tick(self):
        """Deterministic execution for replay/unit tests; caller advances its clock."""
        if not self.running or self.threaded:
            raise RuntimeError("tick requires start(threaded=False)")
        for runner in self.tick_order:
            if self.stop_requested.is_set():
                break
            try:
                self._process(runner)
            except Exception as error:
                self._fault(runner, error)
                raise

    def request_stop(self):
        with self.output_lock:
            self.output_enabled = False
        self.stop_requested.set()
        for group in self.execution_groups:
            group.wake.set()

    def stop(self, timeout=5):
        if self.running:
            log.info("Stopping runtime; inhibiting motion output and draining workers")
        self.request_stop()
        deadline = time.monotonic() + timeout
        pending = []
        producers = [r for r in reversed(self.tick_order) if not r.module.queued_inputs]
        consumers = self.consumer_order
        for runner in producers + consumers:
            if runner.module.queued_inputs:
                if pending:
                    # Leave consumer resources alive until a later stop() can join producers.
                    pending.append(runner.id)
                    continue
                self.producers_stopped.set()
                runner.drain_ready.set()
            if self.threaded and runner.id in self.grouped_modules:
                # Member completion, not a shared-thread join: that thread may
                # still be waiting for a downstream consumer's drain permission.
                group = next(g for g in self.execution_groups if runner in g.members)
                if group.thread is not None and not runner.finished.wait(max(0, deadline - time.monotonic())):
                    pending.append(runner.id)
            elif runner.thread is not None:
                runner.thread.join(max(0, deadline - time.monotonic()))
                if runner.thread.is_alive():
                    pending.append(runner.id)
            elif runner.started:
                self._stop_runner(runner)
        for group in self.execution_groups:
            if group.thread is not None and all(r.finished.is_set() for r in group.members):
                group.thread.join(max(0, deadline - time.monotonic()))
                if group.thread.is_alive():
                    pending.append(group.id)
        self.running = bool(pending)
        if pending:
            log.error("Shutdown timed out; still waiting for: %s", ", ".join(pending))
            raise TimeoutError(f"Workers have not stopped: {pending}")
        log.info("Runtime stopped")

    def log_status(self):
        output = "ENABLED" if self.output_enabled else "LATCHED OFF" if self.hardware_armed else "DISABLED (no --arm)"
        log.info("Hardware output: %s", output)
        for runner in self.runners:
            if runner.context is not None:
                runner.module.log_status(runner.context)


    def run(self, duration=None, *, status_interval=5):
        if not math.isfinite(status_interval) or status_interval < 0:
            raise ValueError("status_interval must be finite and nonnegative")
        self.start()
        try:
            log.info("Runtime running; Ctrl+C stops the program")
            self.log_status()
            deadline = None if duration is None else time.monotonic() + duration
            while not self.stop_requested.is_set():
                remaining = None if deadline is None else max(0., deadline - time.monotonic())
                if remaining == 0:
                    break
                wait = status_interval or remaining
                if remaining is not None and wait is not None:
                    wait = min(wait, remaining)
                if self.stop_requested.wait(wait):
                    break
                if status_interval:
                    self.log_status()
        except KeyboardInterrupt:
            log.info("Ctrl+C received; stopping runtime workers")
        finally:
            self.stop()
        if self.faults:
            raise RuntimeError(str(self.faults))
