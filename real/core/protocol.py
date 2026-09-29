"""Shared, versioned host contracts. No model, transport, or hardware imports."""
from __future__ import annotations

import base64
import dataclasses
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Any

import numpy as np

VERSION = 1
MAX_ARRAY_BYTES = 64 * 1024 * 1024
DTYPES = frozenset(("float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "bool"))


@lru_cache(maxsize=64)
def _dtype_name(dtype):
    """NumPy's dtype.name computes a string; fixed stream types can reuse it."""
    return dtype.name


@lru_cache(maxsize=512)
def _tensor_layout(dtype, shape):
    """Reuse immutable, validated layout metadata, never tensor contents."""
    return TensorSpec(_dtype_name(dtype), shape)


def immutable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if _dtype_name(value.dtype) not in DTYPES:
            raise ValueError(f"Unsupported dtype {value.dtype}")
        array = np.array(value, copy=True, order="C")
        array.setflags(write=False)
        return array
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise TypeError("Field names must be strings")
        return MappingProxyType({k: immutable(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(immutable(v) for v in value)
    if dataclasses.is_dataclass(value) or value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported sample value {type(value).__name__}")


@dataclass(frozen=True)
class TensorSpec:
    dtype: str
    shape: tuple[int, ...]
    units: str = ""
    frame: str = ""
    joints: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "joints", tuple(self.joints))
        if self.dtype not in DTYPES or len(self.shape) > 8 or any(type(d) is not int or d < 0 for d in self.shape):
            raise ValueError("Invalid tensor specification")
        if math.prod(self.shape) * np.dtype(self.dtype).itemsize > MAX_ARRAY_BYTES:
            raise ValueError("Tensor exceeds size limit")

    def validate(self, value, *, prefix: tuple[int, ...] = (), finite: bool = True):
        array = np.asarray(value)
        if _dtype_name(array.dtype) != self.dtype or array.shape != prefix + self.shape:
            raise ValueError(f"Expected {self.dtype}{prefix + self.shape}, got {array.dtype}{array.shape}")
        if finite and not np.isfinite(array).all():
            raise ValueError("Non-finite tensor")
        return array


@dataclass(frozen=True)
class CommandMetadata:
    """Identity and deadline shared by the tensor fields of one command."""
    session_id: str
    sequence: int
    valid_until_ns: int
    source: str = "policy"
    step: int = 0
    stop: bool = False

    def __post_init__(self):
        if (not isinstance(self.session_id, str) or not self.session_id or not isinstance(self.source, str)
                or any(type(value) is not int or value < 0 for value in (self.sequence, self.valid_until_ns, self.step))):
            raise ValueError("Invalid command metadata")
        if type(self.stop) is not bool:
            raise TypeError("Command stop marker must be boolean")


@dataclass(frozen=True)
class PredictionMetadata:
    """Identity and cadence of a model's published action-chunk tensors."""
    session_id: str
    request_id: str
    observation_id: str
    start_step: int
    period_ns: int

    def __post_init__(self):
        if (any(not isinstance(value, str) or not value for value in (self.session_id, self.request_id, self.observation_id))
                or type(self.start_step) is not int or self.start_step < 0
                or type(self.period_ns) is not int or self.period_ns <= 0):
            raise ValueError("Invalid prediction metadata")


@dataclass(frozen=True)
class Sample:
    stream: str
    sequence: int
    received_ns: int
    value: np.ndarray | None = None
    source_ns: int | None = None
    clock: str = "unknown"
    valid: bool = True
    spec_id: str = ""
    command: CommandMetadata | None = None
    prediction: PredictionMetadata | None = None
    batch_id: str = ""

    def __post_init__(self):
        if (not isinstance(self.stream, str) or not self.stream
                or any(type(value) is not int or value < 0 for value in (self.sequence, self.received_ns))):
            raise ValueError("Invalid sample identity or timing")
        if not all(isinstance(value, str) for value in (self.clock, self.spec_id, self.batch_id)):
            raise TypeError("Sample clock, spec_id and batch_id must be strings")
        if self.valid and not isinstance(self.value, np.ndarray):
            raise TypeError("Valid samples require a NumPy tensor; publish structured fields separately")
        if not self.valid and self.value is not None:
            raise ValueError("Invalid samples must have no value")
        if self.value is not None:
            _tensor_layout(self.value.dtype, self.value.shape)
        if self.command is not None and not isinstance(self.command, CommandMetadata):
            raise TypeError("Expected CommandMetadata")
        if self.prediction is not None and not isinstance(self.prediction, PredictionMetadata):
            raise TypeError("Expected PredictionMetadata")
        if self.command is not None and self.prediction is not None:
            raise ValueError("A sample cannot be both a command and a prediction")
        if (self.command is not None or self.prediction is not None) and not self.spec_id:
            raise ValueError("Command and prediction samples require spec_id")
        object.__setattr__(self, "value", immutable(self.value))

    def fresh(self, now_ns: int, max_age_ns: int):
        return self.valid and 0 <= now_ns - self.received_ns <= max_age_ns


@dataclass(frozen=True)
class Observation:
    observation_id: str
    samples: Mapping[str, Sample]
    timestamp_ns: int
    instruction: str = ""
    alignment: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.observation_id or not self.samples:
            raise ValueError("Observation requires an ID and samples")
        if any(not isinstance(v, Sample) for v in self.samples.values()):
            raise TypeError("Observation fields must contain samples")
        object.__setattr__(self, "samples", immutable(self.samples))
        object.__setattr__(self, "alignment", immutable(self.alignment))


@dataclass(frozen=True)
class ActionSpec:
    id: str
    fields: Mapping[str, TensorSpec]
    period_ns: int
    # Optional legacy labels, retained when reading existing artifacts/records.
    representation: str = ""
    embodiment: str = ""
    controller: str = ""

    def __post_init__(self):
        if not self.id or not self.fields or self.period_ns <= 0:
            raise ValueError("Incomplete action specification")
        if any(not isinstance(v, TensorSpec) for v in self.fields.values()):
            raise TypeError("Action fields require TensorSpec")
        object.__setattr__(self, "fields", immutable(self.fields))

    def validate(self, values: Mapping[str, Any], horizon: int | None = None):
        if set(values) != set(self.fields):
            raise ValueError(f"Action fields differ: expected {set(self.fields)}, received {set(values)}")
        for name, spec in self.fields.items():
            spec.validate(values[name], prefix=() if horizon is None else (horizon,))


@dataclass(frozen=True)
class PolicyRequest:
    session_id: str
    request_id: str
    observation: Observation
    action_spec_id: str
    start_step: int
    context: Mapping[str, Any] = field(default_factory=dict)
    version: int = VERSION
    view: str = "ego"

    def __post_init__(self):
        if self.version != VERSION or not self.session_id or not self.request_id or self.start_step < 0:
            raise ValueError("Invalid policy request")
        if not isinstance(self.observation, Observation):
            raise TypeError("Expected Observation")
        if self.view not in ("ego", "exo"):
            raise ValueError("Policy request view must be 'ego' or 'exo'")
        object.__setattr__(self, "context", immutable(self.context))


@dataclass(frozen=True)
class ActionChunk:
    session_id: str
    request_id: str
    observation_id: str
    spec_id: str
    start_step: int
    period_ns: int
    actions: Mapping[str, np.ndarray]
    version: int = VERSION

    def __post_init__(self):
        if self.version != VERSION or not self.session_id or not self.request_id or self.start_step < 0 or self.period_ns <= 0:
            raise ValueError("Invalid action chunk metadata")
        if not self.actions or any(not isinstance(v, np.ndarray) or v.ndim < 1 for v in self.actions.values()):
            raise ValueError("Actions require named arrays with a time dimension")
        lengths = {v.shape[0] for v in self.actions.values()}
        if len(lengths) != 1 or 0 in lengths or any(not np.isfinite(v).all() for v in self.actions.values()):
            raise ValueError("Invalid action horizon or non-finite action")
        object.__setattr__(self, "actions", immutable(self.actions))

    @property
    def horizon(self):
        return next(iter(self.actions.values())).shape[0]

    def validate(self, spec: ActionSpec):
        if self.spec_id != spec.id or self.period_ns != spec.period_ns:
            raise ValueError("Incompatible action specification or period")
        spec.validate(self.actions, self.horizon)


@dataclass(frozen=True)
class RobotCommand:
    """A command referencing the complete ActionSpec by its unique spec_id."""
    session_id: str
    sequence: int
    spec_id: str
    valid_until_ns: int
    values: Mapping[str, np.ndarray]
    source: str = "policy"
    step: int = 0

    def __post_init__(self):
        if not self.session_id or self.sequence < 0 or self.step < 0 or self.valid_until_ns < 0:
            raise ValueError("Invalid command identity or expiry")
        object.__setattr__(self, "values", immutable(self.values))


@dataclass(frozen=True)
class StopCommand(RobotCommand):
    """A producer's explicit stop operation, allowed after motion is inhibited.

    Only controller code should create these. Transport treats them as data and
    does not invent controller-specific stop fields. A stop does not expire.
    """


TYPES = {cls.__name__: cls for cls in (TensorSpec, CommandMetadata, PredictionMetadata, Sample, Observation, ActionSpec, PolicyRequest, ActionChunk, RobotCommand, StopCommand)}


def to_wire(value):
    if dataclasses.is_dataclass(value):
        return {"__type__": type(value).__name__, "fields": {f.name: to_wire(getattr(value, f.name)) for f in dataclasses.fields(value)}}
    if isinstance(value, np.ndarray):
        if value.dtype.name not in DTYPES or value.nbytes > MAX_ARRAY_BYTES:
            raise ValueError("Unsupported array")
        array = value.astype(value.dtype.newbyteorder("<"), copy=False)
        return {"__array__": True, "dtype": array.dtype.str, "shape": list(array.shape), "data": base64.b64encode(array.tobytes()).decode("ascii")}
    if isinstance(value, Mapping):
        if "__type__" in value or "__array__" in value:
            raise ValueError("Reserved wire field")
        return {k: to_wire(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_wire(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def from_wire(value, depth=0):
    if depth > 32:
        raise ValueError("Wire nesting limit exceeded")
    if isinstance(value, list):
        return [from_wire(v, depth + 1) for v in value]
    if not isinstance(value, dict):
        return value
    if "__array__" in value:
        dtype = np.dtype(value["dtype"])
        spec = TensorSpec(dtype.name, tuple(value["shape"]))
        expected = math.prod(spec.shape) * dtype.itemsize
        if len(value["data"]) > ((MAX_ARRAY_BYTES + 2) // 3) * 4:
            raise ValueError("Encoded array too large")
        raw = base64.b64decode(value["data"], validate=True)
        if len(raw) != expected:
            raise ValueError("Array byte count does not match schema")
        return np.frombuffer(raw, dtype=dtype).reshape(spec.shape).astype(dtype.newbyteorder("="), copy=True)
    if "__type__" in value:
        cls = TYPES.get(value["__type__"])
        if cls is None:
            raise ValueError("Unknown wire type")
        fields = {k: from_wire(v, depth + 1) for k, v in value["fields"].items()}
        if cls in (RobotCommand, StopCommand):
            # Older recordings duplicated ActionSpec.embodiment in each command.
            fields.pop("embodiment", None)
        return cls(**fields)
    return {k: from_wire(v, depth + 1) for k, v in value.items()}


def dumps(value) -> bytes:
    return json.dumps(to_wire(value), allow_nan=False, separators=(",", ":")).encode("utf-8")


def loads(data: bytes, max_bytes: int = 128 * 1024 * 1024):
    if len(data) > max_bytes:
        raise ValueError("Message too large")
    return from_wire(json.loads(data))


def action_spec(config: Mapping) -> ActionSpec:
    if isinstance(config, ActionSpec):
        return config
    return ActionSpec(**{**config, "fields": {k: TensorSpec(**v) for k, v in config["fields"].items()}})
