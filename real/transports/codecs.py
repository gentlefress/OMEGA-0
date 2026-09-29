"""Sonic padded-JSON commands and flat MessagePack robot feedback."""
from __future__ import annotations

import json
import math
from functools import lru_cache

import msgpack
import numpy as np

from ..core.protocol import MAX_ARRAY_BYTES, TensorSpec, _tensor_layout

HEADER_SIZE = 1280
WIRE_DTYPES = {"f32": "float32", "f64": "float64", "i32": "int32", "i64": "int64", "u8": "uint8", "bool": "bool"}
DTYPE_CODES = {v: k for k, v in WIRE_DTYPES.items()}
MAX_FIELDS = 128


def sonic_wire_fields(fields, *, topic):
    """Adapt configurable hand widths to the original C++ seven-value fields.

    Keep padding at the transport boundary: adapters and recordings retain the
    configured actuator count. The original planner decoder unconditionally
    reads seven values, and the pose decoder requires [7] or [N, 7].
    """
    if topic not in {"pose", "planner"}:
        return fields
    result = dict(fields)
    for name in ("left_hand_joints", "right_hand_joints"):
        if name not in fields:
            continue
        value = np.asarray(fields[name])
        if value.ndim not in (1, 2) or not value.size or not 1 <= value.shape[-1] <= 7 or value.dtype.kind != 'f':
            raise ValueError(f"Sonic C++ {name} requires 1..7 floating-point channels")
        if value.shape[-1] < 7:
            padded = np.zeros((*value.shape[:-1], 7), dtype=value.dtype)
            padded[..., :value.shape[-1]] = value
            result[name] = padded
    return result



@lru_cache(maxsize=256, typed=True)
def _sonic_json_layout(layout, version):
    descriptors, dtypes = [], []
    for name, dtype, shape in layout:
        if not name or name == '__schema' or len(name.encode()) > 256:
            raise ValueError('Invalid field name')
        spec = _tensor_layout(dtype, shape)
        code = DTYPE_CODES.get(spec.dtype)
        if code not in {'f32', 'f64', 'i32', 'i64', 'u8', 'bool'}:
            raise ValueError('The paired sonic_json SonicDeploy decoder does not support this dtype')
        descriptors.append({'name': name, 'dtype': code, 'shape': list(shape)})
        dtypes.append(dtype.newbyteorder('<'))
    header = json.dumps({"v": version, "endian": "le", "count": 1, "fields": descriptors}, separators=(",", ":")).encode()
    if len(header) > HEADER_SIZE:
        raise ValueError("Sonic JSON header too large")
    return header.ljust(HEADER_SIZE, b'\0'), tuple(dtypes)


def encode_sonic_json(fields, *, topic="pose", version=4):
    if not fields or len(fields) > MAX_FIELDS:
        raise ValueError('Invalid field count')
    arrays = tuple(np.asarray(value) for value in fields.values())
    layout = tuple((name, array.dtype, array.shape) for name, array in zip(fields, arrays))
    header, dtypes = _sonic_json_layout(layout, version)
    if sum(array.nbytes for array in arrays) > MAX_ARRAY_BYTES:
        raise ValueError('Packet too large')
    payload = b''.join(array.astype(dtype, copy=False).tobytes(order='C') for array, dtype in zip(arrays, dtypes))
    return topic.encode() + header + payload


def _decode_fields(descriptors, payload, byte_order="little"):
    if len(descriptors) > MAX_FIELDS or len(payload) > MAX_ARRAY_BYTES or byte_order not in {"little", "big"}:
        raise ValueError("Invalid packet layout")
    result, offset = {}, 0
    for descriptor in descriptors:
        name = descriptor["name"]
        if not name or name == "__schema" or name in result:
            raise ValueError("Invalid/duplicate field name")
        spec = TensorSpec(WIRE_DTYPES[descriptor["dtype"]], tuple(descriptor["shape"]))
        dtype = np.dtype(spec.dtype).newbyteorder("<" if byte_order == "little" else ">")
        size = math.prod(spec.shape) * dtype.itemsize
        if offset + size > len(payload):
            raise ValueError("Truncated field")
        result[name] = np.frombuffer(payload[offset:offset + size], dtype=dtype).reshape(spec.shape).astype(dtype.newbyteorder("="), copy=True)
        offset += size
    if offset != len(payload):
        raise ValueError("Unexpected trailing bytes")
    return result



def decode_sonic_json(packet, *, topic="pose"):
    if not packet.startswith(topic.encode()):
        raise ValueError("Unexpected topic")
    data = packet[len(topic.encode()):]
    if len(data) < HEADER_SIZE:
        raise ValueError("Truncated Sonic JSON header")
    header = json.loads(data[:HEADER_SIZE].rstrip(b"\0"))
    order = {"le": "little", "be": "big"}.get(header.get("endian"))
    if order is None:
        raise ValueError("Unknown Sonic JSON byte order")
    fields = _decode_fields(header["fields"], data[HEADER_SIZE:], order)
    return fields, {"version": header["v"], "count": header.get("count", 1)}



def decode_sonic_telemetry(packet, *, topic='g1_debug'):
    """Original Sonic flat MessagePack feedback, also emitted by telemetry.sonic_zmq.

    Keep wire field names (including *_measured) and the frame index. This
    protocol has no usable monotonic timestamp; the receiver uses arrival time.
    """
    prefix = topic.encode()
    if not packet.startswith(prefix) or len(packet) > MAX_ARRAY_BYTES:
        raise ValueError('Invalid Sonic telemetry packet')
    message = msgpack.unpackb(packet[len(prefix):], raw=False)
    if not isinstance(message, dict) or not 1 <= len(message) <= MAX_FIELDS:
        raise ValueError('Invalid Sonic telemetry fields')
    index = message.get('index')
    if type(index) is not int or not 0 <= index < 2**63:
        raise ValueError('Sonic telemetry requires an integer frame index')
    fields = {}
    for name, value in message.items():
        if name == 'control_loop_type':
            continue  # Informational string, not a tensor stream.
        if not isinstance(name, str) or not name or len(name.encode()) > 256:
            raise ValueError('Invalid Sonic telemetry field name')
        array = np.asarray(value)
        if array.dtype.kind not in 'fiu' or array.ndim > 1:
            raise ValueError('Sonic telemetry fields must be numeric scalars or vectors')
        array = array.astype(np.int64 if name == 'index' else np.float32).reshape(-1)
        TensorSpec(array.dtype.name, array.shape).validate(array)
        fields[name] = array
    return fields, {'sequence': index, 'source_ns': None}
