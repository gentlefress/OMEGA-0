"""Bound adapters for configured tensor groups."""
from __future__ import annotations

import numpy as np

from .protocol import CommandMetadata, StopCommand


def field_ports(bindings, prefix):
    return {port[len(prefix) + 1:]: port for port in bindings if port.startswith(prefix + '.')}


class FieldReader:
    """Resolve a configured prefix once, then read via a cached stream handle."""

    def __init__(self, ctx, prefix):
        ports = field_ports(ctx.inputs, prefix)
        self.fields = tuple(ports)
        self.port = ctx.bind_read(ports.values())
        self.ctx = ctx

    def __bool__(self):
        return bool(self.fields)

    def samples(self):
        return dict(zip(self.fields, self.port.read()))

    def read(self, *, max_age_ns=None, coherent=True):
        """Return tensors, or {} for missing/invalid/stale/mixed-batch fields."""
        samples = self.port.read()
        if not samples or any(s is None or not s.valid or
                (max_age_ns is not None and not s.fresh(self.ctx.now_ns, max_age_ns)) for s in samples):
            return {}
        if coherent and any(s.batch_id != samples[0].batch_id for s in samples[1:]):
            return {}
        return dict(zip(self.fields, (sample.value for sample in samples)))


class FieldWriter:
    """Pack local dictionaries in a prebound order; store only tensor fields.

    Omitted configured fields become invalid in the same commit. Per-field
    metadata retains each source clock/timestamp. Dictionary packing uses local
    field names; no datastore-name resolution or lock sorting occurs here.
    """

    def __init__(self, ctx, prefix):
        ports = field_ports(ctx.outputs, prefix)
        self.fields = tuple(ports)
        self.port = ctx.bind_write(ports.values())

    def __bool__(self):
        return bool(self.fields)

    def write(self, values, *, metadata_by_field=None, **metadata):
        if not self:
            return None
        values = tuple(np.asarray(values[name]) if name in values and values[name] is not None else None
                       for name in self.fields)
        overrides = None if metadata_by_field is None else tuple(metadata_by_field.get(name) for name in self.fields)
        samples = self.port.write(values, field_metadata=overrides, **metadata)
        return None if samples is None else dict(zip(self.fields, samples))

    def invalidate(self):
        return self.port.write((None,) * len(self.port))


def command_metadata(command):
    return CommandMetadata(command.session_id, command.sequence, command.valid_until_ns,
                           command.source, command.step, isinstance(command, StopCommand))


# Named convenience adapters. Repeated module I/O should cache FieldReader /
# FieldWriter during bind() rather than resolving a prefix on every operation.
def write(ctx, prefix, values, **metadata):
    return FieldWriter(ctx, prefix).write(values, **metadata)


def read(ctx, prefix, **options):
    return FieldReader(ctx, prefix).read(**options)


def field_samples(ctx, prefix):
    return FieldReader(ctx, prefix).samples()


def invalidate(ctx, prefix):
    return FieldWriter(ctx, prefix).invalidate()

