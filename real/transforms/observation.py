"""Explicit temporal alignment; source clocks are never assumed synchronized."""
from ..core.protocol import Observation


class ObservationSelector:
    """Optional receive-time alignment helper, called by its consuming module."""

    def __init__(self, inputs, options):
        self.ports = tuple(inputs)
        self.anchor = options.get("anchor", next(iter(inputs), None))
        if self.anchor not in inputs:
            raise ValueError("Observation anchor must name an input")
        self.max_age_ns = int(options.get("max_age_ms", 200) * 1e6)
        self.max_skew_ns = int(options.get("max_skew_ms", 100) * 1e6)
        self.required = set(options.get("required", inputs))
        self.instruction = options.get("instruction", "")
        if not self.required <= inputs.keys():
            raise ValueError("Unknown required observation input")

    def bind(self, ctx):
        self.reader = ctx.bind_read(self.ports)
        self.anchor_index = self.ports.index(self.anchor)

    def select(self, ctx):
        histories = self.reader.history()
        anchor_history = histories[self.anchor_index]
        anchor = anchor_history[-1] if anchor_history else None
        if anchor is None or not anchor.fresh(ctx.now_ns, self.max_age_ns):
            ctx.report("DEGRADED", "Missing or stale observation anchor")
            return
        selected, alignment = {}, {}
        for port, history in zip(self.ports, histories):
            # Latest publication invalidation wins; never recover an invalidated stream
            # merely because historical valid samples remain in the ring.
            latest = history[-1] if history else None
            if latest is None or not latest.fresh(ctx.now_ns, self.max_age_ns):
                sample = None
            else:
                candidates = [s for s in history if s.valid and 0 <= anchor.received_ns - s.received_ns <= self.max_skew_ns]
                sample = candidates[-1] if candidates else None
            if sample is None:
                if port in self.required:
                    ctx.report("DEGRADED", f"No aligned sample for {port}")
                    return
                alignment[port] = {"missing": True}
                continue
            selected[port] = sample
            alignment[port] = {"sequence": sample.sequence, "skew_ns": anchor.received_ns - sample.received_ns, "basis": "host_receive_time"}
        return Observation(f"{ctx.id}:{anchor.sequence}:{ctx.write_generation}", selected, anchor.received_ns, self.instruction, alignment)

