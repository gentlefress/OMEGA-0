"""Topic publishers and direct topic.field subscribers, with lazy pyzmq import."""
from __future__ import annotations

from collections import deque
import logging

from ..core.streams import FieldWriter
from ..core.protocol import TensorSpec
from ..core.streams import field_ports
from ..core.runtime import Module
from .codecs import decode_sonic_telemetry, encode_sonic_json, sonic_wire_fields


class ZmqReceiver(Module):
    """Publish every received field under <topic>.<field>, without output routing."""

    def configure(self, options):
        super().configure(options)
        if {'fields', 'outputs'} & options.keys():
            raise ValueError('Receiver fields are discovered from packets; remove fields')
        if options.get('format', 'sonic_telemetry') != 'sonic_telemetry':
            raise ValueError('Receiver supports only sonic_telemetry')
        self.topic = options['topic']
        self.max_fields = int(options.get('max_fields', 128))
        if not isinstance(self.topic, str) or not self.topic or len(self.topic.encode()) > 128:
            raise ValueError('Receiver requires a nonempty topic of at most 128 bytes')
        if not 1 <= self.max_fields <= 128:
            raise ValueError('Receiver max_fields must be between 1 and 128')

    def output_namespaces(self):
        return {self.topic + '.': self.max_fields}

    def start(self, ctx):
        if ctx.outputs:
            raise ValueError('Receiver output names come from topic.field; remove outputs')
        import zmq
        self.zmq = zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVHWM, self.options.get('high_water_mark', 64))
        self.socket.setsockopt(zmq.SUBSCRIBE, self.topic.encode())
        self.socket.setsockopt(zmq.MAXMSGSIZE, 64 * 1024 * 1024)
        endpoint = self.options['endpoint']
        (self.socket.bind if self.options.get('bind', False) else self.socket.connect)(endpoint)
        self.previous, self.descriptors = {}, {}
        self.writers = {}
        self.names = set()
        self.generation = ctx.generation
        # Some controllers start publishing telemetry only after receiving enable.
        # Silence before the first packet is startup, not a lost connection.
        self.last_packet_ns = None
        self.disconnected = False
        ctx.report('WAITING', f'Waiting for first {self.topic} packet at {endpoint}; startup commands are not blocked by telemetry')

    def _invalidate(self, ctx, names=None):
        for name in self.names if names is None else names:
            self.writers[name].write((None,))
            self.previous.pop(name, None)

    def process(self, ctx):
        if self.generation != ctx.generation:
            self.previous.clear()
            self.generation = ctx.generation
        for _ in range(self.options.get('max_batch', 64)):
            try:
                packet = self.socket.recv(flags=self.zmq.NOBLOCK)
            except self.zmq.Again:
                break
            try:
                values, meta = decode_sonic_telemetry(packet, topic=self.topic)
                if len(self.names | values.keys()) > self.max_fields:
                    raise ValueError('Receiver field limit exceeded')
                # Validate the complete layout before publishing any field.
                for name, value in values.items():
                    if not isinstance(name, str) or not name or len((self.topic + '.' + name).encode()) > 256:
                        raise ValueError('Invalid received field name')
                    if value is not None:
                        spec = self.descriptors.get(name, TensorSpec(value.dtype.name, value.shape))
                        spec.validate(value, finite=False)
                        self.descriptors[name] = spec
                self._invalidate(ctx, self.names - values.keys())
                for name, value in values.items():
                    output = self.writers.get(name)
                    if output is None:
                        output = self.writers[name] = ctx.bind_write([name], namespace=self.topic)
                    if self.previous.get(name) == meta['sequence']:
                        continue  # Duplicate frames must not refresh old measurements.
                    output.write((value,), clock='local.monotonic')
                    self.previous[name] = meta['sequence']
                    self.names.add(name)
                self.last_packet_ns, self.disconnected = ctx.now_ns, False
                ctx.report('OK', f'Receiving {self.topic} packets')
            except (ValueError, KeyError, TypeError, OverflowError) as error:
                self._invalidate(ctx)
                ctx.report('DEGRADED', f'Malformed packet: {error}')
        timeout = int(self.options.get('disconnect_ms', 1000) * 1e6)
        if self.last_packet_ns is not None and ctx.now_ns - self.last_packet_ns > timeout and not self.disconnected:
            self.disconnected = True
            self._invalidate(ctx)
            ctx.report('DEGRADED', 'Telemetry disconnected; hardware output latched off')
            ctx.emit('estop', {'reason': 'telemetry disconnected'})

    def stop(self, ctx):
        if getattr(self, 'socket', None) is not None:
            self.socket.close(linger=0)
            self.socket = None
        if getattr(self, 'context', None) is not None:
            self.context.term()
            self.context = None


class ZmqSender(Module):
    """Publish one topic, or ordered topic groups, through one owned PUB socket.

    With ``topics: {pose: {version: 3}, command: {version: 1}}``, inputs are
    ``pose.<field>`` and ``command.<field>``. Mapping order is send order. Each
    topic keeps its own command batches, expiry, optional fields and receipts.
    A single-topic sender still accepts ``topic`` and unprefixed input fields.
    """
    queued_inputs = True

    def configure(self, options):
        super().configure(options)
        if {'shutdown', 'versions'} & options.keys():
            raise ValueError('Use per-topic version options and producer-owned stop commands')
        if 'topics' in options:
            if {'topic', 'version', 'optional_inputs'} & options.keys():
                raise ValueError('Put version and optional_inputs inside each topics entry')
            topics = options['topics']
            if not isinstance(topics, dict) or not topics:
                raise ValueError('topics must be a nonempty ordered mapping')
            for topic, settings in topics.items():
                if not isinstance(topic, str) or not topic or '.' in topic:
                    raise ValueError('Topic groups require nonempty names without dots')
                if not isinstance(settings, dict) or set(settings) - {'version', 'optional_inputs'}:
                    raise ValueError('Topic options support version and optional_inputs')
        elif not isinstance(options.get('topic'), str) or not options['topic']:
            raise ValueError('Sender requires a topic or topics mapping')
        self.format = options.get('format', 'sonic_json')
        if self.format != 'sonic_json':
            raise ValueError('Sender supports only sonic_json')

    def bind(self, ctx):
        if not ctx.inputs:
            raise ValueError('Sender requires at least one input')
        if not set(field_ports(ctx.outputs, 'sent')) <= ctx.inputs.keys():
            raise ValueError('Receipt outputs must be sent.<input field>')
        if len(field_ports(ctx.outputs, 'sent')) != len(ctx.outputs):
            raise ValueError('Sender outputs must be named sent.<input field>')
        grouped = 'topics' in self.options
        topics = self.options['topics'] if grouped else {self.options['topic']: self.options}
        self.topics = [_SenderTopic(self, ctx, topic, settings, grouped)
                       for topic, settings in topics.items()]
        if grouped and any(port.split('.', 1)[0] not in topics or '.' not in port for port in ctx.inputs):
            raise ValueError('Every grouped sender input must be <topic>.<field>')

    def start(self, ctx):
        import zmq
        self.zmq = zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDHWM, self.options.get('high_water_mark', 4))
        endpoint = self.options['endpoint']
        (self.socket.bind if self.options.get('bind', True) else self.socket.connect)(endpoint)
        for topic in self.topics:
            topic.start(ctx)
        operation = 'Bound' if self.options.get('bind', True) else 'Connecting'
        names = ','.join(topic.topic for topic in self.topics)
        ctx.report('WAITING', f'{operation} {endpoint}, topics={names}, format={self.format}; waiting for inputs')

    def process(self, ctx):
        for topic in self.topics:
            if topic.process(ctx) is False:
                break
        # A successful command topic must not hide a blocked pose topic.
        problems = [topic for topic in self.topics if topic.status[0] == 'DEGRADED']
        if problems:
            ctx.report('DEGRADED', '; '.join(f'{topic.topic}: {topic.status[1]}' for topic in problems))
        else:
            active = [topic.topic for topic in self.topics if topic.sent_count]
            waiting = [topic.topic for topic in self.topics if not topic.sent_count]
            message = ('Published ' + ', '.join(active) + ' packets (delivery not acknowledged)') if active else ''
            if waiting:
                message += ('; ' if message else '') + 'waiting for ' + ', '.join(waiting) + ' inputs'
            ctx.report('OK' if active else 'WAITING', message)

    def log_status(self, ctx):
        if not getattr(self, 'socket', None) or ctx.runtime.health[ctx.id].level == 'FAULT':
            return super().log_status(ctx)
        for topic in self.topics:
            level, message = topic.status
            ctx.logger.log(logging.WARNING if level == 'DEGRADED' else logging.INFO,
                "%s | %s: %s | packets sent=%d", topic.topic, level, message, topic.sent_count)

    def stop(self, ctx):
        try:
            if getattr(self, 'socket', None) is not None:
                self.process(ctx)
        finally:
            if getattr(self, 'socket', None) is not None:
                self.socket.close(linger=0)
                self.socket = None
            if getattr(self, 'context', None) is not None:
                self.context.term()
                self.context = None


class _SenderTopic:
    """Per-topic packet assembly; the enclosing sender owns the socket/thread."""

    def __init__(self, sender, ctx, topic, options, grouped):
        self.sender, self.topic = sender, topic
        ports = field_ports(ctx.inputs, topic) if grouped else {name: name for name in ctx.inputs}
        if not ports:
            raise ValueError(f'Sender topic {topic} requires inputs')
        self.input_names = tuple(ports)
        self.input = ctx.bind_read(ports.values())
        self.sent_out = FieldWriter(ctx, 'sent.' + topic if grouped else 'sent')
        self.optional = set(options.get('optional_inputs', []))
        if not self.optional <= ports.keys():
            raise ValueError('Unknown optional sender input')
        self.version = options.get('version', 1)

    def start(self, ctx):
        self.ready_at = ctx.now_ns + int(self.sender.options.get('warmup_ms', 300) * 1e6)
        self.pending = {port: deque(maxlen=ctx.queue_capacity) for port in self.input_names}
        self.command_ports = set()
        self.last_signature = None
        self.sent_count = 0
        self.status = ('WAITING', 'Waiting for inputs')

    @staticmethod
    def _identity(sample):
        return sample.batch_id, sample.spec_id, sample.command

    def _send(self, ctx, samples):
        commands = [sample for sample in samples.values() if sample is not None and sample.command is not None]
        command = commands[0].command if commands else None
        stopping = command is not None and command.stop
        if commands and any(self._identity(sample) != self._identity(commands[0]) for sample in commands):
            raise ValueError('Sender cannot combine different command publications')
        if command is not None and not stopping and ctx.now_ns >= command.valid_until_ns:
            self.status = ('DEGRADED', 'Expired command discarded')
            return True
        if any(sample is None or not sample.valid
               for port, sample in samples.items() if port not in self.optional):
            self.status = ('DEGRADED', 'Required packet field unavailable')
            return False
        samples = {name: sample for name, sample in samples.items() if sample is not None}
        fields = {name: sample.value for name, sample in samples.items() if sample.valid}
        if not fields:
            self.status = ('DEGRADED', 'No packet fields available')
            return True
        stamps = {sample.source_ns for sample in samples.values()}
        source_ns = next(iter(stamps)) if len(stamps) == 1 else None
        fields = sonic_wire_fields(fields, topic=self.topic)
        packet = encode_sonic_json(fields, topic=self.topic, version=self.version)
        try:
            send = lambda: self.sender.socket.send(packet, flags=self.sender.zmq.NOBLOCK)
            sent = ctx.write_shutdown(send) if stopping else ctx.write_output(send, generation=ctx.write_generation)
        except self.sender.zmq.Again:
            self.backpressured = True
            self.status = ('DEGRADED', 'Transport queue full')
            return False
        if sent:
            self.sent_count += 1
            self.status = ('OK', 'Packets published (delivery not acknowledged)')
            self.sent_out.write(fields, source_ns=source_ns, spec_id=commands[0].spec_id if commands else '', command=command, batch_id=commands[0].batch_id if commands else '', metadata_by_field={name: {'source_ns': sample.source_ns, 'clock': sample.clock}
                                   for name, sample in samples.items()})
        else:
            reason = 'runtime is stopping' if ctx.stopping else 'not armed; launch with --arm' if not ctx.runtime.hardware_armed else 'output latched off or command invalidated'
            self.status = ('DEGRADED', f'Hardware output inhibited: {reason}')
        return True

    def process(self, ctx):
        self.backpressured = False
        for port, batch in zip(self.input_names, self.input.drain()):
            pending = self.pending[port]
            for sample in batch:
                if sample.command is not None:
                    self.command_ports.add(port)
                    # Unavailable optional fields still identify their command batch.
                    pending.append(sample)
                elif not sample.valid:
                    stops = [s for s in pending if s.command is not None and s.command.stop]
                    pending.clear()
                    pending.extend(stops)
                elif port in self.command_ports:
                    raise ValueError('Command fields must retain their command metadata')
        if self.command_ports:
            return self._send_commands(ctx)
        if ctx.now_ns < self.ready_at:
            return
        samples = dict(zip(self.input_names, self.input.read()))
        if any(sample is None for port, sample in samples.items() if port not in self.optional):
            return
        signature = tuple(sample.sequence if sample is not None else None for sample in samples.values())
        if signature != self.last_signature:
            if not self._send(ctx, samples):
                return not self.backpressured
            self.last_signature = signature

    def _send_commands(self, ctx):
        latest = dict(zip(self.input_names, self.input.read()))
        self.command_ports.update(port for port, sample in latest.items()
                                  if sample is not None and sample.command is not None)
        ports = sorted(self.command_ports)
        # Normal live control has exactly one queued publication per field.
        # Match it directly; retain the general join below for backlog/partial batches.
        if all(len(self.pending[port]) == 1 for port in ports):
            samples = {port: self.pending[port][0] for port in ports}
            identity = self._identity(samples[ports[0]])
            if any(self._identity(sample) != identity for sample in samples.values()):
                return
            if ctx.now_ns < self.ready_at and not identity[2].stop:
                return
            if not self._send(ctx, {**latest, **samples}):
                return not self.backpressured
            for port in ports:
                self.pending[port].clear()
            return
        common = None
        for port in ports:
            identities = {self._identity(sample) for sample in self.pending[port]}
            common = identities if common is None else common & identities
        if not common:
            return  # A producer may still be publishing another field this cycle.
        ordered = list(dict.fromkeys(self._identity(sample) for sample in self.pending[ports[0]]
                                     if self._identity(sample) in common))
        for identity in ordered:
            command = identity[2]
            if ctx.now_ns < self.ready_at and not command.stop:
                # Keep the last complete ordinary command until socket warmup.
                if identity == ordered[-1]:
                    continue
            else:
                samples = dict(latest)
                for port in ports:
                    samples[port] = next(sample for sample in self.pending[port] if self._identity(sample) == identity)
                if not self._send(ctx, samples):
                    return not self.backpressured  # Only transport backpressure holds later topics.
            for port in ports:
                self.pending[port] = deque((sample for sample in self.pending[port]
                    if self._identity(sample) != identity), maxlen=ctx.queue_capacity)
