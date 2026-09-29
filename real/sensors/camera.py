"""Camera acquisition modules with optional SDK dependencies and bounded I/O."""
from __future__ import annotations

import socket
import struct

import numpy as np

from ..core.runtime import Module


def prepare_rgb(image, options):
    import cv2
    if "crop" in options:
        x, y, width, height = options["crop"]
        image = image[y:y + height, x:x + width]
        if image.shape[:2] != (height, width):
            raise ValueError("Camera crop outside image")
    if "size" in options:
        image = cv2.resize(image, tuple(options["size"]))
    return image


class OpenCVCamera(Module):
    required_outputs = ("rgb",)

    def bind(self, ctx):
        self.rgb_output = ctx.bind_write(['rgb'] if 'rgb' in ctx.outputs else [])

    def start(self, ctx):
        import cv2
        self.cv2 = cv2
        source = self.options.get("source", 0)
        if isinstance(source, str):
            params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.options.get("timeout_ms", 1000)),
                      cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.options.get("timeout_ms", 1000))]
            self.camera = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
        else:
            self.camera = cv2.VideoCapture(int(source))
        if not self.camera.isOpened():
            raise RuntimeError(f"Could not open camera {source}")
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ctx.report("WAITING", f"Camera {source} opened; waiting for RGB frames")

    def process(self, ctx):
        success, image = self.camera.read()
        if not success:
            self.rgb_output.write((None,))
            ctx.report("DEGRADED", "Camera read failed")
            return
        rgb = prepare_rgb(self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB), self.options)
        self.rgb_output.write((rgb,), source_ns=ctx.now_ns, clock="host.receive")
        ctx.report("OK", f"Receiving RGB frames ({rgb.shape[1]}x{rgb.shape[0]})")

    def stop(self, ctx):
        if getattr(self, "camera", None) is not None:
            self.camera.release()


class ZedCamera(Module):
    required_outputs = ("rgb",)

    def bind(self, ctx):
        self.has_depth = 'depth' in ctx.outputs
        self.output = ctx.bind_write(('rgb', 'depth') if self.has_depth else ('rgb',))

    def start(self, ctx):
        import pyzed.sl as sl
        self.sl = sl
        self.camera = sl.Camera()
        params = sl.InitParameters()
        params.camera_resolution = getattr(sl.RESOLUTION, self.options.get("resolution", "HD720"))
        params.camera_fps = int(self.options.get("fps", 30))
        if self.has_depth:
            params.coordinate_units = sl.UNIT.MILLIMETER
        if self.camera.open(params) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("ZED camera open failed")
        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.parameters = sl.RuntimeParameters()
        ctx.report("WAITING", "ZED opened; waiting for RGB" + (" and depth frames (mm)" if self.has_depth else " frames"))

    def process(self, ctx):
        import cv2
        if self.camera.grab(self.parameters) != self.sl.ERROR_CODE.SUCCESS:
            self.output.write((None,) * len(self.output))
            ctx.report("DEGRADED", "ZED grab failed")
            return
        self.camera.retrieve_image(self.image, self.sl.VIEW.LEFT)
        rgb = cv2.cvtColor(self.image.get_data(), cv2.COLOR_BGRA2RGB)
        stamp = self.camera.get_timestamp(self.sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
        values = [prepare_rgb(rgb, self.options)]
        if self.has_depth:
            self.camera.retrieve_measure(self.depth, self.sl.MEASURE.DEPTH)
            # Depth validity is per pixel; invalid depth pixels remain NaN in raw data.
            depth = self.depth.get_data()
            # Match the resized RGB image while preserving depth samples/NaNs.
            if "crop" in self.options:
                x, y, width, height = self.options["crop"]
                depth = depth[y:y + height, x:x + width]
            if "size" in self.options:
                depth = cv2.resize(depth, tuple(self.options["size"]), interpolation=cv2.INTER_NEAREST)
            values.append(depth)
        self.output.write(values, source_ns=stamp, clock="zed")
        ctx.report("OK", f"Receiving ZED RGB{' + depth (mm)' if self.has_depth else ''} ({values[0].shape[1]}x{values[0].shape[0]})")

    def stop(self, ctx):
        if getattr(self, "camera", None):
            self.camera.close()


class JpegZmqCamera(Module):
    """JPEG subscriber owning an optional OPEN_CAMERA TCP service lease.

    Packet format: 8-byte prefix, int32 size, int64 microseconds, JPEG.
    Set ``service`` options to start/hold the remote camera while subscribed.
    """
    required_outputs = ("rgb",)

    def bind(self, ctx):
        self.rgb_output = ctx.bind_write(['rgb'] if 'rgb' in ctx.outputs else [])

    def start(self, ctx):
        import cv2
        import zmq
        self.cv2, self.zmq = cv2, zmq
        self.context = self.socket = self.service_socket = None
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.SUB)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.RCVHWM, 4)
            self.socket.setsockopt(zmq.MAXMSGSIZE, 16 * 1024 * 1024)
            self.socket.setsockopt(zmq.SUBSCRIBE, self.options.get("prefix", "").encode())
            self.socket.connect(self.options["endpoint"])
            if self.options.get("service") is not None:
                service = self.options["service"]
                ctx.logger.info("Opening camera service at %s:%s", service["host"], service.get("port", 13579))
                self._open_service(service)
            ctx.report("WAITING", f"Waiting for JPEG frames at {self.options['endpoint']}")
        except BaseException:
            self.stop(ctx)
            raise

    def _open_service(self, options):
        def compact_string(value):
            data = value.encode()
            if len(data) > 255:
                raise ValueError("Camera string exceeds wire limit")
            return bytes([len(data)]) + data
        numbers = [options.get("width", 2560), options.get("height", 720), options.get("fps", 60), options.get("bitrate", 4000000), 0, 0, 0]
        payload = b"\xca\xfe\x01" + struct.pack("<7i", *numbers) + compact_string(options.get("camera", "ZED")) + compact_string("")
        command = b"OPEN_CAMERA"
        body = struct.pack("<i", len(command)) + command + struct.pack("<i", len(payload)) + payload
        self.service_socket = socket.create_connection((options["host"], options.get("port", 13579)), timeout=options.get("timeout_seconds", 2))
        self.service_socket.sendall(struct.pack(">I", len(body)) + body)

    def process(self, ctx):
        packet = None
        for _ in range(8):
            try:
                packet = self.socket.recv(flags=self.zmq.NOBLOCK)
            except self.zmq.Again:
                break
        if packet is None:
            return
        try:
            if len(packet) < 20:
                raise ValueError("Truncated camera header")
            size, stamp_us = struct.unpack_from("<iq", packet, 8)
            if size != len(packet) - 20 or size <= 0:
                raise ValueError("Camera JPEG extent mismatch")
            image = self.cv2.imdecode(np.frombuffer(packet[20:], np.uint8), self.cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("JPEG decoding failed")
            rgb = prepare_rgb(self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB), self.options)
            self.rgb_output.write((rgb,), source_ns=stamp_us * 1000, clock=self.options.get("clock", "camera.remote"))
            ctx.report("OK", "Receiving JPEG camera frames")
        except ValueError as error:
            self.rgb_output.write((None,))
            ctx.report("DEGRADED", str(error))

    def stop(self, ctx):
        if getattr(self, "service_socket", None) is not None:
            self.service_socket.close()
            self.service_socket = None
        if getattr(self, "socket", None) is not None:
            self.socket.close(linger=0)
            self.socket = None
        if getattr(self, "context", None) is not None:
            self.context.term()
            self.context = None
