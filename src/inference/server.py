"""WAM HTTP routes, session management, and server lifecycle."""
from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

from omega_real.core.protocol import VERSION, ActionChunk, PolicyRequest, dumps, loads


@dataclass(frozen=True)
class Response:
    body: bytes
    status: int = 200
    content_type: str = "application/json"


class HTTPError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class Service:
    """One model instance, one active session, one outstanding inference.

    Reset invalidates in-flight responses immediately. The actual model reset runs
    under the inference lock before the next prediction. Old sessions cannot rearm.
    """
    def __init__(self, inference):
        self.inference, self.spec = inference, inference.spec
        self.state_lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.session = None
        self.generation = 0
        self.model_generation = -1
        self.retired = set()
        self.seen_requests = set()
        self.ready = True

    def reset(self, session_id):
        if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
            raise ValueError("Invalid session ID")
        with self.state_lock:
            if not self.ready:
                raise ValueError("Inference service is closed")
            if session_id in self.retired:
                raise ValueError("Retired session cannot be resumed")
            if self.session == session_id:
                return {"session_id": session_id, "generation": self.generation}
            if len(self.retired) >= 4096:
                raise ValueError("Session limit reached; restart the server")
            if self.session:
                self.retired.add(self.session)
            self.session = session_id
            self.generation += 1
            self.seen_requests.clear()
            return {"session_id": session_id, "generation": self.generation}

    def predict(self, request):
        if not isinstance(request, PolicyRequest) or request.action_spec_id != self.spec.id:
            raise ValueError("Unsupported policy request/specification")
        if not self.inference_lock.acquire(blocking=False):
            raise BlockingIOError("Model is busy")
        try:
            with self.state_lock:
                if not self.ready or request.session_id != self.session:
                    raise ValueError("Inactive policy session")
                if request.request_id in self.seen_requests:
                    raise ValueError("Duplicate request ID")
                if len(self.seen_requests) >= 100000:
                    raise ValueError("Session request limit reached; create a new session")
                self.seen_requests.add(request.request_id)
                generation = self.generation
            if self.model_generation != generation:
                self.inference.reset(request.session_id)
                self.model_generation = generation
            actions = self.inference.predict(request)
            result = ActionChunk(request.session_id, request.request_id, request.observation.observation_id,
                                 self.spec.id, request.start_step, self.spec.period_ns, {"action": actions})
            result.validate(self.spec)
            with self.state_lock:
                if not self.ready or generation != self.generation:
                    raise ValueError("Response invalidated by session reset")
            return result
        finally:
            self.inference_lock.release()

    def handle(self, method, path, body):
        """Omega-Zero routes and its versioned chunk-policy codec."""
        try:
            if method == "GET" and path == "/health":
                result = {"ready": self.ready, "protocol_version": VERSION, "action_spec": self.spec}
            elif method == "POST" and path == "/reset":
                result = self.reset(loads(body)["session_id"])
            elif method == "POST" and path == "/act":
                result = self.predict(loads(body))
            else:
                raise HTTPError(404, "Unknown route")
            return Response(dumps(result))
        except BlockingIOError as error:
            raise HTTPError(429, str(error)) from error
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPError(400, str(error)) from error

    def close(self):
        # Invalidate outstanding work before waiting for the inference owner.
        with self.state_lock:
            self.ready = False
            self.generation += 1
        with self.inference_lock:
            close = getattr(self.inference, "close", None)
            if close is not None:
                close()


def handler_for(service, *, max_request_bytes=128 * 1024 * 1024, request_timeout=2):
    """Adapt service.handle(method, path, body) -> Response to stdlib HTTP."""
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(request_timeout)

        def log_message(self, *args):
            pass

        def reply(self, response):
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(response.body)))
            self.end_headers()
            self.wfile.write(response.body)

        def dispatch(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if self.headers.get("Transfer-Encoding") or not 0 <= size <= max_request_bytes:
                    raise HTTPError(400, "Invalid request size or transfer encoding")
                body = self.rfile.read(size) if size else b""
                if len(body) != size:
                    raise HTTPError(400, "Truncated request")
                response = service.handle(self.command, self.path, body)
                if not isinstance(response, Response):
                    raise TypeError("Service must return Response")
            except HTTPError as error:
                response = Response(json.dumps({"error": str(error)}).encode(), error.status)
            except (ValueError, TimeoutError) as error:
                response = Response(json.dumps({"error": str(error)}).encode(), 400)
            except Exception as error:
                response = Response(json.dumps({"error": f"Request failed: {type(error).__name__}: {error}"}).encode(), 500)
            try:
                self.reply(response)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass  # Client disconnected while the service was working.

        do_GET = dispatch
        do_POST = dispatch
        do_PUT = dispatch
        do_DELETE = dispatch
        do_PATCH = dispatch

    return Handler


class Server:
    """Own HTTP lifecycle independently of the real-world module runtime.

    Use serve_forever() for a dedicated process, or start()/stop() to embed the
    server in an application. Both paths close the service and request workers.
    """
    def __init__(self, service, *, host="127.0.0.1", port=8014,
                 max_request_bytes=128 * 1024 * 1024, request_timeout=2):
        self.service = service
        self.host, self.port = host, int(port)
        self.max_request_bytes, self.request_timeout = int(max_request_bytes), float(request_timeout)
        self.server = None
        self.thread = None
        self.stop_event = threading.Event()
        self.closed = False

    def _open(self):
        if self.server is not None or self.closed:
            raise RuntimeError("Server already started or closed")
        try:
            self.server = ThreadingHTTPServer((self.host, self.port), handler_for(self.service,
                max_request_bytes=self.max_request_bytes, request_timeout=self.request_timeout))
        except BaseException:
            self.stop()
            raise
        # Join active requests on shutdown so model resources outlive inference.
        self.server.daemon_threads = False
        self.server.timeout = 0.05

    def _run(self):
        while not self.stop_event.is_set():
            self.server.handle_request()

    def start(self):
        self._open()
        self.thread = threading.Thread(target=self._run, name="inference-http")
        try:
            self.thread.start()
        except BaseException:
            self.thread = None
            self.stop()
            raise
        return self

    def serve_forever(self):
        self._open()
        try:
            self._run()
        finally:
            self.stop()

    def stop(self):
        if self.closed:
            return
        self.closed = True
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        try:
            close = getattr(self.service, "close", None)
            if close is not None:
                close()
        finally:
            if self.server is not None:
                self.server.server_close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
