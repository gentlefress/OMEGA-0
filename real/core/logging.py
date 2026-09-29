"""Module labels and asynchronous console output matching sonic_deploy."""
from contextlib import contextmanager
import logging
from logging.handlers import QueueHandler, QueueListener
import os
import queue
import sys

LOG_LEVELS = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING,
              'error': logging.ERROR, 'off': logging.CRITICAL + 1}


class ModuleLogger(logging.LoggerAdapter):
    """Keep configured module IDs in both terminal text and log record metadata."""
    def process(self, message, kwargs):
        message, kwargs = super().process(message, kwargs)
        return f"[{self.extra['module_id']}] {message}", kwargs


class DeployFormatter(logging.Formatter):
    """[YYYY-MM-DD HH:MM:SS.mmm] [info] [component] message."""
    colors = {logging.DEBUG: '\033[36m', logging.INFO: '\033[32m',
              logging.WARNING: '\033[1;33m', logging.ERROR: '\033[1;31m',
              logging.CRITICAL: '\033[1;31m'}

    def __init__(self, *, color=False):
        super().__init__()
        self.color = color

    def format(self, record):
        stamp = self.formatTime(record, '%Y-%m-%d %H:%M:%S') + f'.{int(record.msecs):03d}'
        level = record.levelname.lower()
        if self.color and record.levelno in self.colors:
            level = self.colors[record.levelno] + level + '\033[0m'
        message = super().format(record)
        if not hasattr(record, 'module_id'):
            message = f'[{record.name}] {message}'
        return f'[{stamp}] [{level}] {message}'


class RecentQueueHandler(QueueHandler):
    """Bound logging memory without blocking control workers on console output."""
    def enqueue(self, record):
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            try:
                self.queue.get_nowait()  # Match sonic_deploy's overrun-oldest policy.
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(record)
            except queue.Full:
                pass  # Another producer won the slot; logging never blocks it.


class _QueueListener(QueueListener):
    def enqueue_sentinel(self):
        # Drain during CLI cleanup, after runtime workers have stopped.
        self.queue.put(self._sentinel)


@contextmanager
def console_logging(level='info', *, stream=None):
    """Own CLI logging for one run; library users keep their logging setup."""
    stream = sys.stdout if stream is None else stream
    handler = logging.StreamHandler(stream)
    color = getattr(stream, 'isatty', lambda: False)() and os.environ.get('TERM') != 'dumb' and 'NO_COLOR' not in os.environ
    handler.setFormatter(DeployFormatter(color=color))
    pending = queue.Queue(maxsize=8192)
    queued = RecentQueueHandler(pending)
    queued.setLevel(LOG_LEVELS[level])
    listener = _QueueListener(pending, handler)
    root = logging.getLogger()
    previous_handlers, previous_level = root.handlers[:], root.level
    root.handlers = [queued]
    root.setLevel(LOG_LEVELS[level])
    listener.start()
    try:
        yield
    finally:
        root.handlers = previous_handlers
        root.setLevel(previous_level)
        listener.stop()
        queued.close()
        handler.close()
