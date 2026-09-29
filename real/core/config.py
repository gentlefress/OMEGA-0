"""Configuration with direct Python class paths and lazy imports."""
from __future__ import annotations

import argparse
import importlib
import logging
import math
import os
from pathlib import Path

import yaml

from .runtime import Module, Runtime
from .logging import LOG_LEVELS, ModuleLogger, console_logging

def resolve(name):
    """Import a configured dotted class path (or an explicit callable path)."""
    if ":" in name:
        module, attribute = name.split(":", 1)
    else:
        module, _, attribute = name.rpartition(".")
    if not module or not attribute:
        raise ValueError(f"Expected a fully qualified Python path, got {name!r}")
    value = importlib.import_module(module)
    for part in attribute.split("."):
        value = getattr(value, part)
    return value


def load_config(path):
    """Paths in options are explicit or environment-expanded; never evaluated as code."""
    value = yaml.safe_load(Path(path).read_text())
    def expand(item):
        if isinstance(item, dict):
            return {k: expand(v) for k, v in item.items()}
        if isinstance(item, list):
            return [expand(v) for v in item]
        if isinstance(item, str):
            result = os.path.expandvars(os.path.expanduser(item))
            if "${" in result:
                raise ValueError(f"Unresolved environment variable: {item}")
            return result
        return item
    return expand(value)


def validate_config(config):
    """Check the small workflow vocabulary before importing or creating modules."""
    def mapping(value, allowed, label, required=()):
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be a mapping")
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{label} keys must be strings")
        unknown, missing = set(value) - allowed, set(required) - value.keys()
        if unknown or missing:
            raise ValueError(f"Invalid {label}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")

    mapping(config, {"modules", "events", "execution_groups", "duration"}, "workflow", {"modules"})
    duration = config.get("duration")
    if duration is not None and (isinstance(duration, bool) or not isinstance(duration, (int, float))
                                 or not math.isfinite(duration) or duration < 0):
        raise ValueError("duration must be finite and nonnegative")
    for key in ("modules", "events", "execution_groups"):
        if not isinstance(config.get(key, []), list):
            raise ValueError(f"{key} must be a list")
    for entry in config["modules"]:
        mapping(entry, {"id", "type", "inputs", "outputs", "rate_hz", "options", "queue_size", "specs", "events"},
                "module", {"id", "type"})
        if not all(isinstance(entry[key], str) and entry[key] for key in ("id", "type")):
            raise ValueError("Module id and type must be nonempty strings")
        for key in ("inputs", "outputs", "options", "specs"):
            if not isinstance(entry.get(key, {}), dict):
                raise ValueError(f"{entry['id']}.{key} must be a mapping")
        for key in ("inputs", "outputs"):
            if any(not isinstance(v, str) or not v for pair in entry.get(key, {}).items() for v in pair):
                raise ValueError(f"{entry['id']}.{key} must map nonempty port names to stream names")
        if not isinstance(entry.get("events", []), list):
            raise ValueError(f"{entry['id']}.events must be a list")
    for entry in config.get("execution_groups", []):
        mapping(entry, {"id", "modules", "rate_hz", "trigger"}, "execution group", {"id", "modules", "rate_hz"})


def build_runtime(config, *, allow_hardware=False, clock=None):
    validate_config(config)
    runtime = Runtime(allow_hardware=allow_hardware, configuration=config, **({"clock": clock} if clock else {}))
    for entry in config["modules"]:
        factory = resolve(entry["type"])
        module = factory()
        if not isinstance(module, Module):
            raise TypeError(f"{entry['type']} does not implement Module")
        options = dict(entry.get("options", {}))
        runtime.add(entry["id"], module, inputs=entry.get("inputs"), outputs=entry.get("outputs"),
                    rate_hz=entry.get("rate_hz", 30), options=options, queue_size=entry.get("queue_size", 256), specs=entry.get("specs"), events=entry.get("events", []))
    runtime.runtime_events = config.get("events", [])
    for entry in config.get("execution_groups", []):
        runtime.add_execution_group(entry["id"], entry["modules"], rate_hz=entry["rate_hz"],
                                    trigger=entry.get("trigger"))
    return runtime


def run_cli(description, argv=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--status-interval", type=float, default=5, help="Seconds between per-module progress messages; 0 disables periodic messages")
    parser.add_argument("--log-level", choices=tuple(LOG_LEVELS), default="info", help="Console severity, matching sonic_deploy")
    parser.add_argument("--arm", action="store_true", help="Allow host command transport to send to configured hardware endpoints")
    parser.add_argument("--check", action="store_true", help="Validate config/bindings without starting hardware or sockets")
    args = parser.parse_args(argv)
    with console_logging(args.log_level):
        config = load_config(args.config)
        ModuleLogger(logging.getLogger(__name__), {'module_id': 'config'}).info("Loaded configuration: %s", Path(args.config).resolve())
        runtime = build_runtime(config, allow_hardware=args.arm)
        if args.check:
            runtime._bind()
            print(f"Validated {len(runtime.runners)} modules, {len(runtime.execution_groups)} execution groups")
            return
        runtime.run(args.duration if args.duration is not None else config.get("duration"), status_interval=args.status_interval)
