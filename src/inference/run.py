"""Run a dedicated inference HTTP server from an explicit configuration."""
import argparse
import logging
import math
import os
from pathlib import Path
import signal

import yaml

from .server import Server, Service
from .model import WAMInference


def load_config(path):
    config = yaml.safe_load(os.path.expandvars(Path(path).read_text()))
    if "${" in str(config):
        raise ValueError("Configuration contains unresolved environment variables")
    return config


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Inference configuration must be a mapping")
    unknown = set(config) - {"host", "port", "model",
                             "max_request_bytes", "request_timeout"}
    if unknown:
        raise ValueError(f"Unknown inference server options: {sorted(unknown)}")
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("model must be a mapping of WAM loading options")
    unknown = set(model) - {"artifact", "checkpoint", "external_models", "device", "attn_implementation"}
    if unknown:
        raise ValueError(f"Unknown inference model options: {sorted(unknown)}")
    if any(value is not None and (not isinstance(value, str) or not value.strip()) for value in model.values()):
        raise ValueError("Inference model options must be nonempty strings")
    if bool(model.get("artifact")) == bool(model.get("checkpoint")):
        raise ValueError("Specify exactly one of model.artifact or model.checkpoint")
    if model.get("checkpoint") and not model.get("external_models"):
        raise ValueError("model.external_models is required for an original checkpoint")
    if model.get("artifact") and any(model.get(key) is not None for key in ("external_models", "attn_implementation")):
        raise ValueError("external_models and attn_implementation only apply to original checkpoints")
    if not 0 <= int(config.get("port", 8014)) <= 65535:
        raise ValueError("Invalid server port")
    timeout = float(config.get("request_timeout", 2))
    if int(config.get("max_request_bytes", 128*1024*1024)) <= 0 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Request size and timeout must be positive")


def create_server(config):
    validate_config(config)
    inference = WAMInference.load(**config["model"])
    return Server(Service(inference), **{key: value for key, value in config.items() if key != "model"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Validate configuration without loading model weights or opening sockets")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.check:
        validate_config(config)
        print("Validated inference server configuration")
        return
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
    logging.getLogger(__name__).info("Loading WAM inference from %s", args.config)
    server = create_server(config)
    logging.getLogger(__name__).info("Model loaded; starting HTTP server at http://%s:%s", config.get("host", "127.0.0.1"), config.get("port", 8014))
    previous = signal.signal(signal.SIGTERM, lambda *_: server.stop_event.set())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
