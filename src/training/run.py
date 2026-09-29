"""Train from an explicit YAML configuration."""
import argparse
import os
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = yaml.safe_load(os.path.expandvars(args.config.read_text()))
    if "${" in str(config):
        parser.error("Configuration contains unresolved environment variables")
    if args.resume:
        config.setdefault("train", {})["resume"] = args.resume
    from .engine import train
    train(config)


if __name__ == "__main__":
    main()
