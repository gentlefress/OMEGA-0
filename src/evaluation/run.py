"""Evaluate an exported model on a configured dataset and objective."""
import argparse
import json
import os
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from accelerate import Accelerator
    from torch.utils.data import DataLoader
    from omega.models.artifact import load_model
    from omega.models.config import ModelConfig
    from omega.training.engine import component, evaluate
    from omega.training.losses import FinetuneObjective
    config = yaml.safe_load(os.path.expandvars(args.config.read_text()))
    accelerator = Accelerator(mixed_precision=config.get("mixed_precision", "no"))
    model, meta = load_model(args.artifact, accelerator.device)
    objective = component(config["objective"]) if "objective" in config else FinetuneObjective(ModelConfig.model_validate(meta["model"]), accelerator.device)
    collator = component(config["collator"]) if "collator" in config else None
    loader = DataLoader(component(config["data"]), batch_size=config.get("batch_size", 1), collate_fn=collator)
    model, loader = accelerator.prepare(model, loader)
    result = evaluate(model, objective, loader, accelerator, config.get("max_batches"))
    if accelerator.is_main_process:
        body = json.dumps(result, indent=2) + "\n"
        if args.output:
            args.output.write_text(body)
        print(body)


if __name__ == "__main__":
    main()
