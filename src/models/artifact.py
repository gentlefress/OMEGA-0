"""Portable policy metadata and strict weight loading, independent of training."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from omega.models.external import resolve_factory
from omega.models.config import checkpoint_config


def build_model(config, *, factory=None, factory_options=None):
    if factory:
        return resolve_factory(factory)(config=config, **(factory_options or {}))
    from omega.models.config import ModelConfig
    from omega.models.wam import ACVJEPAModel
    from transformers import Qwen3VLForConditionalGeneration
    cfg = ModelConfig.model_validate(config)
    backbone = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg.model_name_or_path, local_files_only=cfg.local_files_only,
        attn_implementation=cfg.attn_implementation, dtype=torch.bfloat16,
    )
    return ACVJEPAModel(model_cfg=cfg, vlm_model=backbone)


def read_artifact(directory):
    root = Path(directory).resolve()
    meta = json.loads((root / "artifact.json").read_text())
    if meta.get("version") != 1:
        raise ValueError("Unsupported artifact version")
    for key in ("model", "preprocessing", "normalization", "weights"):
        if key not in meta:
            raise ValueError(f"Artifact missing {key}")
    path = (root / meta["weights"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Weights must be a file inside the artifact")
    return root, meta


def load_model(directory, device="cpu"):
    root, meta = read_artifact(directory)
    config = checkpoint_config(meta["model"])
    for key in ("model_name_or_path", "text_model", "wan_checkpoint"):
        value = config.get(key)
        if isinstance(value, str) and value.startswith("./"):
            config[key] = str(root / value)
    # Exported weights include initialization; do not reload a training warm start.
    if config.get("ac_vjepa_encoder_use_rope") is None:
        config["ac_vjepa_encoder_use_rope"] = config.get("ac_vjepa_encoder_ckpt") is not None
    config["ac_vjepa_encoder_ckpt"] = None
    model = build_model(config, factory=meta.get("model_factory"), factory_options=meta.get("model_factory_options"))
    path = root / meta["weights"]
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        weights = load_file(str(path))
    else:
        weights = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval(), {**meta, "model": config}


def export_artifact(directory, model, metadata):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    meta = {**metadata, "version": 1, "weights": "model.pt"}
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, root / "model.pt")
    (root / "artifact.json").write_text(json.dumps(meta, indent=2) + "\n")
    read_artifact(root)
