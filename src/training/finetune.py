"""Trainability and optimizer groups for the active finetuning objective."""
import torch
from pathlib import Path


def configure_trainable(model, config):
    if hasattr(model, "frame_encoder"):
        model.frame_encoder.requires_grad_(not config.ac_vjepa_freeze_frame_encoder)
    if hasattr(model, "predictor") and config.ac_vjepa_force_predictor_trainable:
        model.predictor.requires_grad_(True)
    if hasattr(model, "vlm_model"):
        model.vlm_model.requires_grad_(config.tune_vlm)
        language = getattr(getattr(model.vlm_model, "model", None), "language_model", None)
        if language is not None and hasattr(language, "norm"):
            language.norm.requires_grad_(False)
    if hasattr(model, "text_encoder"):
        model.text_encoder.requires_grad_(False)
    if config.gradient_checkpointing and config.tune_vlm:
        model.vlm_model.gradient_checkpointing_enable()


def optimizer_groups(model, config, learning_rate):
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad or name.startswith("frame_encoder."):
            continue
        if name.startswith("vlm_model.") and not config.tune_vlm:
            continue
        group = "predictor" if name.startswith("predictor.") else "vlm_model" if name.startswith("vlm_model.") else "other"
        groups.setdefault(group, []).append(param)
    if not groups:
        raise ValueError("Model has no trainable parameters")
    return [{"params": params, "lr": config.lang_backbone_lr if group == "vlm_model" else learning_rate,
             "group_name": group} for group, params in groups.items()]


def load_predictor_init(model, path, key="predictor"):
    """Load the original finetune warm start, including full-model exports."""
    path = Path(path).expanduser()
    if path.is_dir():
        candidates = [path / name for name in ("model.safetensors", "pytorch_model.safetensors", "pytorch_model.bin", "model.bin", "model.pt")]
        path = next((p for p in candidates if p.is_file()), path)
        if path.is_dir():
            files = list(path.glob("*.safetensors"))
            if len(files) != 1:
                raise ValueError(f"No unambiguous predictor checkpoint in {path}")
            path = files[0]
    checkpoint = model._read_checkpoint(str(path))
    checkpoint = checkpoint.get("state_dict", checkpoint)
    state, source, variant, matched = model._select_module_state_dict(
        checkpoint=checkpoint, module_state=model.predictor.state_dict(),
        nested_keys=[key, "predictor", "vit_predictor", "ac_predictor", "online_predictor"],
        strip_prefixes=["predictor.", "vit_predictor.", "ac_predictor.", "online_predictor.", "module.predictor."])
    if matched <= 0:
        raise ValueError(f"No compatible predictor weights in {path}")
    missing, unexpected = model.predictor.load_state_dict(state, strict=False)
    return {"path": str(path), "source": source, "variant": variant, "matched": matched,
            "missing": list(missing), "unexpected": list(unexpected)}


def preserve_frozen_eval(model):
    # Calling train() recursively must not enable target/text dropout.
    for name in ("frame_encoder", "text_encoder", "vlm_model"):
        child = getattr(model, name, None)
        if child is not None and not any(p.requires_grad for p in child.parameters()):
            child.eval()
