#!/usr/bin/env python3
"""Fine-tune ω-0 (HF ``keycharon/Omega-zero``) on HA-Lab data with OMEGA-0's own training loop.

Recipe = the released checkpoint's ``run_config.json`` (its WAM fine-tuning run), with only
the action space and chunk changed:

* model: predictor trained, Qwen3-VL / V-JEPA ViT-g / T5 frozen; loss = L1 on actions (suffix)
  + MSE on predicted Wan2.2 future latents; training-time RTC (``rtc``, ``max_delay`` 8:
  per-sample clean prefix of 0..7 rows, loss on the rest), as in the checkpoint;
* data (``omega_halab_data.py``): actions = 64 SONIC token + 12 Inspire openness (``--action-dim
  76``) [+ 2 camera (78)], std-normalized; state = 45-D ``sonic_state``, bounds_q99-normalized
  (stats from our train split); ego image + instruction; chunk ``--chunk`` rows (50);
* optimizer: AdamW lr 1e-4, betas (0.95, 0.999), wd 1e-6, warmup 1000, cosine, grad clip 1,
  bf16, global batch 256 (per-GPU ``--batch-size`` x GPUs x ``--accumulation``).

Weight surgery from the checkpoint (everything else loads unchanged, strict):

* action head ``input_proj`` / ``final_layer.linear``: 64 token columns/rows copied; each hand's
  6 finger rows = minus that side's trigger row (openness = 1 - trigger), input columns = minus
  trigger column / 6; camera rows/columns zero;
* ``motion_queries`` (25 rows) and ``ego_queries`` (7 Wan latent frames x 256) linearly
  interpolated to ``--chunk`` rows and ``1 + (chunk - 1) // 4`` frames (the Wan VAE encodes the
  ``chunk`` future frames to that many latents); ``temporal_pos_emb`` (unused) keeps its init.

Not in the released loop, added here: DDP ``find_unused_parameters`` (``view_3`` and
``temporal_pos_emb`` get no gradient; the original ran DeepSpeed ZeRO-3), resume from the
newest complete ``checkpoint-*`` (chain legs), ``artifact_template.json`` for mid-training
``keep-<step>`` copies (``load_policy``), timestamps on log lines, and the per-step
``RTC Enabled`` print silenced.

    torchrun --nproc_per_node 4 omega_halab_train.py --checkpoint <Omega-zero> --external-models <ext> \\
        --data-root <omega_halab_data out> --output <run dir> --action-dim 76
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

TRIGGER_L, TRIGGER_R, TOKEN = 64, 65, 64
HAND_L, HAND_R, CAMERA = slice(64, 70), slice(70, 76), slice(76, 78)


def _interp_rows(x, rows: int):
    """(1, R, D) -> (1, rows, D), linear like ``SimpleDiTAdaLN``'s ``dec_pos_a`` resize."""
    import torch.nn.functional as F

    return F.interpolate(x.transpose(1, 2), size=rows, mode="linear", align_corners=False).transpose(1, 2)


def adapt(key: str, src, ref):
    """Checkpoint tensor ``src`` reshaped to the model's ``ref``, or None (keep the init)."""
    import torch

    if tuple(src.shape) == tuple(ref.shape):
        return src
    src = src.to(ref.dtype)
    if key == "predictor.motion_queries":
        return _interp_rows(src, ref.shape[1])
    if key == "predictor.ego_queries":
        frames, new_frames = src.shape[1] // 256, ref.shape[1] // 256
        per_frame = src.reshape(frames, 256 * src.shape[2]).T.unsqueeze(0)  # (1, 256*D, frames)
        out = torch.nn.functional.interpolate(per_frame, size=new_frames, mode="linear", align_corners=False)
        return out[0].T.reshape(1, new_frames * 256, src.shape[2])
    if key == "predictor.action_head.net.input_proj.weight":  # (width, action_dim)
        out = torch.zeros_like(ref)
        out[:, :TOKEN] = src[:, :TOKEN]
        out[:, HAND_L] = -src[:, TRIGGER_L : TRIGGER_L + 1] / 6
        out[:, HAND_R] = -src[:, TRIGGER_R : TRIGGER_R + 1] / 6
        return out
    if key in ("predictor.action_head.net.final_layer.linear.weight", "predictor.action_head.net.final_layer.linear.bias"):
        out = torch.zeros_like(ref)  # (action_dim, width) / (action_dim,)
        out[:TOKEN] = src[:TOKEN]
        out[HAND_L] = -src[TRIGGER_L]
        out[HAND_R] = -src[TRIGGER_R]
        return out
    return None


def build_model(config, *, checkpoint: str):
    """``model_factory``: the checkpoint architecture for ``config`` with surgered checkpoint weights."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    from omega.models.config import ModelConfig
    from omega.models.wam import ACVJEPAModel

    cfg = ModelConfig.model_validate(config)
    backbone = Qwen3VLForConditionalGeneration._from_config(
        AutoConfig.from_pretrained(checkpoint, local_files_only=True),
        attn_implementation=cfg.attn_implementation, dtype=torch.bfloat16)
    model = ACVJEPAModel(model_cfg=cfg, vlm_model=backbone)
    # The predictor hard-codes 1792 video queries = 7 Wan latent frames (25 future frames) x 16x16;
    # the JEPA target of ``chunk`` future frames has 1 + (chunk - 1) // 4 latent frames.
    queries = model.predictor.ego_queries
    latent_frames = 1 + (cfg.action_chunk_size - 1) // 4
    if queries.shape[1] != latent_frames * 256:
        model.predictor.ego_queries = torch.nn.Parameter(queries.new_zeros(1, latent_frames * 256, queries.shape[2]))
    weights = load_file(str(Path(checkpoint) / "model.safetensors"))
    # Safetensors drops tied duplicate names (as in inference/checkpoint.py::_load_original).
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    for names in aliases.values():
        present = next((n for n in names if n in weights), None)
        for n in names:
            if present and n not in weights:
                weights[n] = weights[present]
    state, changed, kept = {}, [], []
    for key, ref in model.state_dict().items():
        value = adapt(key, weights[key], ref) if key in weights else None
        if value is None:
            kept.append(key)
            value = ref
        elif tuple(weights[key].shape) != tuple(ref.shape):
            changed.append(f"{key} {tuple(weights[key].shape)}->{tuple(ref.shape)}")
        state[key] = value
    unused = sorted(set(weights) - set(state))
    model.load_state_dict(state, strict=True)
    print(json.dumps({"surgery": changed, "kept_init": kept, "unused_checkpoint_keys": unused[:20]}), flush=True)
    return model


def load_policy(directory: str, device: str = "cuda"):
    """``keep-<step>`` / ``checkpoint-*`` -> ``(model, metadata)`` for ``omega.inference.model.WAMInference``."""
    import torch

    root = Path(directory)
    meta = json.loads((root / "artifact.json").read_text()) if (root / "artifact.json").exists() else \
        json.loads((root.parent / "artifact_template.json").read_text())
    model = build_model(meta["model"], **meta["model_factory_options"])
    weights = torch.load(root / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    weights = {k.removeprefix("module."): v for k, v in weights.items()}
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval(), meta


def latest_checkpoint(output: Path) -> Path | None:
    done = [p for p in output.glob("checkpoint-*") if p.name[len("checkpoint-"):].isdigit() and (p / "progress.json").exists()]
    return max(done, key=lambda p: int(p.name[len("checkpoint-"):])) if done else None


def make_config(args) -> dict:
    from omega.models.config import ModelConfig, checkpoint_config

    ckpt, ext = Path(args.checkpoint), Path(args.external_models)
    run = json.loads((ckpt / "run_config.json").read_text())
    model = checkpoint_config(run["model"])
    model.update(model_name_or_path=str(ckpt), text_model=str(ext / "t5-base"), local_files_only=True,
                 wan_checkpoint=str(ext / "Wan2.2_VAE.pth"), ac_vjepa_encoder_ckpt=None, attn_implementation=args.attn,
                 action_dim=args.action_dim, action_chunk_size=args.chunk, rtc=True, max_delay=args.max_delay)
    model = ModelConfig.model_validate(model).model_dump()
    stats_path = str(Path(args.data_root) / "stats.json")
    stats = json.loads(Path(stats_path).read_text())
    action_key = f"halab{args.action_dim}"
    if action_key not in stats:
        raise SystemExit(f"{stats_path} has no {action_key}")

    def data(split):
        return {"factory": "omega.data.dataset:dataset", "options": {
            "reader": {"factory": "omega_halab_data:HALabOmegaDataset", "options": {
                "data_root": args.data_root, "split": split, "action_chunk_size": args.chunk,
                "action_dim": args.action_dim, "text_model": model["text_model"]}},
            "processor": {"pretrained_model_name_or_path": str(ckpt), "local_files_only": True},
            "transforms": [
                {"factory": "omega.data.transforms:WholeBodyRepackTransform",
                 "options": {"action_chunk_size": args.chunk, "pad_action_dim": args.action_dim, "pad_state_dim": 45}},
                {"factory": "omega.data.normalization:ActionStateTransform",
                 "options": {"stat_path": stats_path, "state_stat_path": stats_path, "stat_action_key": action_key,
                             "stat_state_key": "halab_state", "action_norm_type": "std", "normalize_state": True,
                             "state_norm_type": "bounds_q99", "clip_value": None, "eps": 1e-6}},
                {"factory": "omega.data.transforms:WAMModelTransform",
                 "options": {"include_state": True, "img_aug": False,
                             "resize": {"size": [270, 480]}, "center_crop": {"size": [270, 480]},
                             "jepa_resize": {"size": [256, 256]}, "jepa_center_crop": {"size": [256, 256]}}},
            ]}}

    output = Path(args.output)
    train = {
        "batch_size": args.batch_size, "gradient_accumulation_steps": args.accumulation, "max_steps": args.max_steps,
        "workers": args.workers, "mixed_precision": "bf16", "learning_rate": 1e-4, "betas": [0.95, 0.999],
        "weight_decay": 1e-6, "eps": 1e-8, "max_grad_norm": 1.0, "warmup_steps": 1000, "schedule": "cosine",
        "save_every": args.save_every, "log_every": args.log_every, "keep_checkpoints": 2,
        "validation_every": args.validation_every, "eval_batches": 20, "output": str(output), "seed": 1039,
        "rng_mode": "legacy", "exact_resume": False, "drop_last": True, "persistent_workers": True,
        "keep_frozen_eval": False, "strict_predictor_only": True,
        "report_to": "wandb", "project": args.wandb_project,
        "tracker_options": {"wandb": {"name": output.name, "id": output.name, "resume": "allow"}},
    }
    resume = latest_checkpoint(output)
    if resume is not None:
        train["resume"] = str(resume)
    state_stats = stats["halab_state"]
    return {
        "model": model,
        "model_factory": "omega_halab_train:build_model",
        "model_factory_options": {"checkpoint": str(ckpt)},
        "data": data("train"), "validation_data": data("val"),
        "collator": {"factory": "omega.data.collate:PaddedCollatorForTogether",
                     "options": {"model_max_length": 4096, "pad_token_id": 151643, "t5_pad_token_id": 0}},
        "train": train,
        # Same preprocessing / normalization metadata as inference/checkpoint.py::_load_original.
        "artifact": {"preprocessing": {"history_size": 1, "image_field": "image", "vlm_size": [270, 480],
                                       "precision": "bfloat16", "seed": 1039, "require_state": True},
                     "normalization": {"action": {"mode": "std", "eps": 1e-6, **stats[action_key]},
                                       "state": {"mode": "bounds_q99", "eps": 1e-6, **state_stats}}},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="HF keycharon/Omega-zero snapshot dir")
    parser.add_argument("--external-models", required=True, help="dir with t5-base/ and Wan2.2_VAE.pth")
    parser.add_argument("--data-root", required=True, help="omega_halab_data.py output")
    parser.add_argument("--output", required=True)
    parser.add_argument("--action-dim", type=int, choices=(76, 78), default=76)
    parser.add_argument("--chunk", type=int, default=50)
    parser.add_argument("--max-delay", type=int, default=8, help="RTC: prefix rows ~ U{0..max_delay-1} (checkpoint: 8)")
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=8, help="per GPU, per micro-step")
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--save-every", type=int, default=100, help="checkpoint period (released run: 100)")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--validation-every", type=int, default=1000)
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--wandb-project", default="halab-omega0")
    args = parser.parse_args()

    import os

    from accelerate import Accelerator, DistributedDataParallelKwargs
    import omega.models.diffusion.diffusion.gaussian_diffusion as gaussian_diffusion
    import omega.training.engine as engine

    class HALabAccelerator(Accelerator):
        def __init__(self, *a, **k):
            k.setdefault("kwargs_handlers", [DistributedDataParallelKwargs(find_unused_parameters=True)])
            super().__init__(*a, **k)

        def print(self, *a, **k):
            super().print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, **k)

    engine.Accelerator = HALabAccelerator
    gaussian_diffusion.print = lambda *a, **k: None  # training_losses prints "RTC Enabled" every micro-step

    config = make_config(args)
    output = Path(args.output)
    if int(os.environ.get("RANK", "0")) == 0:
        output.mkdir(parents=True, exist_ok=True)
        template = {**config["artifact"], "version": 1, "weights": "pytorch_model.bin", "model": config["model"],
                    "model_factory": config["model_factory"], "model_factory_options": config["model_factory_options"]}
        (output / "artifact_template.json").write_text(json.dumps(template, indent=2) + "\n")
        world = int(os.environ.get("WORLD_SIZE", "1"))
        print(f"global batch {args.batch_size} x {world} GPUs x {args.accumulation} = {args.batch_size * world * args.accumulation}; "
              f"resume: {config['train'].get('resume')}", flush=True)
    engine.train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
