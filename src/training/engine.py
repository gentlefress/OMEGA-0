"""Small Accelerate engine: distributed execution, evaluation and exact step resume.

Checkpoints belong to this engine and must be trusted. Deterministic mid-epoch
resume requires workers=0; worker process RNG is not claimed to be restorable.
"""
from __future__ import annotations

import json
import math
import random
import shutil
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler
from accelerate import Accelerator
from accelerate.utils import set_seed

from omega.models.config import ModelConfig
from omega.models.external import resolve_factory
from omega.models.artifact import build_model, export_artifact
from .finetune import configure_trainable, optimizer_groups, preserve_frozen_eval, load_predictor_init
from .losses import FinetuneObjective


class SeededDataset:
    """Make augmentation independent of data-loader lookahead and skipped batches."""
    def __init__(self, dataset, seed):
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            seed = (self.seed + self.epoch * len(self) + index) % (2 ** 32)
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            return self.dataset[index]
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)


def component(spec, **kwargs):
    return resolve_factory(spec["factory"])(**spec.get("options", {}), **kwargs)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, objective, loader, accelerator, max_batches=None):
    model.eval()
    sums, count = {}, 0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            with accelerator.autocast():
                metrics = objective(model, batch)
            for name, value in metrics.items():
                if value.ndim == 0:
                    sums[name] = sums.get(name, 0.) + accelerator.reduce(value.detach(), reduction="mean").item()
            count += 1
    if not count:
        raise ValueError("Evaluation dataset is empty")
    return {key: value / count for key, value in sums.items()}


def train(config):
    stage = config.get("stage", "finetune")
    if stage not in {"vlm_pretrain", "finetune"}:
        raise ValueError("Training stage must be vlm_pretrain or finetune")
    if stage == "vlm_pretrain":
        from . import vlm_pretrain
        from omega.data.collate import PaddedCollatorForActionPrediction
        cfg = vlm_pretrain.VLMPretrainConfig.model_validate(config.get("model", {}))
        if any(key in config for key in ("model_factory", "model_factory_options", "predictor_init", "artifact")):
            raise ValueError("Pretraining builds and exports a Qwen backbone; WAM model/artifact options do not apply")
    else:
        cfg = ModelConfig.model_validate(config.get("model", {}))
    options = config.get("train", {})
    seed, workers = int(options.get("seed", 42)), int(options.get("workers", 0))
    accumulation = int(options.get("gradient_accumulation_steps", 1))
    max_steps = int(options.get("max_steps", 1000))
    if min(accumulation, max_steps, int(options.get("batch_size", 1))) < 1:
        raise ValueError("Training sizes must be positive")
    accelerator = Accelerator(cpu=options.get("cpu", False), mixed_precision=options.get("mixed_precision", "no"),
                              gradient_accumulation_steps=accumulation, log_with=options.get("report_to"))
    if options.get("report_to"):
        accelerator.init_trackers(options.get("project", "omega-zero"), config={"seed": seed, "max_steps": max_steps},
                                  init_kwargs=options.get("tracker_options", {}))
    set_seed(seed)
    dataset_kwargs = {}
    if stage == "vlm_pretrain":
        # CPU backward needs FP32 on platforms without BF16 backward kernels.
        # FP16 autocast also needs FP32 parameters for GradScaler to unscale.
        dtype = torch.bfloat16 if accelerator.device.type == "cuda" and accelerator.mixed_precision == "bf16" else torch.float32
        model, processor, action_tokenizer = vlm_pretrain.build_model(cfg, dtype=dtype)
        dataset_kwargs = {"vlm_processor": processor, "action_tokenizer": action_tokenizer}
    else:
        model = build_model(cfg.model_dump(), factory=config.get("model_factory"), factory_options=config.get("model_factory_options"))
        if options.get("configure_trainable", True):
            configure_trainable(model, cfg)
    if config.get("initial_weights"):
        model.load_state_dict(torch.load(config["initial_weights"], map_location="cpu", weights_only=True), strict=True)
    if config.get("predictor_init"):
        accelerator.print(json.dumps(load_predictor_init(model, **config["predictor_init"])))
    legacy_rng = options.get("rng_mode", "indexed") == "legacy"
    dataset = component(config["data"], **dataset_kwargs)
    if not legacy_rng:
        dataset = SeededDataset(dataset, seed)
    collate = component(config["collator"]) if config.get("collator") else None
    if collate is None and stage == "vlm_pretrain":
        collate = PaddedCollatorForActionPrediction(cfg.model_max_length, processor.tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(dataset, generator=generator)
    batch_sampler = component(config["batch_sampler"]) if config.get("batch_sampler") else None
    if batch_sampler is not None and not legacy_rng:
        raise ValueError("Mixture batch samplers require rng_mode: legacy (tuple dataset indices)")
    batching = {"batch_sampler": batch_sampler} if batch_sampler is not None else {
        "batch_size": options.get("batch_size", 1), "sampler": sampler, "drop_last": options.get("drop_last", False)}
    loader = DataLoader(dataset, **batching, collate_fn=collate, num_workers=workers,
                        generator=generator if legacy_rng else torch.Generator().manual_seed(seed),
                        worker_init_fn=worker_init_fn, persistent_workers=bool(workers and options.get("persistent_workers", False)))
    group_builder = vlm_pretrain.optimizer_groups if stage == "vlm_pretrain" else optimizer_groups
    groups = group_builder(model, cfg, options.get("learning_rate", 1e-4))
    if stage == "finetune" and options.get("strict_predictor_only") and cfg.ac_vjepa_freeze_frame_encoder and not cfg.tune_vlm:
        if {g["group_name"] for g in groups} != {"predictor"}:
            raise ValueError("Non-predictor parameters in predictor-only optimizer")
    optimizer = torch.optim.AdamW(groups,
                                  weight_decay=options.get("weight_decay", cfg.weight_decay), betas=tuple(options.get("betas", [0.9, 0.999])), eps=options.get("eps", 1e-8))
    warmup = int(options.get("warmup_steps", 0))
    schedule_steps = int(options.get("schedule_steps", max_steps))
    def schedule(step):
        if step < warmup:
            return step / max(1, warmup)
        if options.get("schedule", "constant") == "constant":
            return 1.
        return 0.5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, schedule_steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    # Step the scheduler once per optimizer update, independent of rank count.
    if batch_sampler is not None and hasattr(batch_sampler, "rank"):
        # The retained mixture samplers already shard indices by rank. Only move
        # their batches to the device; a second shard would skip training data.
        from accelerate.data_loader import prepare_data_loader
        model, optimizer = accelerator.prepare(model, optimizer)
        loader = prepare_data_loader(loader, device=accelerator.device, num_processes=1, process_index=0, put_on_device=True)
    else:
        model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    accelerator.register_for_checkpointing(scheduler)
    if config.get("objective"):
        objective = component(config["objective"])
    elif stage == "vlm_pretrain":
        objective = vlm_pretrain.VLMPretrainObjective(action_tokenizer.action_token_begin_idx, action_tokenizer.n_bins)
    else:
        objective = FinetuneObjective(cfg, accelerator.device, dataset_action_is_delta=config.get("dataset_action_is_delta", False))
    output = Path(options.get("output", "artifacts/train"))
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    accelerator.wait_for_everyone()
    step, epoch, batch_offset = 0, 0, 0
    resume = options.get("resume")
    if resume:
        if workers and options.get("exact_resume", True):
            raise ValueError("Exact resume requires train.workers=0")
        progress = json.loads((Path(resume) / "progress.json").read_text())
        if progress["world_size"] != accelerator.num_processes or progress["batches"] != len(loader) or progress["accumulation"] != accumulation:
            raise ValueError("Resume requires the same rank count, dataset batching and accumulation")
        step, epoch, batch_offset = progress["step"], progress["epoch"], progress["batch_offset"]
        accelerator.load_state(resume)
        # Restore explicitly: some Accelerate versions silently suppress RNG errors.
        rng = torch.load(Path(resume) / f"omega_rng_{accelerator.process_index}.pt", weights_only=False, map_location="cpu")
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        if legacy_rng:
            generator.set_state(rng["epoch_generator"])
    if not len(loader):
        raise ValueError("Training dataset is empty")
    def checkpoint():
        target = output / f"checkpoint-{step:08d}"
        accelerator.save_state(str(target), safe_serialization=False)
        if stage == "vlm_pretrain":
            state_dict = accelerator.get_state_dict(model)
            if accelerator.is_main_process:
                vlm_pretrain.export_backbone(target / "backbone", accelerator.unwrap_model(model), processor, state_dict)
        torch.save({"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    "epoch_generator": epoch_generator_state}, target / f"omega_rng_{accelerator.process_index}.pt")
        if accelerator.is_main_process:
            (target / "progress.json").write_text(json.dumps({"step": step, "epoch": epoch, "batch_offset": batch_offset,
                "world_size": accelerator.num_processes, "batches": len(loader), "accumulation": accumulation}) + "\n")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and options.get("keep_checkpoints"):
            for old in sorted(output.glob("checkpoint-*"))[:-int(options["keep_checkpoints"])]:
                shutil.rmtree(old)
        return target
    while step < max_steps:
        if not legacy_rng:
            dataset.epoch = epoch
            generator.manual_seed(seed + epoch)
        epoch_generator_state = generator.get_state()
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)
        # skip_first_batches skips indices, without replaying stochastic transforms.
        active = accelerator.skip_first_batches(loader, batch_offset) if batch_offset else loader
        model.train()
        if options.get("keep_frozen_eval", True):
            preserve_frozen_eval(accelerator.unwrap_model(model))
        for batch in active:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    losses = objective(model, batch)
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError("Non-finite training loss")
                accelerator.backward(losses["loss"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), options.get("max_grad_norm", 1.))
                optimizer.step()
                optimizer.zero_grad()
            batch_offset += 1
            if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                scheduler.step()
                step += 1
                if step % int(options.get("log_every", 1)) == 0 or step == max_steps:
                    metrics = {name: float(value.detach()) for name, value in losses.items() if value.ndim == 0}
                    metrics.update(step=step, lr=scheduler.get_last_lr()[0])
                    accelerator.print(json.dumps(metrics))
                    accelerator.log(metrics, step=step)
                validation_every = int(options.get("validation_every", 0))
                if validation_every and step % validation_every == 0 and config.get("validation_data"):
                    validation = DataLoader(component(config["validation_data"], **dataset_kwargs), batch_size=options.get("batch_size", 1), collate_fn=collate)
                    metrics = evaluate(model, objective, accelerator.prepare(validation), accelerator, options.get("eval_batches"))
                    if accelerator.is_main_process:
                        (output / f"metrics-{step:08d}.json").write_text(json.dumps(metrics, indent=2) + "\n")
                    model.train()
                    if options.get("keep_frozen_eval", True):
                        preserve_frozen_eval(accelerator.unwrap_model(model))
                if step % int(options.get("save_every", max_steps)) == 0 or step == max_steps:
                    checkpoint()
                if step >= max_steps:
                    break
        if step < max_steps:
            epoch, batch_offset = epoch + 1, 0
    if config.get("validation_data"):
        validation = DataLoader(component(config["validation_data"], **dataset_kwargs), batch_size=options.get("batch_size", 1), collate_fn=collate)
        metrics = evaluate(model, objective, accelerator.prepare(validation), accelerator, options.get("eval_batches"))
        if accelerator.is_main_process:
            (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if config.get("artifact") and accelerator.is_main_process:
        export_artifact(output / f"policy-{step:08d}", accelerator.unwrap_model(model),
                        {**config["artifact"], "model": cfg.model_dump(), **{k: config[k] for k in ("model_factory", "model_factory_options") if k in config}})
    accelerator.wait_for_everyone()
    accelerator.end_training()
    return accelerator.unwrap_model(model), step
