"""Qwen action-token pretraining, sharing the optimizer loop with WAM finetuning.

Derived from psi/trainers/pretrain.py and qwen3vl_mixin.py. The original active
objective is causal language-model loss over the assistant's FAST action answer.
"""
from collections import defaultdict
import re

import torch
from pydantic import BaseModel, ConfigDict, Field


class VLMPretrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name_or_path: str
    action_tokenizer: str
    action_dim: int = Field(default=75, gt=0)
    action_chunk_size: int = Field(default=1, gt=0)
    action_bins: int = Field(default=2048, gt=0)
    local_files_only: bool = True
    attn_implementation: str = "flash_attention_2"
    model_max_length: int = Field(default=8192, gt=0)
    tune_mm_llm: bool = True
    tune_mm_vision: bool = False
    tune_mm_mlp: bool = True
    gradient_checkpointing: bool = True
    mm_projector_lr: float | None = Field(default=1e-5, ge=0)
    vision_tower_lr: float | None = Field(default=1e-6, ge=0)
    weight_decay: float = Field(default=0.01, ge=0)


def build_model(config, *, dtype=torch.bfloat16):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from omega.data.action_tokenizer import FastActionTokenizer

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model_name_or_path, local_files_only=config.local_files_only,
        attn_implementation=config.attn_implementation, dtype=dtype)
    if not hasattr(model.config, "hidden_size"):
        # Accelerate/DeepSpeed's automatic bucket sizes use the top-level width.
        model.config.hidden_size = model.config.text_config.hidden_size
    processor = AutoProcessor.from_pretrained(config.model_name_or_path, local_files_only=config.local_files_only)
    tokenizer = processor.tokenizer
    tokenizer.model_max_length = config.model_max_length
    views = {token: token not in tokenizer.get_vocab() for token in ("<EGO_VIEW>", "<EXO_VIEW>")}
    tokenizer.add_special_tokens({"additional_special_tokens": list(views)}, replace_additional_special_tokens=False)
    actions = FastActionTokenizer(tokenizer, config.action_tokenizer,
        config.action_chunk_size, config.action_dim, config.action_bins, local_files_only=config.local_files_only)
    # Resize after adding all tokens, including the action delimiters.
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=192, mean_resizing=True)
    with torch.no_grad():
        embeddings = model.get_input_embeddings().weight
        for token, word in (("<EGO_VIEW>", "first"), ("<EXO_VIEW>", "third")):
            if views[token]:
                source = tokenizer.encode(word, add_special_tokens=False)[0]
                embeddings[tokenizer.convert_tokens_to_ids(token)].copy_(embeddings[source])
    configure_trainable(model, config)
    return model, processor, actions


def configure_trainable(model, config):
    model.requires_grad_(False)
    model.visual.requires_grad_(config.tune_mm_vision)
    model.visual.merger.requires_grad_(config.tune_mm_mlp)
    model.language_model.requires_grad_(config.tune_mm_llm)
    # New action/view embeddings must learn even with a frozen language backbone.
    model.get_input_embeddings().requires_grad_(True)
    model.get_output_embeddings().requires_grad_(True)
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()


def optimizer_groups(model, config, learning_rate):
    groups = defaultdict(list)
    layer_norm_parameters = {id(parameter) for module in model.modules() if isinstance(module, torch.nn.LayerNorm)
                             for parameter in module.parameters()}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        rate = learning_rate
        if config.mm_projector_lr:
            if "merger" in name:
                rate = config.mm_projector_lr
            elif "visual" in name and config.vision_tower_lr:
                rate = config.vision_tower_lr
        no_decay = id(parameter) in layer_norm_parameters or re.search(
            r"bias|layernorm|rmsnorm|(?:^|\.)norm(?:$|\.)|_norm(?:$|\.)", name.lower())
        groups[(rate, 0.0 if no_decay else config.weight_decay)].append(parameter)
    if not groups:
        raise ValueError("Pretraining model has no trainable parameters")
    return [{"params": parameters, "lr": rate, "weight_decay": decay}
            for (rate, decay), parameters in groups.items()]


class VLMPretrainObjective:
    def __init__(self, action_token_begin_idx, action_bins):
        self.begin, self.bins = action_token_begin_idx, action_bins

    def __call__(self, model, batch):
        output = model(**{key: batch[key] for key in (
            "input_ids", "attention_mask", "pixel_values", "labels", "image_grid_thw")})
        labels = batch["labels"][:, 1:]
        with torch.no_grad():
            predictions = output.logits[:, :-1].argmax(dim=-1)
            mask = (labels >= self.begin) & (labels < self.begin + self.bins)
            accuracy = ((predictions == labels) & mask).sum().float() / mask.sum().clamp_min(1)
        return {"loss": output.loss, "action_accuracy": accuracy}


def export_backbone(directory, model, processor, state_dict):
    """A normal Hugging Face model/processor directory usable by WAM finetuning."""
    model.save_pretrained(directory, state_dict=state_dict, safe_serialization=True)
    processor.save_pretrained(directory)
