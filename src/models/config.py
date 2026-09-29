# Derived from Omega-0's project models; upstream attribution: LICENSE and NOTICE.md.
from __future__ import annotations
from typing import List
from pydantic import BaseModel, ConfigDict, Field

class ModelConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text_model: str = 'google-t5/t5-base'
    local_files_only: bool = True
    wan_checkpoint: str | None = None
    attn_implementation: str = 'sdpa'
    rtc: bool = False
    max_delay: int = 8
    action_dim: int = 7
    action_chunk_size: int = 6
    dropout: float = 0.1
    loss_w: List[float] = Field(default_factory=lambda : [0.1, 0.2, 0.1])
    weight_decay: float = 0.01
    model_name_or_path: str = 'Qwen/Qwen3-VL-2B-Instruct'
    tune_vlm: bool = False
    gradient_checkpointing: bool = True
    lang_backbone_lr: float = 1e-05
    ac_vjepa_jepa_loss_weight: float = 1.0
    ac_vjepa_action_loss_weight: float = 1.0
    ac_vjepa_action_temporal_loss_weight: float = 0.0
    ac_vjepa_hidden_dim: int = 1024
    ac_vjepa_depth: int = 8
    ac_vjepa_num_heads: int = 8
    ac_vjepa_latent_dim: int = 768
    ac_vjepa_state_dim: int | None = None
    ac_vjepa_use_state: bool = False
    ac_vjepa_force_null_state: bool = False
    ac_vjepa_state_dropout_prob: float = 0.0
    ac_vjepa_image_size: int = 224
    ac_vjepa_patch_size: int = 16
    ac_vjepa_encoder_hidden_dim: int = 768
    ac_vjepa_encoder_depth: int = 4
    ac_vjepa_encoder_heads: int = 8
    ac_vjepa_encoder_mlp_ratio: float | None = None
    ac_vjepa_encoder_use_rope: bool | None = None
    ac_vjepa_encoder_ckpt: str | None = None
    ac_vjepa_encoder_ckpt_key: str | None = 'target_encoder'
    ac_vjepa_predictor_ckpt_key: str | None = 'predictor'
    ac_vjepa_load_predictor_from_ckpt: bool = True
    ac_vjepa_freeze_frame_encoder: bool = True
    ac_vjepa_force_predictor_trainable: bool = True


# These were serialized by the original shared config, but the active WAM
# model, one-video finetune loop and WAM server did not use them.
# In particular, finetune's state-dropout schedule calls were commented out.
# Accept them only when reading an existing checkpoint; new configs stay strict.
_OLD_CHECKPOINT_FIELDS = frozenset({
    'resnet_store_path',
    'pretrained_action_header_path',
    'action_exec_horizon',
    'observation_horizon',
    'img_chunk',
    'n_cams',
    'use_obs',
    'noise_scheduler',
    'train_diffusion_steps',
    'eval_diffusion_steps',
    'share_cam_features',
    'early_fusion',
    'odim',
    'n_conditions',
    'token_fusion',
    'num_blocks',
    'dim_feedforward',
    'nhead',
    'activation',
    'view_feature_dim',
    'use_film',
    'combined_temb',
    'use_dit',
    'vlm_ckpt_step',
    'tune_mm_llm',
    'tune_mm_vision',
    'tune_mm_mlp',
    'mm_projector_lr',
    'vision_tower_lr',
    'data_flatten',
    'data_packing',
    'max_pixels',
    'min_pixels',
    'posttrain_mode',
    'ac_vjepa_abs_action_loss_weight',
    'ac_vjepa_use_action_tokenizer',
    'ac_vjepa_action_tokenizer_type',
    'ac_vjepa_action_tokenizer_ckpt',
    'ac_vjepa_action_tokenizer_bins',
    'ac_vjepa_action_token_loss_weight',
    'ac_vjepa_action_token_use_absolute',
    'ac_vjepa_state_dropout_prob_end',
    'ac_vjepa_state_dropout_anneal_steps',
    'time_dim',
    'hidden_dim',
    'optim',
    'model_max_length',
    'variant',
    'wan_factory',
    'wan_kwargs',
})

def checkpoint_config(values):
    if values.get("posttrain_mode", "ac_vjepa") != "ac_vjepa":
        raise ValueError("Only ac_vjepa WAM checkpoints are supported")
    return {key: value for key, value in values.items() if key not in _OLD_CHECKPOINT_FIELDS}
