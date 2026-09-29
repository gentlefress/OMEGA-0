"""WAM model for pretraining, finetuning, and action/video inference."""
# Derived from the original Omega predictor; attribution: LICENSE and NOTICE.md.
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from .external import WanVideoEncoder, load_text_encoder
from .predictor import ACVJEPAPredictor
from .vision import FrozenFrameEncoder

logger = logging.getLogger(__name__)


@dataclass
class ACVJEPAOutput:
    pred_next_latent: torch.Tensor
    pred_actions: torch.Tensor
    loss_actions: Optional[torch.Tensor] = None
    recon_video: Optional[torch.Tensor] = None
    target_next_latent: Optional[torch.Tensor] = None
    pred_next_latent_exo: Optional[torch.Tensor] = None
    target_next_latent_exo: Optional[torch.Tensor] = None


class ACVJEPAModel(nn.Module):
    """
    Post-training model:
      - Qwen3VL backbone -> VLM conditioning feature
      - VisionTransformer-style frame encoder (frozen in trainer)
      - AC predictor + action head
    """

    def __init__(self, model_cfg, vlm_model: Qwen3VLForConditionalGeneration):
        super().__init__()
        self.model_cfg = model_cfg
        self.vlm_model = vlm_model
        self.use_state_token = bool(model_cfg.ac_vjepa_use_state)
        self.force_null_state = bool(model_cfg.ac_vjepa_force_null_state)
        self.state_dropout_prob = float(model_cfg.ac_vjepa_state_dropout_prob)
        self.model_cfg = model_cfg
        hidden_size = self.vlm_model.config.text_config.hidden_size
        state_dim = model_cfg.ac_vjepa_state_dim if model_cfg.ac_vjepa_state_dim is not None else model_cfg.action_dim
        self.state_dim = state_dim
        encoder_hidden_dim = int(model_cfg.ac_vjepa_encoder_hidden_dim)
        encoder_mlp_ratio = getattr(model_cfg, 'ac_vjepa_encoder_mlp_ratio', None)
        if encoder_mlp_ratio is None:
            encoder_mlp_ratio = 48.0 / 11.0 if encoder_hidden_dim == 1408 else 4.0
        else:
            target_mlp_hidden = int(round(encoder_hidden_dim * float(encoder_mlp_ratio)))
            encoder_mlp_ratio = target_mlp_hidden / float(encoder_hidden_dim)
        encoder_use_rope = getattr(model_cfg, 'ac_vjepa_encoder_use_rope', None)
        if encoder_use_rope is None:
            encoder_use_rope = bool(model_cfg.ac_vjepa_encoder_ckpt is not None)
        self.frame_encoder = FrozenFrameEncoder(image_size=model_cfg.ac_vjepa_image_size, patch_size=model_cfg.ac_vjepa_patch_size, embed_dim=model_cfg.ac_vjepa_encoder_hidden_dim, latent_dim=model_cfg.ac_vjepa_latent_dim, depth=model_cfg.ac_vjepa_encoder_depth, num_heads=model_cfg.ac_vjepa_encoder_heads, mlp_ratio=float(encoder_mlp_ratio), drop_rate=model_cfg.dropout, use_rope=bool(encoder_use_rope))
        self.wan_encoder = WanVideoEncoder(model_cfg.wan_checkpoint)
        self.patch_size = [1, 2, 2]
        self.vae_stride = [4, 16, 16]
        self.predictor = ACVJEPAPredictor(context_embed_dim=model_cfg.ac_vjepa_encoder_hidden_dim, vlm_embed_dim=hidden_size, state_dim=state_dim, action_dim=model_cfg.action_dim, action_horizon=model_cfg.action_chunk_size, predictor_embed_dim=model_cfg.ac_vjepa_hidden_dim, depth=model_cfg.ac_vjepa_depth, num_heads=model_cfg.ac_vjepa_num_heads, drop_rate=model_cfg.dropout, is_frame_causal=True, use_rope=True, grid_size=model_cfg.ac_vjepa_image_size // model_cfg.ac_vjepa_patch_size, use_state_token=self.use_state_token)
        self.text_encoder = load_text_encoder(model_cfg)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()
        if model_cfg.ac_vjepa_encoder_ckpt is not None:
            self._load_encoder_checkpoint(model_cfg.ac_vjepa_encoder_ckpt)

    def _load_encoder_checkpoint(self, ckpt_path: str) -> None:
        if not os.path.exists(ckpt_path):
            raise ValueError(f'AC-VJEPA encoder checkpoint not found: {ckpt_path}')
        checkpoint = self._read_checkpoint(ckpt_path)
        if not isinstance(checkpoint, dict):
            raise ValueError(f'Unsupported checkpoint format for AC-VJEPA: {type(checkpoint)} at {ckpt_path}')
        if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
            checkpoint = checkpoint['state_dict']
        frame_state = self.frame_encoder.state_dict()
        encoder_nested_key = getattr(self.model_cfg, 'ac_vjepa_encoder_ckpt_key', None)
        encoder_nested_candidates = [encoder_nested_key] if encoder_nested_key else []
        encoder_nested_candidates += ['target_encoder', 'encoder', 'ema_encoder', 'context_encoder', 'frame_encoder']
        encoder_prefix_candidates = ['target_encoder.', 'encoder.', 'ema_encoder.', 'context_encoder.', 'frame_encoder.', 'module.target_encoder.', 'module.encoder.', 'module.ema_encoder.', 'module.context_encoder.', 'module.frame_encoder.']
        (enc_sd, enc_source, enc_variant, enc_matched) = self._select_module_state_dict(checkpoint=checkpoint, module_state=frame_state, nested_keys=encoder_nested_candidates, strip_prefixes=encoder_prefix_candidates)
        if enc_matched <= 0:
            hint = self._build_checkpoint_hint(checkpoint)
            raise ValueError(f'Failed to match frame_encoder weights from {ckpt_path}. Set --model.ac-vjepa-encoder-ckpt-key to the correct nested key.' + hint)
        (missing, unexpected) = self.frame_encoder.load_state_dict(enc_sd, strict=False)
        logger.info('Loaded frame_encoder from %s (%s, %s): matched=%d missing=%d unexpected=%d', ckpt_path, enc_source, enc_variant, enc_matched, len(missing), len(unexpected))
        if len(missing) > 20 and any(('mlp.fc1.weight' in k for k in missing)):
            logger.warning('Many frame_encoder MLP weights were not loaded (missing=%d). Likely architecture mismatch (e.g. ViT-g requires mlp_ratio=48/11). Current config: hidden_dim=%s depth=%s heads=%s rope=%s mlp_ratio=%s', len(missing), self.model_cfg.ac_vjepa_encoder_hidden_dim, self.model_cfg.ac_vjepa_encoder_depth, self.model_cfg.ac_vjepa_encoder_heads, getattr(self.model_cfg, 'ac_vjepa_encoder_use_rope', None), getattr(self.model_cfg, 'ac_vjepa_encoder_mlp_ratio', None))
        if not bool(getattr(self.model_cfg, 'ac_vjepa_load_predictor_from_ckpt', True)):
            logger.info('Skip predictor checkpoint loading by config: ac_vjepa_load_predictor_from_ckpt=False')
            return
        predictor_state = self.predictor.state_dict()
        predictor_nested_key = getattr(self.model_cfg, 'ac_vjepa_predictor_ckpt_key', None)
        predictor_nested_candidates = [predictor_nested_key] if predictor_nested_key else []
        predictor_nested_candidates += ['predictor', 'vit_predictor', 'ac_predictor', 'online_predictor']
        predictor_prefix_candidates = ['predictor.', 'vit_predictor.', 'ac_predictor.', 'online_predictor.', 'module.predictor.']
        (pred_sd, pred_source, pred_variant, pred_matched) = self._select_module_state_dict(checkpoint=checkpoint, module_state=predictor_state, nested_keys=predictor_nested_candidates, strip_prefixes=predictor_prefix_candidates)
        if pred_matched <= 0:
            logger.warning('No compatible predictor weights found in %s; predictor will train from scratch.', ckpt_path)
            return
        (missing, unexpected) = self.predictor.load_state_dict(pred_sd, strict=False)
        logger.info('Loaded predictor from %s (%s, %s): matched=%d missing=%d unexpected=%d', ckpt_path, pred_source, pred_variant, pred_matched, len(missing), len(unexpected))

    @staticmethod
    def _read_checkpoint(ckpt_path: str) -> Any:
        if ckpt_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            return load_file(ckpt_path)
        return torch.load(ckpt_path, map_location='cpu')

    @staticmethod
    def _to_tensor_state_dict(value: Any) -> dict[str, torch.Tensor]:
        if not isinstance(value, dict):
            return {}
        return {str(k): v for (k, v) in value.items() if torch.is_tensor(v)}

    @staticmethod
    def _strip_common_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cleaned: dict[str, torch.Tensor] = {}
        for (key, value) in state_dict.items():
            k = key
            changed = True
            while changed:
                changed = False
                for p in ('module.', 'backbone.'):
                    if k.startswith(p):
                        k = k[len(p):]
                        changed = True
            cleaned[k] = value
        return cleaned

    @staticmethod
    def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for (key, value) in state_dict.items():
            if key.startswith(prefix):
                out[key[len(prefix):]] = value
            else:
                out[key] = value
        return out

    @staticmethod
    def _adapt_checkpoint_tensor(key: str, value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor | None:
        if key == 'patch_embed.proj.weight':
            if value.ndim == 5 and ref.ndim == 4:
                if value.shape[0] == ref.shape[0] and value.shape[1] == ref.shape[1] and (value.shape[-2:] == ref.shape[-2:]):
                    return value.mean(dim=2)
        if tuple(value.shape) == tuple(ref.shape):
            return value
        return None

    @staticmethod
    def _build_checkpoint_hint(checkpoint: dict[str, Any]) -> str:
        try:
            cand = None
            for key in ('target_encoder', 'encoder', 'ema_encoder'):
                v = checkpoint.get(key)
                if isinstance(v, dict):
                    cand = v
                    break
            if cand is None:
                return ''
            inspected = {}
            for (k, v) in cand.items():
                if not torch.is_tensor(v):
                    continue
                nk = str(k)
                for p in ('module.', 'backbone.', 'module.backbone.'):
                    if nk.startswith(p):
                        nk = nk[len(p):]
                inspected[nk] = v
            qkv = inspected.get('blocks.0.attn.qkv.weight')
            norm = inspected.get('norm.weight')
            patch_w = inspected.get('patch_embed.proj.weight')
            if qkv is None or norm is None:
                return ''
            embed_dim = int(norm.shape[0])
            enc_depth = 0
            for k in inspected.keys():
                if k.startswith('blocks.') and '.norm1.weight' in k:
                    try:
                        i = int(k.split('.')[1])
                        enc_depth = max(enc_depth, i + 1)
                    except Exception:
                        pass
            patch_size = None
            if patch_w is not None:
                if patch_w.ndim == 5:
                    patch_size = int(patch_w.shape[-1])
                elif patch_w.ndim == 4:
                    patch_size = int(patch_w.shape[-1])
            likely_heads = 16 if embed_dim % 16 == 0 else 8
            hint = f' Detected checkpoint encoder looks like embed_dim={embed_dim}, depth={enc_depth}' + (f', patch={patch_size}' if patch_size is not None else '') + f'. Try flags: --model.ac-vjepa-encoder-hidden-dim={embed_dim}' + (f' --model.ac-vjepa-encoder-depth={enc_depth}' if enc_depth > 0 else '') + f' --model.ac-vjepa-encoder-heads={likely_heads}' + (f' --model.ac-vjepa-patch-size={patch_size}' if patch_size is not None else '') + ' --model.ac-vjepa-encoder-ckpt-key=target_encoder'
            return hint
        except Exception:
            return ''

    def _select_module_state_dict(self, checkpoint: dict[str, Any], module_state: dict[str, torch.Tensor], nested_keys: list[str], strip_prefixes: list[str]) -> tuple[dict[str, torch.Tensor], str, str, int]:
        module_keys = set(module_state.keys())
        candidates: list[tuple[str, dict[str, torch.Tensor]]] = []
        root_sd = self._to_tensor_state_dict(checkpoint)
        if root_sd:
            candidates.append(('root', root_sd))
        seen = set()
        for key in nested_keys:
            if key is None or key in seen:
                continue
            seen.add(key)
            nested = self._to_tensor_state_dict(checkpoint.get(key))
            if nested:
                candidates.append((f"root['{key}']", nested))
        best_sd: dict[str, torch.Tensor] = {}
        best_source = 'none'
        best_variant = 'none'
        best_score_keys = 0
        best_score_numel = 0
        for (source, cand) in candidates:
            base = self._strip_common_prefixes(cand)
            variants: list[tuple[str, dict[str, torch.Tensor]]] = [('as_is', base)]
            for prefix in strip_prefixes:
                variants.append((f'strip:{prefix}', self._strip_prefix(base, prefix)))
            for (variant_name, variant_sd) in variants:
                filtered: dict[str, torch.Tensor] = {}
                for (k, v) in variant_sd.items():
                    if k not in module_keys:
                        continue
                    adapted = self._adapt_checkpoint_tensor(k, v, module_state[k])
                    if adapted is None:
                        continue
                    filtered[k] = adapted
                score_keys = len(filtered)
                score_numel = int(sum((t.numel() for t in filtered.values())))
                if score_keys > best_score_keys or (score_keys == best_score_keys and score_numel > best_score_numel):
                    best_sd = filtered
                    best_source = source
                    best_variant = variant_name
                    best_score_keys = score_keys
                    best_score_numel = score_numel
        return (best_sd, best_source, best_variant, best_score_keys)

    def _extract_vlm_feature(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        vlm_outputs = self.vlm_model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw, output_hidden_states=True, return_dict=True)
        hidden_states = vlm_outputs.hidden_states[-1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        pooled = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        return pooled

    def _cast_frames_to_frame_encoder_dtype(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Keep frame input dtype/device aligned with frame_encoder weights.
        This avoids bf16/fp32 mismatch under mixed-precision + deepspeed paths.
        """
        ref = self.frame_encoder.patch_embed.proj.weight
        if frames.dtype != ref.dtype or frames.device != ref.device:
            frames = frames.to(device=ref.device, dtype=ref.dtype)
        return frames

    def _prepare_state_feature(self, states: Optional[torch.Tensor], state_valid_mask: Optional[torch.Tensor], batch_size: int, dtype: torch.dtype, device: torch.device) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.use_state_token:
            return (None, None)
        if self.force_null_state:
            state_feature = torch.zeros(batch_size, self.state_dim, dtype=dtype, device=device)
            valid = torch.zeros(batch_size, 1, dtype=dtype, device=device)
            return (state_feature, valid)
        if states is None:
            state_feature = torch.zeros(batch_size, self.state_dim, dtype=dtype, device=device)
            valid = torch.zeros(batch_size, 1, dtype=dtype, device=device)
            return (state_feature, valid)
        if states.ndim == 3:
            state_feature = states[:, -1, :]
        elif states.ndim == 2:
            state_feature = states
        else:
            raise ValueError(f'Invalid states shape: {states.shape}')
        state_feature = state_feature.to(device=device, dtype=dtype)
        if state_feature.shape[-1] < self.state_dim:
            state_feature = F.pad(state_feature, (0, self.state_dim - state_feature.shape[-1]))
        elif state_feature.shape[-1] > self.state_dim:
            state_feature = state_feature[..., :self.state_dim]
        if state_valid_mask is None:
            valid = torch.ones(batch_size, 1, dtype=dtype, device=device)
        else:
            valid = state_valid_mask.to(device=device, dtype=dtype)
            if valid.ndim == 1:
                valid = valid.unsqueeze(-1)
            valid = valid[:, :1]
            valid = valid.clamp_(0.0, 1.0)
        if self.training and self.state_dropout_prob > 0.0:
            drop = (torch.rand(batch_size, 1, device=device) < self.state_dropout_prob).to(dtype)
            valid = valid * (1.0 - drop)
        return (state_feature, valid)

    def _encode_context_from_history(self, context_frames: torch.Tensor, context_valid_mask: Optional[torch.Tensor]=None) -> tuple[torch.Tensor, torch.Tensor]:
        context_frames = self._cast_frames_to_frame_encoder_dtype(context_frames)
        (B, K, C, H, W) = context_frames.shape
        flat_frames = context_frames.reshape(B * K, C, H, W)
        (token_flat, _, _) = self.frame_encoder.encode_tokens(flat_frames)
        (n_ctx, d_ctx) = (token_flat.shape[1], token_flat.shape[2])
        token_seq_raw = token_flat.view(B, K, n_ctx, d_ctx)
        token_seq = token_seq_raw
        if context_valid_mask is not None:
            valid = context_valid_mask.to(device=token_seq_raw.device, dtype=token_seq_raw.dtype)
            if valid.ndim == 1:
                valid = valid.unsqueeze(1)
            elif valid.ndim > 2:
                valid = valid.view(B, -1)
            if valid.shape[1] > K:
                valid = valid[:, -K:]
            elif valid.shape[1] < K:
                valid = F.pad(valid, (K - valid.shape[1], 0), value=0.0)
            valid = valid.clamp(0.0, 1.0)
            token_seq = token_seq_raw * valid[:, :, None, None]
            valid_bool = valid > 0.5
            any_valid = valid_bool.any(dim=1)
            pos = torch.arange(1, K + 1, device=token_seq_raw.device).view(1, K)
            last_valid = (valid_bool.long() * pos).amax(dim=1) - 1
            last_valid = torch.where(any_valid, last_valid, torch.full_like(last_valid, K - 1))
            current_tokens = token_seq_raw[torch.arange(B, device=token_seq_raw.device), last_valid]
        else:
            current_tokens = token_seq_raw[:, 0, :, :]
        current_latent = self.frame_encoder.project_tokens_to_latent(current_tokens)
        return (token_seq, current_latent)

    def best_output_size(self, w, h, dw, dh, expected_area):
        ratio = w / h
        ow = (expected_area * ratio) ** 0.5
        oh = expected_area / ow
        ow1 = int(ow // dw * dw)
        oh1 = int(expected_area / ow1 // dh * dh)
        assert ow1 % dw == 0 and oh1 % dh == 0 and (ow1 * oh1 <= expected_area)
        ratio1 = ow1 / oh1
        oh2 = int(oh // dh * dh)
        ow2 = int(expected_area / oh2 // dw * dw)
        assert oh2 % dh == 0 and ow2 % dw == 0 and (ow2 * oh2 <= expected_area)
        ratio2 = ow2 / oh2
        if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2, ratio2 / ratio):
            return (ow1, oh1)
        else:
            return (ow2, oh2)

    def fast_wan_vae_encode(self, video_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        video_tensor = video_tensor.to(device)
        (B, T, C, H, W) = video_tensor.shape
        if video_tensor.dtype == torch.uint8:
            video_tensor = video_tensor.float() / 255.0
        dh = self.patch_size[1] * self.vae_stride[1]
        dw = self.patch_size[2] * self.vae_stride[2]
        max_area = 256 * 256
        (ow, oh) = self.best_output_size(W, H, dw, dh, max_area)
        if ow != W or oh != H:
            scale = max(ow / W, oh / H)
            (new_w, new_h) = (round(W * scale), round(H * scale))
            video_tensor = video_tensor.view(B * T, C, H, W)
            video_tensor = F.interpolate(video_tensor, size=(new_h, new_w), mode='bicubic', align_corners=False, antialias=True)
            x1 = (new_w - ow) // 2
            y1 = (new_h - oh) // 2
            video_tensor = video_tensor[:, :, y1:y1 + oh, x1:x1 + ow]
            video_tensor = video_tensor.view(B, T, C, oh, ow)
        video_tensor = video_tensor.sub(0.5).div(0.5)
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4).contiguous()
        z = self.wan_encoder.encode(video_tensor)
        return z

    def tensor_to_pil(self, tensor):
        if tensor.dim() == 4:
            img_list = [tes.permute(1, 2, 0).cpu().numpy().astype(np.uint8) for tes in tensor]
            img_list = [img * 255.0 if img.max() <= 1.0 else img for img in img_list]
            img_list = [Image.fromarray(img, mode='RGB') for img in img_list]
            return img_list
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.max() <= 1.0:
            tensor = tensor * 255.0
        np_arr = tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        if np_arr.shape[-1] == 1:
            np_arr = np_arr.squeeze(-1)
            return Image.fromarray(np_arr, mode='L')
        else:
            return Image.fromarray(np_arr, mode='RGB')

    def _encode_inputs(self, input_ids, attention_mask, pixel_values, image_grid_thw,
                       t5_input_ids, t5_attention_mask, current_frames, context_frames,
                       context_valid_mask):
        vlm_feature = self._extract_vlm_feature(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        seq_lens = t5_attention_mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(t5_input_ids, t5_attention_mask).last_hidden_state
        prompt_emb = prompt_emb.clone().to(dtype=torch.bfloat16)
        for (i, v) in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        text_feature = prompt_emb
        mask_expanded = t5_attention_mask.unsqueeze(-1).to(text_feature.dtype)
        sum_features = (text_feature * mask_expanded).sum(dim=1)
        valid_token_counts = mask_expanded.sum(dim=1).clamp_min(1e-09)
        pooled_text_feature = sum_features / valid_token_counts
        text_feature = pooled_text_feature.unsqueeze(1)
        device = vlm_feature.device
        B = vlm_feature.shape[0]
        with torch.no_grad():
            if context_frames is not None:
                (context_tokens_ego, current_latent_ego) = self._encode_context_from_history(context_frames, context_valid_mask)
            elif current_frames.ndim == 5:
                (context_tokens_ego, current_latent_ego) = self._encode_context_from_history(current_frames, context_valid_mask)
            else:
                current_frames = self._cast_frames_to_frame_encoder_dtype(current_frames)
                (ctx_flat, current_latent_ego, _) = self.frame_encoder.encode_tokens(current_frames)
                context_tokens_ego = ctx_flat.unsqueeze(1)
        return vlm_feature, text_feature, context_tokens_ego, current_latent_ego

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, t5_input_ids: torch.Tensor, t5_attention_mask: torch.Tensor, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor, states: Optional[torch.Tensor], current_frames: torch.Tensor, context_frames: Optional[torch.Tensor]=None, next_frames: Optional[torch.Tensor]=None, state_valid_mask: Optional[torch.Tensor]=None, context_valid_mask: Optional[torch.Tensor]=None, actions: Optional[torch.Tensor]=None, timesteps: Optional[torch.Tensor]=None, view_token: Optional[torch.Tensor]=None, sigmas: Optional[torch.Tensor]=None) -> ACVJEPAOutput:
        vlm_feature, text_feature, context_tokens_ego, current_latent_ego = self._encode_inputs(
            input_ids, attention_mask, pixel_values, image_grid_thw, t5_input_ids,
            t5_attention_mask, current_frames, context_frames, context_valid_mask)
        device = vlm_feature.device
        B = vlm_feature.shape[0]
        if next_frames is not None:
            with torch.no_grad():
                target_next_latent_ego = self.fast_wan_vae_encode(next_frames, device)
        else:
            target_next_latent_ego = current_latent_ego
        (B, D_in, T, H, W) = target_next_latent_ego.shape
        target_ego = target_next_latent_ego.clone()
        target_next_latent_ego = target_next_latent_ego.reshape(B, D_in, T, -1).permute(0, 2, 3, 1)
        (pred_next_tokens_ego, pred_actions) = self.predictor(actions=actions, timesteps=timesteps, context_tokens_ego=context_tokens_ego[:, 0:1], context_latents_ego=target_next_latent_ego, states=states, vlm_feature=vlm_feature, text_feature=text_feature, view_token=view_token, sigmas=sigmas)
        (B, N_ego, D_in) = pred_next_tokens_ego.shape
        pred_next_latent_ego = pred_next_tokens_ego.reshape(B, -1, H, W, D_in).permute(0, 4, 1, 2, 3)
        return ACVJEPAOutput(pred_next_latent=pred_next_latent_ego, pred_actions=pred_actions, loss_actions=None, target_next_latent=target_ego)

    def sample(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, t5_input_ids: torch.Tensor, t5_attention_mask: torch.Tensor, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor, states: Optional[torch.Tensor], current_frames: torch.Tensor, context_frames: Optional[torch.Tensor]=None, next_frames: Optional[torch.Tensor]=None, state_valid_mask: Optional[torch.Tensor]=None, context_valid_mask: Optional[torch.Tensor]=None, init_action: Optional[torch.Tensor]=None, view_token: Optional[torch.Tensor]=None) -> ACVJEPAOutput:
        vlm_feature, text_feature, context_tokens_ego, current_latent_ego = self._encode_inputs(
            input_ids, attention_mask, pixel_values, image_grid_thw, t5_input_ids,
            t5_attention_mask, current_frames, context_frames, context_valid_mask)
        device = vlm_feature.device
        B = vlm_feature.shape[0]
        (pred_ego, pred_actions) = self.predictor.sample(context_tokens_ego=context_tokens_ego[:, 0:1], vlm_feature=vlm_feature, text_feature=text_feature, init_action=init_action, states=states, view_token=view_token)
        return ACVJEPAOutput(pred_next_latent=None, pred_actions=pred_actions, loss_actions=None, recon_video=None, target_next_latent=None, pred_next_latent_exo=None, target_next_latent_exo=None)
