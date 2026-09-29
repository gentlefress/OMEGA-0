"""Action/video predictor with view conditioning and the DiT action head."""
# Derived from the original Omega predictor; attribution: LICENSE and NOTICE.md.
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .attention import ACRoPECrossAttention, ACRoPESelfAttention
from .common import DropPath, MLP, SwiGLUFFN, trunc_normal_
from .dit import dit_ddim_xl


class ACBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, wide_silu=True, norm_layer=nn.LayerNorm, use_sdpa=True, is_causal=False, grid_size=16, use_rope=False, **kwargs):
        super().__init__()
        self.norm_prefix = norm_layer(dim)
        self.norm_motion = norm_layer(dim)
        self.norm_video = norm_layer(dim)
        if use_rope:
            self.attn_mot = ACRoPESelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, use_sdpa=use_sdpa, is_causal=is_causal, grid_size=grid_size, proj_drop=drop)
            self.attn_prefix = ACRoPESelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, use_sdpa=use_sdpa, is_causal=is_causal, grid_size=grid_size, proj_drop=drop)
            self.attn_video = ACRoPESelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, use_sdpa=True, is_causal=True, grid_size=grid_size, proj_drop=drop)
            self.cross_attn_mp = ACRoPECrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, use_sdpa=use_sdpa, grid_size=grid_size, proj_drop=drop)
            self.cross_attn_vp = ACRoPECrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, use_sdpa=use_sdpa, grid_size=grid_size, proj_drop=drop)
            self.cross_attn_mv = ACRoPECrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, use_sdpa=use_sdpa, grid_size=grid_size, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2_motion = norm_layer(dim)
        self.norm2_prefix = norm_layer(dim)
        self.norm2_video = norm_layer(dim)
        self.norm_mv = norm_layer(dim)
        self.norm_mp = norm_layer(dim)
        self.norm_vp = norm_layer(dim)
        self.norm_mv_2 = norm_layer(dim)
        self.norm_mp_2 = norm_layer(dim)
        self.norm_vp_2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp_mot = SwiGLUFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop)
            self.mlp_prefix = SwiGLUFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop)
            self.mlp_video = SwiGLUFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop)
            self.mlp_mp = SwiGLUFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop)
            self.mlp_mv = SwiGLUFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop)
            self.mlp_vp = SwiGLUFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop)
        else:
            self.mlp_mot = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.mlp_prefix = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.mlp_video = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.mlp_mp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.mlp_mv = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.mlp_vp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, prefix, motion, video, mask=None, attn_mask=None, T=None, H=None, W=None, action_tokens=0):
        image_tokens = prefix.size(1) - 3
        H = W = math.isqrt(image_tokens)
        if H * W != image_tokens:
            raise ValueError('Active predictor requires a square image-token grid')
        prefix = self.norm_prefix(prefix)
        motion = self.norm_motion(motion)
        video = self.norm_video(video)
        mot = self.attn_mot(motion, mask=mask, attn_mask=attn_mask, T=T, H=H, W=W, num_mot=motion.size(1), action_tokens=action_tokens)
        prx = self.attn_prefix(prefix, mask=mask, attn_mask=attn_mask, T=T, H=H, W=W, num_vlm=3, num_img=image_tokens, action_tokens=action_tokens)
        vid = self.attn_video(video, mask=mask, attn_mask=attn_mask, T=T, H=H, W=W, num_vid_ego=video.size(1), action_tokens=action_tokens)
        prefix = prefix + self.drop_path(prx)
        prx = self.norm2_prefix(prefix)
        prefix = prefix + self.drop_path(self.mlp_prefix(prx))
        motion = motion + self.drop_path(mot)
        mot = self.norm2_motion(motion)
        motion = motion + self.drop_path(self.mlp_mot(mot))
        video = video + self.drop_path(vid)
        vid = self.norm2_motion(video)
        video = video + self.drop_path(self.mlp_video(vid))
        motion = motion + self.cross_attn_mp(self.norm_mp(motion), prefix, T=T, H=H, W=W, num_vlm=3, num_img=image_tokens, num_mot=motion.size(1))
        motion = motion + self.mlp_mp(self.norm_mp_2(motion))
        video = video + self.cross_attn_vp(self.norm_vp(video), prefix, T=T, H=H, W=W, num_vlm=3, num_img=image_tokens, num_mot=None, num_vid_ego=video.size(1))
        video = video + self.mlp_vp(self.norm_vp_2(video))
        motion = motion + self.cross_attn_mv(self.norm_mv(motion), video, num_vlm=None, num_img=None, num_mot=motion.size(1), num_vid_ego=video.size(1))
        motion = motion + self.mlp_mv(self.norm_mv_2(motion))
        return (prefix, motion, video)


class ACVJEPAPredictor(nn.Module):

    def __init__(self, context_embed_dim: int, vlm_embed_dim: int, state_dim: int, action_dim: int, action_horizon: int, predictor_embed_dim: int=1024, depth: int=24, num_heads: int=16, mlp_ratio: float=4.0, qkv_bias: bool=True, qk_scale=None, drop_rate: float=0.0, attn_drop_rate: float=0.0, drop_path_rate: float=0.0, norm_layer=nn.LayerNorm, init_std: float=0.02, use_silu: bool=False, wide_silu: bool=True, is_frame_causal: bool=True, use_rope: bool=True, use_sdpa: bool=True, grid_size: int=14, use_state_token: bool=False):
        super().__init__()
        self.action_horizon = action_horizon
        self.state_dim = state_dim
        self.is_frame_causal = is_frame_causal
        self.use_state_token = use_state_token
        self.vlm_encoder = nn.Linear(vlm_embed_dim, predictor_embed_dim, bias=True)
        self.view_proj = nn.Linear(predictor_embed_dim, predictor_embed_dim, bias=True)
        self.predictor_embed_ego = nn.Linear(context_embed_dim, predictor_embed_dim, bias=True)
        self.motion_queries = nn.Parameter(torch.zeros(1, action_horizon, predictor_embed_dim))
        self.ego_queries = nn.Parameter(torch.zeros(1, 1792, predictor_embed_dim))
        self.temporal_pos_emb = nn.Parameter(torch.zeros(1, action_horizon + 8 * 256 + 2, predictor_embed_dim))
        self.view_1 = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.view_3 = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        trunc_normal_(self.motion_queries, std=0.02)
        trunc_normal_(self.ego_queries, std=0.02)
        trunc_normal_(self.view_1, std=0.02)
        trunc_normal_(self.view_3, std=0.02)
        trunc_normal_(self.temporal_pos_emb, std=0.02)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_blocks = nn.ModuleList([ACBlock(use_rope=True, grid_size=grid_size, dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, act_layer=nn.SiLU if use_silu else nn.GELU, wide_silu=wide_silu, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(6)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, 48, bias=True)
        self.text_proj = nn.Linear(768, predictor_embed_dim)
        self.state_encoder = nn.Linear(45, predictor_embed_dim)
        self.action_head = dit_ddim_xl(width=predictor_embed_dim, target_channels=action_dim, z_channels=predictor_embed_dim)
        self._mask_cache: dict[tuple[int, int], torch.Tensor] = {}
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):

        def rescale(param, layer_id):
            if param is not None:
                param.div_(math.sqrt(2.0 * layer_id))
        for (layer_id, layer) in enumerate(self.predictor_blocks):
            current_layer = layer_id + 1
            rescale(layer.attn_prefix.proj.weight.data, current_layer)
            rescale(layer.attn_mot.proj.weight.data, current_layer)
            rescale(layer.attn_video.proj.weight.data, current_layer)
            rescale(layer.cross_attn_mp.proj.weight.data, current_layer)
            rescale(layer.cross_attn_vp.proj.weight.data, current_layer)
            rescale(layer.cross_attn_mv.proj.weight.data, current_layer)
            rescale(layer.mlp_prefix.fc2.weight.data, current_layer)
            rescale(layer.mlp_mot.fc2.weight.data, current_layer)
            rescale(layer.mlp_video.fc2.weight.data, current_layer)
            rescale(layer.mlp_mp.fc2.weight.data, current_layer)
            rescale(layer.mlp_vp.fc2.weight.data, current_layer)
            rescale(layer.mlp_mv.fc2.weight.data, current_layer)

    def _build_prefix_query_causal_mask(self, prefix_len: int, query_len: int, device: torch.device) -> torch.Tensor:
        key = (prefix_len, query_len)
        if key not in self._mask_cache:
            total = prefix_len + query_len
            mask = torch.zeros(total, total, dtype=torch.bool)
            mask[:prefix_len, :prefix_len] = True
            for i in range(query_len):
                row = prefix_len + i
                mask[row, :prefix_len] = True
                mask[row, prefix_len:prefix_len + i + 1] = True
            self._mask_cache[key] = mask
        return self._mask_cache[key].to(device=device, non_blocking=True)

    def _get_dynamic_attention_mask_no_exo(self, batch_size: int, prefix_len: int, m_len: int, ego_len: int, device: torch.device) -> torch.Tensor:
        B = batch_size
        total = prefix_len + m_len + ego_len
        key = (prefix_len, m_len, ego_len)
        if key not in self._mask_cache:
            base_mask = torch.zeros(total, total, dtype=torch.bool, device=device)
            base_mask[:prefix_len, :prefix_len] = True
            base_mask[prefix_len:, :prefix_len] = True
            base_mask[prefix_len:, prefix_len:] = True
            self._mask_cache[key] = base_mask
        batch_mask = self._mask_cache[key].unsqueeze(0).expand(B, -1, -1).clone()
        batch_mask = batch_mask.unsqueeze(1)
        return batch_mask

    def _get_dynamic_attention_mask(self, prefix_len: int, m_len: int, ego_len: int, exo_len: int, has_exo_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        B = has_exo_mask.size(0)
        total = prefix_len + m_len + ego_len + exo_len
        key = (prefix_len, m_len, ego_len, exo_len)
        if key not in self._mask_cache:
            base_mask = torch.zeros(total, total, dtype=torch.bool, device=device)
            base_mask[:prefix_len, :prefix_len] = True
            base_mask[prefix_len:, :prefix_len] = True
            base_mask[prefix_len:, prefix_len:] = True
            self._mask_cache[key] = base_mask
        batch_mask = self._mask_cache[key].unsqueeze(0).expand(B, -1, -1).clone()
        missing_exo_idx = has_exo_mask == 0
        if missing_exo_idx.any():
            exo_start = prefix_len + m_len + ego_len
            batch_mask[missing_exo_idx, :, exo_start:] = False
            batch_mask[missing_exo_idx, exo_start:, :] = False
            batch_mask[missing_exo_idx, exo_start:, exo_start:] = True
        batch_mask = batch_mask.unsqueeze(1)
        return batch_mask

    def _get_causal_attention_mask_no_exo(self, batch_size: int, prefix_len: int, m_len: int, ego_len: int, device: torch.device) -> torch.Tensor:
        B = batch_size
        total = prefix_len + m_len + ego_len
        key = (prefix_len, m_len, ego_len)
        if key not in self._mask_cache:
            base_mask = torch.zeros(total, total, dtype=torch.bool, device=device)
            base_mask[:prefix_len, :prefix_len] = True
            base_mask[prefix_len:, :prefix_len] = True
            for i in range(m_len):
                base_mask[prefix_len + i, prefix_len:prefix_len + i + 1] = True
            for i in range(ego_len):
                curr_pos = prefix_len + m_len + i
                base_mask[curr_pos, prefix_len + m_len:curr_pos + 1] = True
            for i in range(m_len):
                base_mask[prefix_len + i, prefix_len + m_len:] = True
            for i in range(ego_len):
                base_mask[prefix_len + m_len + i, prefix_len:prefix_len + m_len] = True
            self._mask_cache[key] = base_mask
        batch_mask = self._mask_cache[key].unsqueeze(0).expand(B, -1, -1).clone()
        batch_mask = batch_mask.unsqueeze(1)
        return batch_mask

    def _view_embedding(self, batch_size, view_token):
        # Training encodes ego as 1 and exo as 0. Keep checkpoint parameter names.
        if view_token is None:
            return self.view_1.expand(batch_size, -1, -1)
        token = torch.as_tensor(view_token, device=self.view_1.device)
        if token.numel() not in (1, batch_size) or not ((token == 0) | (token == 1)).all():
            raise ValueError("view_token must contain one 0 (exo) or 1 (ego) per sample")
        token = token.reshape(-1, 1, 1).expand(batch_size, 1, 1)
        return torch.where(token == 1, self.view_1, self.view_3)

    def _predict_tokens(self, context_tokens, vlm_feature, text_feature, view_token):
        ref = self.predictor_embed_ego.weight
        context_tokens = context_tokens.to(device=ref.device, dtype=ref.dtype)
        vlm_feature = vlm_feature.to(device=ref.device, dtype=ref.dtype)
        batch, frames, patches, _ = context_tokens.shape
        context = self.predictor_embed_ego(context_tokens).view(batch, frames * patches, -1)
        view = self.view_proj(self._view_embedding(batch, view_token))
        vlm = self.vlm_encoder(vlm_feature).unsqueeze(1)
        text = self.text_proj(text_feature.to(device=ref.device, dtype=ref.dtype))
        prefix = torch.cat([vlm, text, view, context], dim=1)
        motion = self.motion_queries.expand(batch, -1, -1)
        video = self.ego_queries.expand(batch, -1, -1)
        for block in self.predictor_blocks:
            prefix, motion, video = block(prefix, motion, video, mask=None, attn_mask=None,
                                         T=self.action_horizon, H=16, W=16, action_tokens=0)
        return motion, video, text, context

    def _state_features(self, states):
        if states is None:
            return None
        ref = self.state_encoder.weight
        return self.state_encoder(states.to(device=ref.device, dtype=ref.dtype))

    def forward(self, actions, timesteps, context_tokens_ego, context_latents_ego,
                states, vlm_feature, text_feature=None, view_token=None, sigmas=None):
        motion, video, text, context = self._predict_tokens(
            context_tokens_ego, vlm_feature, text_feature, view_token)
        _, pred_actions = self.action_head(actions.to(motion.dtype), motion,
            text_feature=text, states_feature=self._state_features(states), prefix=context, sigmas=sigmas)
        return self.predictor_proj(self.predictor_norm(video)), pred_actions

    def sample(self, context_tokens_ego, vlm_feature, text_feature=None,
               init_action=None, states=None, view_token=None):
        motion, _, text, _ = self._predict_tokens(context_tokens_ego, vlm_feature, text_feature, view_token)
        pred_actions = self.action_head.sample(motion, None, init_action, text, self._state_features(states))
        return None, pred_actions

    def sample_video(self, context_tokens_ego, vlm_feature, text_feature=None,
                     init_action=None, view_token=None):
        _, video, _, _ = self._predict_tokens(context_tokens_ego, vlm_feature, text_feature, view_token)
        return self.predictor_proj(self.predictor_norm(video)), None
