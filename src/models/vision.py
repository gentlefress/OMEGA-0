"""Frozen image encoder and patch embedding."""
# Derived from the original Omega predictor; attribution: LICENSE and NOTICE.md.
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention, RoPEAttention
from .common import DropPath, MLP, SwiGLUFFN, trunc_normal_


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int | tuple[int, int], cls_token: bool=False) -> np.ndarray:
    if isinstance(grid_size, int):
        grid_h = grid_w = grid_size
    else:
        (grid_h, grid_w) = grid_size
    grid_h_vals = np.arange(grid_h, dtype=float)
    grid_w_vals = np.arange(grid_w, dtype=float)
    (grid_w_mesh, grid_h_mesh) = np.meshgrid(grid_w_vals, grid_h_vals)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h_mesh)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w_mesh)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, wide_silu=True, norm_layer=nn.LayerNorm, use_sdpa=True, is_causal=False, grid_size=16, use_rope=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_rope:
            self.attn = RoPEAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, use_sdpa=use_sdpa, is_causal=is_causal, grid_size=grid_size, proj_drop=drop)
        else:
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, use_sdpa=use_sdpa, is_causal=is_causal, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop)
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        if isinstance(self.attn, RoPEAttention):
            y = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)
        else:
            y = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class FrozenFrameEncoder(nn.Module):
    """
    VisionTransformer-style frame encoder used as frozen JEPA target encoder.

    Architecture aligned with vjepa2/src/models/vision_transformer.py:
      - PatchEmbed
      - (optional) sin-cos positional embedding + interpolation
      - Transformer Block stack
      - final norm

    We additionally keep `latent_proj` so JEPA loss can be computed in configurable
    latent dimension.
    """

    def __init__(self, image_size: int=224, patch_size: int=16, in_chans: int=3, embed_dim: int=768, latent_dim: int=768, depth: int=12, num_heads: int=12, mlp_ratio: float=4.0, qkv_bias: bool=True, qk_scale=None, drop_rate: float=0.0, attn_drop_rate: float=0.0, drop_path_rate: float=0.0, norm_layer=nn.LayerNorm, init_std: float=0.02, use_silu: bool=False, wide_silu: bool=True, use_sdpa: bool=True, use_rope: bool=False, handle_nonsquare_inputs: bool=True):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        (self.img_height, self.img_width) = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.handle_nonsquare_inputs = handle_nonsquare_inputs
        self.base_grid_h = self.img_height // self.patch_size
        self.base_grid_w = self.img_width // self.patch_size
        self.num_patches = self.base_grid_h * self.base_grid_w
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        if self.use_rope:
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(use_rope=use_rope, grid_size=self.base_grid_h, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, use_sdpa=use_sdpa, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, act_layer=nn.SiLU if use_silu else nn.GELU, wide_silu=wide_silu, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.latent_proj = nn.Linear(embed_dim, latent_dim, bias=True)
        self.init_std = init_std
        if self.pos_embed is not None:
            self._init_pos_embed(self.pos_embed.data)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed: torch.Tensor):
        embed_dim = pos_embed.size(-1)
        sincos = get_2d_sincos_pos_embed(embed_dim, (self.base_grid_h, self.base_grid_w), cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for (layer_id, layer) in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def interpolate_pos_encoding(self, x: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        (_, N, dim) = pos_embed.shape
        (_, _, H, W) = x.shape
        if H == self.img_height and W == self.img_width:
            return pos_embed
        h = H // self.patch_size
        w = W // self.patch_size
        assert h * w > 0 and N == self.base_grid_h * self.base_grid_w
        pos = pos_embed.reshape(1, self.base_grid_h, self.base_grid_w, dim).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(h, w), mode='bicubic', align_corners=False)
        pos = pos.permute(0, 2, 3, 1).view(1, h * w, dim)
        return pos

    def encode_context_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if x.ndim != 4:
            raise ValueError(f'Expected image tensor [B,C,H,W], got {x.shape}')
        (_, _, H, W) = x.shape
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        T = 1
        if not self.handle_nonsquare_inputs:
            T = h_patches = w_patches = None
        if not self.use_rope:
            pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
            x = self.patch_embed(x)
            x = x + pos_embed
        else:
            x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x, mask=None, attn_mask=None, T=T, H_patches=h_patches, W_patches=w_patches)
        x = self.norm(x)
        return (x, (H // self.patch_size, W // self.patch_size))

    def project_tokens_to_latent(self, context_tokens: torch.Tensor) -> torch.Tensor:
        token_latents = self.latent_proj(context_tokens)
        return token_latents.mean(dim=1)

    def project_tokens_to_latent_no_pooled(self, context_tokens: torch.Tensor) -> torch.Tensor:
        token_latents = self.latent_proj(context_tokens)
        return token_latents

    def encode_tokens(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        (context_tokens, (grid_h, grid_w)) = self.encode_context_tokens(frames)
        pooled_latent = self.project_tokens_to_latent(context_tokens)
        return (context_tokens, pooled_latent, (grid_h, grid_w))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        (context_tokens, _) = self.encode_context_tokens(frames)
        return self.project_tokens_to_latent(context_tokens)
