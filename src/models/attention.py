"""Attention and rotary positions shared by the vision encoder and predictor."""
# Derived from the original Omega predictor; attribution: LICENSE and NOTICE.md.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_3d_positions(T, H, W, device, dtype, grid_size=16):
    """
    Return:
        t_pos, h_pos, w_pos: [N] where N = T * H * W

    Assumes flatten order:
        for t in T:
            for h in H:
                for w in W:
                    token(t, h, w)
    """
    N = T * H * W
    ids = torch.arange(N, device=device)
    hw = H * W
    t_pos = (ids // hw).to(dtype=dtype)
    h_pos = (ids % hw // W).to(dtype=dtype)
    w_pos = (ids % W).to(dtype=dtype)
    h_pos = h_pos * (grid_size / H)
    w_pos = w_pos * (grid_size / W)
    return (t_pos, h_pos, w_pos)


def rotate_queries_or_keys(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    (B, num_heads, N, D) = x.size()
    assert D % 2 == 0, 'Embedding dimension must be a multiple of 2'
    pos = pos.to(device=x.device, dtype=x.dtype)
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000 ** omega
    freq = torch.einsum('...,f->...f', pos, omega)
    emb_sin = freq.sin().to(dtype=x.dtype)
    emb_cos = freq.cos().to(dtype=x.dtype)
    emb_sin = emb_sin.repeat_interleave(2, dim=-1)
    emb_cos = emb_cos.repeat_interleave(2, dim=-1)
    y = x.unflatten(-1, (-1, 2))
    (y1, y2) = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return (x * emb_cos + y * emb_sin).to(dtype=x.dtype)


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, use_sdpa=True, is_causal=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

    def forward(self, x, mask=None, attn_mask=None):
        del mask
        (B, N, C) = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        (q, k, v) = (qkv[0], qkv[1], qkv[2])
        if attn_mask is not None or self.use_sdpa:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob if self.training else 0.0, is_causal=self.is_causal, attn_mask=attn_mask)
        else:
            attn = q @ k.transpose(-2, -1) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RoPEAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, use_sdpa=True, grid_size=14, is_causal=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.d_dim = int(2 * (head_dim // 3 // 2))
        self.h_dim = int(2 * (head_dim // 3 // 2))
        self.w_dim = int(2 * (head_dim // 3 // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal

    def _get_frame_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
        else:
            tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        width_ids = ids - tokens_per_frame * frame_ids - tokens_per_row * height_ids
        return (frame_ids, height_ids, width_ids)

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        (B, N, C) = x.size()
        grid_depth = int(N // (self.grid_size * self.grid_size))
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        (q, k, v) = (qkv[0], qkv[1], qkv[2])
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            (d_mask, h_mask, w_mask) = self.separate_positions(mask, H_patches, W_patches)
        else:
            if T is None or H_patches is None or W_patches is None:
                ids = torch.arange(int(grid_depth * self.grid_size * self.grid_size), device=x.device)
            else:
                ids = torch.arange(int(T * H_patches * W_patches), device=x.device)
            (d_mask, h_mask, w_mask) = self.separate_positions(ids, H_patches, W_patches)
        s = 0
        qd = rotate_queries_or_keys(q[..., s:s + self.d_dim], pos=d_mask)
        kd = rotate_queries_or_keys(k[..., s:s + self.d_dim], pos=d_mask)
        s += self.d_dim
        qh = rotate_queries_or_keys(q[..., s:s + self.h_dim], pos=h_mask)
        kh = rotate_queries_or_keys(k[..., s:s + self.h_dim], pos=h_mask)
        s += self.h_dim
        qw = rotate_queries_or_keys(q[..., s:s + self.w_dim], pos=w_mask)
        kw = rotate_queries_or_keys(k[..., s:s + self.w_dim], pos=w_mask)
        s += self.w_dim
        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)
        if attn_mask is not None or self.use_sdpa:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob if self.training else 0.0, is_causal=self.is_causal, attn_mask=attn_mask)
        else:
            attn = q @ k.transpose(-2, -1) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ACRoPESelfAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, use_sdpa=True, is_causal=False, grid_size=14):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.d_dim = int(2 * (head_dim // 3 // 2))
        self.h_dim = int(2 * (head_dim // 3 // 2))
        self.w_dim = int(2 * (head_dim // 3 // 2))
        self.grid_size = grid_size

    def _apply_1d_temporal_rope(self, q, k, num_tokens):
        if num_tokens <= 0:
            return (q, k)
        pos = torch.arange(num_tokens, device=q.device, dtype=q.dtype).unsqueeze(0)
        q_d = rotate_queries_or_keys(q[..., :self.d_dim], pos)
        k_d = rotate_queries_or_keys(k[..., :self.d_dim], pos)
        q_out = torch.cat([q_d, q[..., self.d_dim:]], dim=-1)
        k_out = torch.cat([k_d, k[..., self.d_dim:]], dim=-1)
        return (q_out, k_out)

    def _apply_image_rope_qk(self, q_img, k_img, H=None, W=None):
        """
        q_img, k_img: [B, heads, N_img, head_dim]
        Apply spatial RoPE to image tokens.
        """
        N_img = q_img.shape[2]
        if H is None or W is None:
            return (q_img, k_img)
        assert H * W == N_img, f'Image RoPE mismatch: H*W={H * W}, but N_img={N_img}. Check prefix length and num_vlm.'
        ids_img = torch.arange(N_img, device=q_img.device)
        h_pos = (ids_img // W).to(dtype=q_img.dtype) * (self.grid_size / H)
        w_pos = (ids_img % W).to(dtype=q_img.dtype) * (self.grid_size / W)
        d_pos = torch.zeros_like(h_pos)
        q_img_d = rotate_queries_or_keys(q_img[..., :self.d_dim], d_pos.unsqueeze(0))
        k_img_d = rotate_queries_or_keys(k_img[..., :self.d_dim], d_pos.unsqueeze(0))
        s = self.d_dim
        q_img_h = rotate_queries_or_keys(q_img[..., s:s + self.h_dim], h_pos.unsqueeze(0))
        k_img_h = rotate_queries_or_keys(k_img[..., s:s + self.h_dim], h_pos.unsqueeze(0))
        s += self.h_dim
        q_img_w = rotate_queries_or_keys(q_img[..., s:s + self.w_dim], w_pos.unsqueeze(0))
        k_img_w = rotate_queries_or_keys(k_img[..., s:s + self.w_dim], w_pos.unsqueeze(0))
        s += self.w_dim
        q_img = torch.cat([q_img_d, q_img_h, q_img_w, q_img[..., s:]], dim=-1)
        k_img = torch.cat([k_img_d, k_img_h, k_img_w, k_img[..., s:]], dim=-1)
        return (q_img, k_img)

    def apply_3d_video_rope(self, q, k, T, H, W):
        """
        q, k: [B, heads, N, head_dim]
        N must equal T * H * W
        """
        (B, heads, N, D) = q.shape
        assert N == T * H * W, f'Video token length mismatch: N={N}, but T*H*W={T * H * W}. Check your video token layout.'
        (t_pos, h_pos, w_pos) = build_3d_positions(T=T, H=H, W=W, device=q.device, dtype=q.dtype, grid_size=self.grid_size)
        q_t = rotate_queries_or_keys(q[..., :self.d_dim], t_pos.unsqueeze(0))
        k_t = rotate_queries_or_keys(k[..., :self.d_dim], t_pos.unsqueeze(0))
        s = self.d_dim
        q_h = rotate_queries_or_keys(q[..., s:s + self.h_dim], h_pos.unsqueeze(0))
        k_h = rotate_queries_or_keys(k[..., s:s + self.h_dim], h_pos.unsqueeze(0))
        s += self.h_dim
        q_w = rotate_queries_or_keys(q[..., s:s + self.w_dim], w_pos.unsqueeze(0))
        k_w = rotate_queries_or_keys(k[..., s:s + self.w_dim], w_pos.unsqueeze(0))
        s += self.w_dim
        q = torch.cat([q_t, q_h, q_w, q[..., s:]], dim=-1)
        k = torch.cat([k_t, k_h, k_w, k[..., s:]], dim=-1)
        return (q, k)

    def forward(self, x, mask=None, attn_mask=None, H=None, W=None, T=None, num_vlm=None, num_img=None, num_mot=None, num_vid_ego=None, num_vid_exo=None, action_tokens=0, has_exo=False):
        (B, N, C) = x.size()
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        (q, k, v) = (qkv[0], qkv[1], qkv[2])
        if num_vlm is not None:
            split_sizes = [num_vlm, num_img]
            q_splits = list(q.split(split_sizes, dim=2))
            k_splits = list(k.split(split_sizes, dim=2))
            (q_vlm, q_img) = q_splits
            (k_vlm, k_img) = k_splits
        if H is not None and W is not None and (num_img is not None):
            ids_img = torch.arange(H * W, device=x.device)
            h_pos = (ids_img // W).float() * (self.grid_size / H)
            w_pos = (ids_img % W).float() * (self.grid_size / W)
            d_pos = torch.zeros_like(h_pos)
            q_img_d = rotate_queries_or_keys(q_img[..., 0:self.d_dim], d_pos.unsqueeze(0))
            k_img_d = rotate_queries_or_keys(k_img[..., 0:self.d_dim], d_pos.unsqueeze(0))
            s = self.d_dim
            q_img_h = rotate_queries_or_keys(q_img[..., s:s + self.h_dim], h_pos.unsqueeze(0))
            k_img_h = rotate_queries_or_keys(k_img[..., s:s + self.h_dim], h_pos.unsqueeze(0))
            s += self.h_dim
            q_img_w = rotate_queries_or_keys(q_img[..., s:s + self.w_dim], w_pos.unsqueeze(0))
            k_img_w = rotate_queries_or_keys(k_img[..., s:s + self.w_dim], w_pos.unsqueeze(0))
            s += self.w_dim
            q_img = torch.cat([q_img_d, q_img_h, q_img_w, q_img[..., s:]], dim=-1)
            k_img = torch.cat([k_img_d, k_img_h, k_img_w, k_img[..., s:]], dim=-1)
            q = torch.cat([q_vlm, q_img], dim=2)
            k = torch.cat([k_vlm, k_img], dim=2)
        if self.use_sdpa:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob if self.training else 0.0, is_causal=False)
        else:
            attn = q @ k.transpose(-2, -1) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ACRoPECrossAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, use_sdpa=True, grid_size=14):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.d_dim = int(2 * (head_dim // 3 // 2))
        self.h_dim = int(2 * (head_dim // 3 // 2))
        self.w_dim = int(2 * (head_dim // 3 // 2))
        self.grid_size = grid_size

    def _apply_1d_temporal_rope(self, q, num_tokens):
        pos = torch.arange(num_tokens, device=q.device, dtype=q.dtype).unsqueeze(0)
        q_d = rotate_queries_or_keys(q[..., :self.d_dim], pos)
        q_out = torch.cat([q_d, q[..., self.d_dim:]], dim=-1)
        return q_out

    def forward(self, q_seq, kv_seq, H=None, W=None, T=None, num_vlm=None, num_img=None, num_mot=None, num_vid_ego=None, num_vid_exo=None, has_exo=False):
        (B, Nq, C) = q_seq.shape
        (B, Nkv, C) = kv_seq.shape
        q = self.q_proj(q_seq).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(kv_seq).unflatten(-1, (2, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        (k, v) = (kv[0], kv[1])
        if num_vlm is not None:
            # Match the deployed WAM: motion/video queries carry temporal
            # positions while image keys carry spatial positions.
            if num_mot is not None:
                q = self._apply_1d_temporal_rope(q, num_mot)
            elif has_exo:
                ego, exo = q.split([num_vid_ego, num_vid_exo], dim=2)
                q = torch.cat([ego, self._apply_1d_temporal_rope(exo, num_vid_exo)], dim=2)
            else:
                q = self._apply_1d_temporal_rope(q, num_vid_ego)
            split_sizes_kv = [num_vlm, num_img]
            k_splits = list(k.split(split_sizes_kv, dim=2))
            (k_vlm, k_img) = k_splits
            ids_img = torch.arange(H * W, device=q_seq.device)
            h_pos = (ids_img // W).float() * (self.grid_size / H)
            w_pos = (ids_img % W).float() * (self.grid_size / W)
            d_pos = torch.zeros_like(h_pos)
            k_img_d = rotate_queries_or_keys(k_img[..., :self.d_dim], d_pos.unsqueeze(0))
            s = self.d_dim
            k_img_h = rotate_queries_or_keys(k_img[..., s:s + self.h_dim], h_pos.unsqueeze(0))
            s += self.h_dim
            k_img_w = rotate_queries_or_keys(k_img[..., s:s + self.w_dim], w_pos.unsqueeze(0))
            s += self.w_dim
            k_img = torch.cat([k_img_d, k_img_h, k_img_w, k_img[..., s:]], dim=-1)
            k = torch.cat([k_vlm, k_img], dim=2)
        else:
            q = self._apply_1d_temporal_rope(q, num_mot)
            k = self._apply_1d_temporal_rope(k, num_vid_ego)
        if self.use_sdpa:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob if self.training else 0.0, is_causal=False)
        else:
            attn = q @ k.transpose(-2, -1) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
