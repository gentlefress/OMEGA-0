# Derived from Omega-0's project models; upstream attribution: LICENSE and NOTICE.md.
# Source: src/psi/models/DiffMLPs_noview_text.py
from __future__ import annotations
from .attention import Attention
from .common import _no_grad_trunc_normal_, modulate, trunc_normal_
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math
from omega.models.diffusion.diffusion import create_diffusion
import torch.nn.functional as F
from omega.models.diffusion.transport import create_transport, Sampler
import numpy as np

class _TimeNetwork(nn.Module):

    def __init__(self, time_dim, out_dim, learnable_w=False):
        assert time_dim % 2 == 0, 'time_dim must be even!'
        half_dim = int(time_dim // 2)
        super().__init__()
        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.w = nn.Parameter(w, requires_grad=learnable_w)
        self.out_net = nn.Sequential(nn.Linear(time_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, x):
        x = x[..., None] * self.w
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=-1)
        return self.out_net(x)



class CrossAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, use_sdpa=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa

    def forward(self, query, key_value, attn_mask=None):
        (B, N, C) = query.shape
        M = key_value.shape[1]
        q = self.q(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(key_value).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(key_value).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.use_sdpa:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob if self.training else 0.0, attn_mask=attn_mask)
        else:
            attn = q @ k.transpose(-2, -1) * self.scale
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class ActionDiTBlock(nn.Module):

    def __init__(self, hidden_dim: int, cond_dim: int, num_heads: int, mlp_ratio: float=4.0, dropout: float=0.0, prefix: bool=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-06)
        self.attn = Attention(dim=hidden_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-06)
        self.prefix_mask = prefix
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_hidden_dim), nn.SiLU(), nn.Linear(mlp_hidden_dim, hidden_dim))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * hidden_dim, bias=True))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def build_prefix_action_mask(self, prefix_len: int=1, action_len: int=16, device: torch.device=torch.device('cpu')) -> torch.Tensor:
        total_len = prefix_len + action_len
        mask = torch.ones((total_len, total_len), dtype=torch.bool, device=device)
        mask[:prefix_len, prefix_len:] = False
        return mask

    def forward(self, a, c, img_feature=None) -> torch.Tensor:
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=-1)
        x_norm1 = modulate(self.norm1(a), shift_msa, scale_msa)
        x = a + gate_msa * self.attn(x_norm1)
        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_norm2)
        return x

class DiT_DDIM(nn.Module):

    def __init__(self, action_horizon, target_channels, z_channels, depth, width, num_sampling_steps, learn_sigma=False):
        super(DiT_DDIM, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleDiTAdaLN(in_channels=target_channels, model_channels=width, out_channels=target_channels * 2 if learn_sigma else target_channels, z_channels=z_channels, num_res_blocks=depth, action_horizon=action_horizon)
        self.train_diffusion = create_diffusion(timestep_respacing='', noise_schedule='cosine')
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule='cosine')

    def forward(self, target, z1, text_feature=None, states_feature=None, prefix=None, view_latent=None, sigmas=None, mask=None):
        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        z1 = torch.cat([z1, text_feature], dim=1)
        if states_feature is not None:
            z1 = torch.cat([z1, states_feature], dim=1)
        model_kwargs = dict(c1=z1)
        action_prefix = None
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, action_prefix, sigmas, model_kwargs)
        loss = loss_dict['loss']
        model_output = loss_dict['pred_xstart']
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return (loss.mean(), model_output)

    def forward_fm(self, noisy_action, z1, timesteps, sigmas=None, mask=None):
        pred_vel = self.net(noisy_action, timesteps, z1)
        return pred_vel

    def sample(self, z1, prefix=None, init_action=None, text_feature=None, states_feature=None, temperature=1.0, cfg=1.0):
        noise = init_action
        z1 = torch.cat([z1, text_feature], dim=1)
        if states_feature is not None:
            z1 = torch.cat([z1, states_feature], dim=1)
        model_kwargs = dict(c1=z1, cfg_scale=cfg)
        sample_fn = self.net.forward_with_cfg
        sampled_token_latent = self.gen_diffusion.ddim_sample_loop(sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False, eta=0.0)
        return sampled_token_latent

class DiffMLPs_DDPM(nn.Module):

    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, learn_sigma=False):
        super(DiffMLPs_DDPM, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(in_channels=target_channels, model_channels=width, out_channels=target_channels * 2 if learn_sigma else target_channels, z_channels=z_channels, num_res_blocks=depth)
        self.train_diffusion = create_diffusion(timestep_respacing='', noise_schedule='cosine')
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule='cosine')

    def forward(self, target, z1, mask=None):
        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        model_kwargs = dict(c1=z1)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict['loss']
        model_output = loss_dict['pred_xstart']
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return (loss.mean(), model_output)

    def sample(self, init_action, z1, temperature=1.0, cfg=1.0):
        noise = init_action
        model_kwargs = dict(c1=z1, cfg_scale=cfg)
        sample_fn = self.net.forward_with_cfg
        sampled_token_latent = self.gen_diffusion.ddim_sample_loop(sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False, eta=0.0)
        return sampled_token_latent

def diffmlps_ddpm_xl(**kwargs):
    return DiffMLPs_DDPM(depth=2, num_sampling_steps='2', learn_sigma=False, **kwargs)

def dit_ddim_xl(**kwargs):
    return DiT_DDIM(action_horizon=25, depth=18, num_sampling_steps='2', learn_sigma=False, **kwargs)
DiffMLPs_models = {'DDPM-XL': diffmlps_ddpm_xl, 'DDIM-DiT': dit_ddim_xl}


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size, bias=True), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(device=self.mlp[0].weight.device, dtype=self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.in_ln = nn.LayerNorm(channels, eps=1e-06)
        self.mlp = nn.Sequential(nn.Linear(channels, channels, bias=True), nn.SiLU(), nn.Linear(channels, channels, bias=True))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True))

    def forward(self, x, y):
        (shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-06)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True))

    def forward(self, x, c):
        (shift, scale) = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class SimpleDiTAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output p_sample_loopTensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(self, in_channels, model_channels, out_channels, z_channels, num_res_blocks, action_horizon, grad_checkpointing=False):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing
        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.cond_mlp = nn.Linear(model_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.action_head = nn.ModuleList([ActionDiTBlock(hidden_dim=model_channels, cond_dim=model_channels, num_heads=16, mlp_ratio=4.0, dropout=0.0, prefix=False) for _ in range(18)])
        self.final_layer = FinalLayer(model_channels, out_channels)
        self.dec_pos_a = nn.Parameter(torch.empty(action_horizon, model_channels), requires_grad=True)
        trunc_normal_(self.dec_pos_a, std=0.02)
        self.initialize_weights()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        nn.init.normal_(self.cond_embed.weight, std=0.02)
        nn.init.normal_(self.cond_mlp.weight, std=0.02)
        for block in self.action_head:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c1):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C x ...] Tensor of outputs.
        """
        ref_w = self.input_proj.weight
        x = x.to(device=ref_w.device, dtype=ref_w.dtype)
        c1 = c1.to(device=ref_w.device, dtype=ref_w.dtype)
        t = t.to(device=ref_w.device, dtype=ref_w.dtype)
        # Preserve the legacy 25-row parameter and checkpoint keys. Configured
        # horizons use a deterministic interpolation of that positional table.
        position = self.dec_pos_a
        if x.size(1) != position.size(0):
            position = F.interpolate(position.T.unsqueeze(0), size=x.size(1), mode='linear', align_corners=False)[0].T
        x = self.input_proj(x) + position
        t = t.to(ref_w.dtype)
        t = self.time_embed(t).to(dtype=ref_w.dtype).unsqueeze(1)
        if c1.size(1) == x.size(1) + 2:
            text = c1[:, -2:-1]
            cond = c1[:, :-2]
            state = c1[:, -1:]
            action_cond = self.cond_embed(cond)
            y = self.cond_mlp(t + action_cond + text + state)
        elif c1.size(1) == x.size(1) + 1:
            text = c1[:, -1:]
            cond = c1[:, :-1]
            action_cond = self.cond_embed(cond)
            y = self.cond_mlp(t + action_cond + text)
        else:
            raise ValueError('Action conditioning must contain horizon tokens, text, and optional state')
        if self.grad_checkpointing and (not torch.jit.is_scripting()):
            for block in self.action_head:
                x = checkpoint(block, x, y)
        else:
            for block in self.action_head:
                x = block(x, y)
        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c1, cfg_scale):
        model_out = self.forward(x, t, c1)
        return model_out

class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output p_sample_loopTensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(self, in_channels, model_channels, out_channels, z_channels, num_res_blocks, grad_checkpointing=False):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing
        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed1 = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(model_channels))
        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)
        self.initialize_weights()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c1):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C x ...] Tensor of outputs.
        """
        ref_w = self.input_proj.weight
        x = x.to(device=ref_w.device, dtype=ref_w.dtype)
        c1 = c1.to(device=ref_w.device, dtype=ref_w.dtype)
        t = t.to(device=ref_w.device, dtype=ref_w.dtype)
        x = self.input_proj(x)
        t = self.time_embed(t).to(dtype=ref_w.dtype)
        try:
            c1 = self.cond_embed1(c1)
        except:
            print(c1.shape)
        y = t + c1
        if self.grad_checkpointing and (not torch.jit.is_scripting()):
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)
        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c1, cfg_scale):
        model_out = self.forward(x, t, c1)
        return model_out

    def forward_with_cfg_x0(self, x, t, c1, cfg_scale):
        half = x[:len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c1)
        (eps, rest) = (model_out[:, :self.in_channels], model_out[:, self.in_channels:])
        (cond_eps, uncond_eps) = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
