"""Resource loading for pretrained models and configured data components."""
from __future__ import annotations

from importlib import import_module


def resolve_factory(name: str):
    """Resolve an explicitly configured, trusted Python ``module:object`` factory."""
    module, sep, attribute = name.partition(":")
    if not sep or not module or not attribute or module.split(".")[0] in {"psi", "gear_sonic"}:
        raise ValueError("Expected an independent module:object factory")
    value = import_module(module)
    for part in attribute.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"{name} is not callable")
    return value


def load_text_encoder(config):
    from transformers import T5EncoderModel
    return T5EncoderModel.from_pretrained(config.text_model, local_files_only=config.local_files_only)


class WanVideoEncoder:
    """Lazily load the TI2V-5B VAE, preserving batched FP32 latent processing.

    Wan is only needed for video supervision/reconstruction. Loading on the first
    input's device also respects the local device selected by distributed training.
    Its frozen weights remain outside the trainable WAM checkpoint.
    """

    def __init__(self, checkpoint: str | None):
        self.checkpoint = checkpoint
        self._encoder = None

    def _load(self, device):
        import torch
        from pathlib import Path
        from .wan_vae import Wan2_2_VAE

        if self._encoder is None:
            if not self.checkpoint:
                raise ValueError("Video training/reconstruction requires wan_checkpoint (Wan2.2_VAE.pth)")
            path = Path(self.checkpoint).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Wan2.2 VAE checkpoint not found: {path}")
            self._encoder = Wan2_2_VAE(vae_pth=str(path), device=device, dtype=torch.float32)
        elif torch.device(self._encoder.device) != device:
            self._encoder.model.to(device)
            self._encoder.scale = [value.to(device) for value in self._encoder.scale]
            self._encoder.device = device
        return self._encoder

    def encode(self, videos):
        import torch

        if videos.ndim not in (4, 5) or videos.shape[-4] != 3:
            raise ValueError("Wan videos must have shape [B, 3, T, H, W] or [3, T, H, W]")
        encoder = self._load(videos.device)
        batched = videos.ndim == 5
        # The upstream wrapper loops over a list. Its underlying model supports
        # a whole batch, as used by the original OMEGA training implementation.
        with torch.autocast(device_type=videos.device.type, enabled=False):
            latent = encoder.model.encode(videos.float() if batched else videos.float().unsqueeze(0), encoder.scale)
        return latent.float() if batched else latent.float().squeeze(0)

    def decode(self, latents):
        import torch

        if latents.ndim not in (4, 5) or latents.shape[-4] != 48:
            raise ValueError("Wan latents must have shape [B, 48, T, H, W] or [48, T, H, W]")
        encoder = self._load(latents.device)
        batched = latents.ndim == 5
        with torch.autocast(device_type=latents.device.type, enabled=False):
            video = encoder.model.decode(latents.float() if batched else latents.float().unsqueeze(0), encoder.scale)
        video = video.float().clamp_(-1, 1)
        return video if batched else video.squeeze(0)
