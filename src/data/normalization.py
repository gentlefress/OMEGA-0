"""Explicit, dimension-checked normalization shared by offline and online data."""
import numpy as np


class Normalizer:
    def __init__(self, config, dimension):
        self.mode = config.get("mode", "none")
        self.dimension = dimension
        self.eps = float(config.get("eps", 1e-6))
        self.clip = config.get("clip_value")
        if self.eps <= 0:
            raise ValueError("Normalization epsilon must be positive")
        def array(name):
            value = np.asarray(config[name], np.float32)
            if value.shape != (dimension,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid normalization statistic {name}")
            return value
        if self.mode == "std":
            std = array("std")
            if (std < 0).any():
                raise ValueError("Negative standard deviation")
            self.offset, self.scale = array("mean"), np.where(std < self.eps, 1., std) + self.eps
        elif self.mode in {"bounds", "bounds_q99", "latent_trigger"}:
            low, high = (array("q01"), array("q99")) if self.mode in {"bounds_q99", "latent_trigger"} else (array("min"), array("max"))
            self.offset, self.scale = low, high - low + self.eps
        elif self.mode in {"none", "latent_notrigger"}:
            self.offset, self.scale = np.zeros(dimension), np.ones(dimension)
        else:
            raise ValueError(f"Unknown normalization mode {self.mode}")
        if (self.scale <= 0).any():
            raise ValueError("Normalization scale must be positive")

    def _check(self, values):
        values = np.asarray(values, np.float32)
        if values.ndim < 1 or values.shape[-1] != self.dimension or not np.isfinite(values).all():
            raise ValueError("Normalization input has invalid shape or values")
        return values

    def normalize(self, values):
        if self.mode == "latent_trigger":
            out = self._check(values).copy()
            out[..., -2:] = 2 * (out[..., -2:] - self.offset[-2:]) / self.scale[-2:] - 1
            return out
        out = (self._check(values) - self.offset) / self.scale
        if self.mode == "std" and self.clip is not None:
            out = np.clip(out, -self.clip, self.clip)
        return (out * 2 - 1 if self.mode.startswith("bounds") else out).astype(np.float32)

    def denormalize(self, values):
        out = self._check(values)
        if self.mode == "latent_trigger":
            out = out.copy()
            out[..., -2:] = (out[..., -2:] + 1) / 2 * self.scale[-2:] + self.offset[-2:]
            return out
        if self.mode.startswith("bounds"):
            out = (out + 1) / 2
        return (out * self.scale + self.offset).astype(np.float32)


class NormalizeFields:
    def __init__(self, action, action_dim, state=None, state_dim=None):
        self.action = Normalizer(action, action_dim)
        self.state = Normalizer(state, state_dim) if state is not None else None

    def __call__(self, data, **kwargs):
        data = dict(data)
        data["raw_actions"] = data["actions"].copy()
        data["actions"] = self.action.normalize(data["actions"])
        if self.state is not None and data.get("states") is not None:
            data["states"] = self.state.normalize(data["states"])
        return data


class ActionStateTransform:
    """Load named statistics once and normalize training actions and states."""

    def __init__(self, stat_path, state_stat_path=None, stat_action_key="wam_smpl",
                 stat_state_key="wam_smpl", action_norm_type="bounds_q99",
                 normalize_state=True, state_norm_type="bounds_q99", clip_value=None,
                 eps=1e-6):
        import json
        from pathlib import Path

        with Path(stat_path).expanduser().open() as handle:
            action_stats = json.load(handle)
        if state_stat_path:
            with Path(state_stat_path).expanduser().open() as handle:
                state_stats = json.load(handle)
        else:
            state_stats = action_stats
        self.action_config = dict(action_stats[stat_action_key], mode=action_norm_type,
                                  eps=eps, clip_value=clip_value)
        self.state_config = (dict(state_stats[stat_state_key], mode=state_norm_type,
                                  eps=eps, clip_value=clip_value) if normalize_state else None)
        self.normalizers = {}

    def _normalize(self, value, field, config):
        dimension = value.shape[-1]
        key = (field, dimension)
        if key not in self.normalizers:
            self.normalizers[key] = Normalizer(config, dimension)
        return self.normalizers[key].normalize(value)

    def __call__(self, data, **kwargs):
        data = dict(data)
        data["raw_actions"] = np.array(data["actions"], copy=True)
        data["actions"] = self._normalize(data["actions"], "action", self.action_config)
        if self.state_config is not None and data.get("states") is not None:
            data["states"] = self._normalize(data["states"], "state", self.state_config)
        return data
