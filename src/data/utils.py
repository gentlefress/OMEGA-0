"""Array padding shared by whole-body transforms."""
import numpy as np


def pad_to_len(array, target_len, dim=1, pad_value=0.0):
    import torch
    axis = dim
    if array.shape[axis] >= target_len:
        mask = torch.ones_like(array, dtype=torch.bool) if isinstance(array, torch.Tensor) else np.ones(array.shape, dtype=bool)
        return array, mask
    shape = list(array.shape)
    shape[axis] = target_len - shape[axis]
    if isinstance(array, torch.Tensor):
        pad = torch.full(shape, pad_value, dtype=array.dtype, device=array.device)
        result = torch.cat((array, pad), dim=axis)
        mask = torch.cat((torch.ones_like(array, dtype=torch.bool), torch.zeros_like(pad, dtype=torch.bool)), dim=axis)
    else:
        result = np.concatenate((array, np.full(shape, pad_value, dtype=array.dtype)), axis=axis)
        mask = np.concatenate((np.ones(array.shape, dtype=bool), np.zeros(shape, dtype=bool)), axis=axis)
    return result, mask
