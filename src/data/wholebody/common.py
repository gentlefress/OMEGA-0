"""Shared episode inspection and sampling functions for whole-body readers."""
from __future__ import annotations
import hashlib
import logging
from typing import List, Tuple
import h5py
import numpy as np
from decord import VideoReader, cpu

log = logging.getLogger(__name__)

def is_validation_episode(name, every=20):
    # Stable across missing/repaired files and independent of directory ordering.
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big") % every == 0

def latent_length(path):
    try:
        with h5py.File(path, "r") as handle:
            length = int(handle["latent"].shape[0])
            if handle["state"].shape[0] != length:
                raise ValueError("latent and state frame counts differ")
    except (OSError, KeyError, ValueError, IndexError) as error:
        raise ValueError(f"Invalid episode {path}: {error}") from error
    if length <= 0:
        log.warning("Skipping empty episode: %s", path)
    return length

def safe_index(index, length):
    return min(max(index, 0), length - 1)

def read_frames(path, indices):
    try:
        reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        if not len(reader):
            raise ValueError("empty video")
        return reader.get_batch([safe_index(int(i), len(reader)) for i in indices]).asnumpy()
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"Cannot decode video {path}: {error}") from error

def locate_frame(cumsum_lens, idx: int) -> Tuple[int, int]:
    if len(cumsum_lens) == 0 or idx < 0 or idx >= cumsum_lens[-1]:
        raise IndexError(f'Frame index out of bounds: {idx}')
    episode_id = int(np.searchsorted(cumsum_lens, idx, side='right'))
    if episode_id == 0:
        frame_id = idx
    else:
        frame_id = idx - int(cumsum_lens[episode_id - 1])
    return (episode_id, frame_id)

def context_indices(num_past_frames, upsample_rate, frame_id: int, length: int) -> Tuple[List[int], np.ndarray]:
    indices: List[int] = []
    valid: List[float] = []
    for off in range(num_past_frames, -1, -1):
        raw = frame_id - off * upsample_rate
        indices.append(safe_index(raw, length))
        valid.append(1.0 if 0 <= raw < length else 0.0)
    return (indices, np.array(valid, dtype=np.float32))

def future_indices(action_chunk_size, upsample_rate, frame_id: int, length: int) -> Tuple[List[int], np.ndarray]:
    indices: List[int] = []
    valid: List[float] = []
    start_step = 0
    action_len = action_chunk_size + 1
    for t in range(start_step, start_step + action_len):
        raw = frame_id + t * upsample_rate
        indices.append(safe_index(raw, length))
        valid.append(1.0 if 0 <= raw < length else 0.0)
    return (indices, np.array(valid, dtype=np.float32))

def action_indices(action_chunk_size, upsample_rate, use_delta_actions, frame_id: int, length: int) -> Tuple[List[int], np.ndarray]:
    start_step = 1
    action_len = action_chunk_size + (1 if use_delta_actions else 0)
    indices: List[int] = []
    valid: List[float] = []
    if action_chunk_size == 1 and not use_delta_actions:
        indices.append(safe_index(frame_id + 1, length))
        valid.append(1.0 if 0 <= frame_id + 1 < length else 0.0)
    else:
        for t in range(start_step, start_step + action_len):
            raw = frame_id + t * upsample_rate
            indices.append(safe_index(raw, length))
            valid.append(1.0 if 0 <= raw < length else 0.0)
    return (indices, np.array(valid, dtype=np.float32))
