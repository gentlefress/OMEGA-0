"""Deterministic episode-level train/validation holdout for HA-Lab.

This is intentionally the same rule used by OpenHLM's
``openpi.ha.episode_split``: sort the complete key list, shuffle with Python's
``random.Random(seed)``, then retain a stable sorted order in each split.
"""

from __future__ import annotations

import random
from typing import Hashable, Sequence, TypeVar


T = TypeVar("T", bound=Hashable)


def split_episode_ids(
    ids: Sequence[T],
    *,
    fraction: float = 0.05,
    seed: int = 7,
) -> tuple[list[T], list[T]]:
    """Split episode ids into reproducible ``(train, val)`` lists.

    Rules match OpenHLM: zero episodes produces two empty lists; one episode
    stays in train; otherwise validation has ``max(1, round(fraction * N))``
    items, capped at ``N - 1``.
    """

    if not 0.0 <= float(fraction) < 1.0:
        raise ValueError(f"fraction must be in [0, 1), got {fraction}")

    ordered = sorted(ids, key=lambda value: (str(type(value).__name__), str(value)))
    if not ordered:
        return [], []
    if len(ordered) == 1 or float(fraction) <= 0.0:
        return list(ordered), []

    num_val = int(round(float(fraction) * len(ordered)))
    num_val = min(max(1, num_val), len(ordered) - 1)

    shuffled = list(ordered)
    random.Random(int(seed)).shuffle(shuffled)
    val_set = set(shuffled[:num_val])
    return (
        [value for value in ordered if value not in val_set],
        [value for value in ordered if value in val_set],
    )


def halab_episode_key(task_key: str, episode_index: int) -> str:
    """Stable HA-Lab key shared with ``LeRobotVLADataset.episode_keys``."""

    return f"{task_key}/{int(episode_index)}"
