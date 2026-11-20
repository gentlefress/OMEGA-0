"""Discover LeRobot v2.1 task dataset roots under a shared data parent directory."""

from __future__ import annotations

from pathlib import Path

# Non-task folders under slow_storage/data (tests, scratch, etc.).
_DEFAULT_EXCLUDE_DIR_NAMES = frozenset(
    {
        "garbage",
        "Andrew_test",
        "data_save_test",
        "data_save_test_v2",
    }
)


def is_lerobot_dataset_root(path: Path) -> bool:
    path = path.expanduser().resolve()
    return (path / "meta" / "info.json").is_file() and (path / "meta" / "episodes.jsonl").is_file()


def discover_lerobot_dataset_roots(
    data_parent: Path,
    *,
    task_names: list[str] | None = None,
    exclude_dir_names: frozenset[str] = _DEFAULT_EXCLUDE_DIR_NAMES,
    exclude_name_substrings: tuple[str, ...] = ("(Copy)",),
) -> list[Path]:
    """
    Return sorted LeRobot dataset roots under ``data_parent``.

    A valid task folder has ``meta/info.json`` and ``meta/episodes.jsonl``.
    New tasks appear automatically when added as sibling directories.
    """
    parent = data_parent.expanduser().resolve()
    if not parent.is_dir():
        raise FileNotFoundError(f"data parent not found: {parent}")

    if task_names is not None:
        roots: list[Path] = []
        for name in task_names:
            candidate = parent / name
            if not is_lerobot_dataset_root(candidate):
                raise FileNotFoundError(f"task dataset not found or invalid: {candidate}")
            roots.append(candidate)
        return roots

    roots = []
    for child in sorted(parent.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        if child.name in exclude_dir_names:
            continue
        if any(substr in child.name for substr in exclude_name_substrings):
            continue
        if is_lerobot_dataset_root(child):
            roots.append(child.resolve())
    return roots


def resolve_dataset_roots(
    *,
    data_parent: Path | None = None,
    dataset_root: Path | None = None,
    dataset_roots: list[Path] | None = None,
    task_names: list[str] | None = None,
) -> list[Path]:
    """Resolve one or many task roots from CLI / training config fields."""
    if dataset_roots:
        roots = [Path(p).expanduser().resolve() for p in dataset_roots]
        for root in roots:
            if not is_lerobot_dataset_root(root):
                raise FileNotFoundError(f"invalid LeRobot dataset root: {root}")
        return roots

    if dataset_root is not None:
        root = Path(dataset_root).expanduser().resolve()
        if not is_lerobot_dataset_root(root):
            raise FileNotFoundError(f"invalid LeRobot dataset root: {root}")
        return [root]

    if data_parent is not None:
        return discover_lerobot_dataset_roots(
            Path(data_parent),
            task_names=task_names,
        )

    raise ValueError("provide data_parent, dataset_root, or dataset_roots")
