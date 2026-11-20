#!/usr/bin/env python3
"""HA-Lab LeRobot episodes as ω-0 (OMEGA-0) WAM fine-tuning data.

``main`` (run once, CPU): per episode of ``--data-parent`` one ``annotation/<split>/<task>__<idx>.hdf5``

* ``latent`` (T, 78) = ``action.latent_state`` (64 SONIC token) | ``action.hand_action`` (12 Inspire
  openness, 1 = open, [left 6 | right 6]) | ``action.camera_action`` (2); a 76-D run reads the first 76;
* ``state`` (T, 45) = ω-0's ``sonic_state`` from ``observation.state``: 29 q | 5+5 Inspire openness |
  pelvis rot6d with the episode's first yaw removed (``omega0_home_stats.rot6d_yaw_removed``);
* ``instruction`` (``meta/episodes.jsonl`` task), ``view`` = ``first``, attr ``video`` = the episode's
  ``observation.images.left`` mp4 (read in place, no copy); T = min(parquet rows, video frames).

Split = the GR00T / OpenHLM global episode split (``split_episode_ids`` over every
``<task>/<episode_index>`` key, fraction 0.05, seed 7; same discovery rules as
``vla_task_registry.resolve_dataset_roots``), so validation episodes match the GR00T 38-task run.
``stats.json`` (train rows): ``halab76`` / ``halab78`` action mean/std/min/max/q01/q99 and
``halab_state`` (45), keyed for ``omega.data.normalization.ActionStateTransform``.

``HALabOmegaDataset`` is the training reader: ``omega.data.wholebody.finetune.FinetuneDataset``'s
sample (same frame/action indexing, boundary masks and output keys) on these files, with two
differences that do not change the loss:

* video is decoded with PyAV (OMEGA's reader needs decord, which cannot read every codec);
* ``context_images`` holds only the current frame.  The WAM uses ``context_tokens_ego[:, 0:1]``
  and never the rest, but encodes every context frame through the frozen ViT-g first (51 frames
  for a 50-row chunk); ``next_observations`` still carries all 51 frames for the Wan targets.

    <OMEGA-0>/.venv/bin/python omega_halab_data.py --data-parent <halab> --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

# observation.state (76) = [29 q | 29 dq | 4 quat wxyz | 12 Inspire openness | 2 camera].
Q, QUAT, HAND = slice(0, 29), slice(58, 62), slice(62, 74)
LATENT_DIM = 78


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def episode_split(data_parent: Path, fraction: float, seed: int):
    """``(task_roots, train_keys, val_keys)`` exactly as ``build_openhlm_halab_episode_split``.

    ``vla_task_registry.py`` / ``episode_split.py`` next to this file are unchanged copies of
    HA-Lab ``data_loader/ha/`` (task discovery and the OpenHLM split rule).
    """
    here = Path(__file__).resolve().parent
    registry = _load_module(here / "vla_task_registry.py")
    splitter = _load_module(here / "episode_split.py")
    roots = registry.resolve_dataset_roots(data_parent=data_parent)
    keys = []
    for root in roots:
        for line in (root / "meta" / "episodes.jsonl").read_text().splitlines():
            if line.strip():
                keys.append(splitter.halab_episode_key(root.name, int(json.loads(line)["episode_index"])))
    train, val = splitter.split_episode_ids(keys, fraction=fraction, seed=seed)
    return roots, train, val


def video_frames(path: Path) -> int:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.frames:
            return int(stream.frames)
        return sum(1 for _ in container.decode(stream))


def convert_episode(root: Path, info: dict, row: dict, rot6d) -> dict | None:
    import pandas as pd

    index = int(row["episode_index"])
    chunk = index // int(info.get("chunks_size", 1000))
    data_path = root / info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet").format(
        episode_chunk=chunk, episode_index=index)
    video = root / info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4").format(
        episode_chunk=chunk, video_key="observation.images.left", episode_index=index)
    if not data_path.exists() or not video.exists():
        print(f"[skip] {root.name}/{index}: missing {data_path if not data_path.exists() else video}", flush=True)
        return None
    df = pd.read_parquet(data_path, columns=["observation.state", "action.latent_state", "action.hand_action", "action.camera_action"])
    col = lambda name: np.stack(df[name].to_numpy()).astype(np.float32)
    obs = col("observation.state")
    latent = np.concatenate([col("action.latent_state"), col("action.hand_action"), col("action.camera_action")], axis=1)
    n = min(len(obs), video_frames(video))
    if n < 2:
        print(f"[skip] {root.name}/{index}: {n} frames", flush=True)
        return None
    obs, latent = obs[:n], latent[:n]
    hands = obs[:, HAND]
    state = np.concatenate([obs[:, Q], hands[:, 0:5], hands[:, 6:11], rot6d(obs[:, QUAT], obs[0, QUAT])], axis=1)
    tasks = row.get("tasks") or [root.name.replace("_", " ")]
    return {"latent": latent, "state": state.astype(np.float32), "instruction": str(tasks[0]), "video": str(video)}


def stats(values: np.ndarray) -> dict[str, list[float]]:
    return {"mean": values.mean(0).tolist(), "std": values.std(0).tolist(),
            "min": values.min(0).tolist(), "max": values.max(0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(), "q99": np.quantile(values, 0.99, axis=0).tolist()}


def main() -> int:
    import h5py

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-parent", required=True, help="HA-Lab LeRobot task folders (e.g. datasets/ha-lab-dataset-38tasks/halab)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--val-episode-fraction", type=float, default=0.05)
    parser.add_argument("--val-split-seed", type=int, default=7)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from omega0_home_stats import rot6d_yaw_removed

    out = Path(args.out)
    roots, train_keys, val_keys = episode_split(Path(args.data_parent), args.val_episode_fraction, args.val_split_seed)
    split_of = {**{k: "train" for k in train_keys}, **{k: "val" for k in val_keys}}
    print(f"{len(roots)} tasks, {len(train_keys)} train / {len(val_keys)} val episodes", flush=True)
    actions, states, written = [], [], {"train": [], "val": []}
    for root in roots:
        info = json.loads((root / "meta" / "info.json").read_text())
        for line in (root / "meta" / "episodes.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = f"{root.name}/{int(row['episode_index'])}"
            ep = convert_episode(root, info, row, rot6d_yaw_removed)
            if ep is None:
                continue
            split = split_of[key]
            path = out / "annotation" / split / f"{root.name}__{int(row['episode_index']):06d}.hdf5"
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "w") as f:
                f["latent"], f["state"] = ep["latent"], ep["state"]
                f["instruction"], f["view"] = ep["instruction"], "first"
                f.attrs["video"], f.attrs["episode_key"] = ep["video"], key
            written[split].append(key)
            if split == "train":
                actions.append(ep["latent"])
                states.append(ep["state"])
        print(f"[convert] {root.name}: {len(written['train']) + len(written['val'])} episodes so far", flush=True)
    action, state = np.concatenate(actions).astype(np.float64), np.concatenate(states).astype(np.float64)
    (out / "stats.json").write_text(json.dumps({"halab78": stats(action), "halab76": stats(action[:, :76]),
                                                "halab_state": stats(state)}, indent=1))
    (out / "split.json").write_text(json.dumps({"scheme": "openhlm_global_episode_split", "fraction": args.val_episode_fraction,
                                                "seed": args.val_split_seed, "train_keys": written["train"],
                                                "val_keys": written["val"]}, indent=1))
    print(f"wrote {len(written['train'])} train / {len(written['val'])} val episodes, {len(action)} train rows -> {out}", flush=True)
    return 0


def read_frames(path: str, start: int, count: int, length: int) -> np.ndarray:
    """Frames ``start .. start+count-1`` (indices past ``length - 1`` repeat the last frame), (count, H, W, 3)."""
    import av

    last = min(start + count - 1, length - 1)
    frames = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        fps, tb, t0 = float(stream.average_rate), float(stream.time_base), stream.start_time or 0
        container.seek(t0 + int(start / fps / tb), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            index = int(round((frame.pts - t0) * tb * fps))
            if index < start:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            if index >= last:
                break
    if not frames:
        raise ValueError(f"no frames decoded from {path} at {start}")
    frames += [frames[-1]] * (count - len(frames))
    return np.stack(frames[:count])


class HALabOmegaDataset:
    """``FinetuneDataset`` sample layout over ``annotation/<split>/*.hdf5`` (see module docstring)."""

    def __init__(self, data_root: str, split: str = "train", action_chunk_size: int = 50, action_dim: int = 76,
                 upsample_rate: int = 1, *, text_model: str, local_files_only: bool = True) -> None:
        import h5py
        from transformers import T5Tokenizer

        self.action_chunk_size, self.action_dim, self.upsample_rate = int(action_chunk_size), int(action_dim), int(upsample_rate)
        if not 0 < self.action_dim <= LATENT_DIM:
            raise ValueError(f"action_dim must be in 1..{LATENT_DIM}")
        if self.upsample_rate != 1:
            raise ValueError("HALabOmegaDataset decodes consecutive frames only (upsample_rate 1)")
        self.files = sorted((Path(data_root) / "annotation" / split).glob("*.hdf5"))
        if not self.files:
            raise ValueError(f"no episodes under {data_root}/annotation/{split}")
        self.lengths = []
        for path in self.files:
            with h5py.File(path, "r") as f:
                self.lengths.append(int(f["latent"].shape[0]))
        self.cumsum = np.cumsum(self.lengths)
        self.tokenizer = T5Tokenizer.from_pretrained(text_model, use_fast=False, local_files_only=local_files_only)

    def __len__(self) -> int:
        return int(self.cumsum[-1])

    def __getitem__(self, idx: int) -> dict:
        import h5py

        episode = int(np.searchsorted(self.cumsum, idx, side="right"))
        frame_id = idx - (int(self.cumsum[episode - 1]) if episode else 0)
        length = self.lengths[episode]
        with h5py.File(self.files[episode], "r") as f:
            latent = f["latent"][()][:, : self.action_dim].astype(np.float32)
            state = f["state"][frame_id : frame_id + 1].astype(np.float32)
            instruction = f["instruction"][()]
            video = str(f.attrs["video"])
        instruction = (instruction.decode() if isinstance(instruction, bytes) else str(instruction)).strip().lower()
        # common.future_indices / action_indices (upsample_rate 1): frames t .. t+H, actions t+1 .. t+H.
        steps = np.arange(self.action_chunk_size + 1) * self.upsample_rate
        obs_mask = (frame_id + steps < length).astype(np.float32)
        action_raw = frame_id + steps[1:]
        action_mask = (action_raw < length).astype(np.float32)
        actions = latent[np.minimum(action_raw, length - 1)]
        frames = read_frames(video, frame_id, self.action_chunk_size + 1, length)
        encoded = self.tokenizer(instruction, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        return {"states": state, "actions": actions, "action_mask": action_mask,
                "observations": frames[0:1], "current_images": frames[0:1],
                "context_images": frames[0:1], "context_valid": obs_mask[0:1],
                "next_observations": frames, "view_token": 1, "instruction": instruction, "view": "first",
                "dataset": "halab", "dataset_name": "halab", "episode_index": episode, "frame_id": frame_id,
                "obs_mask": obs_mask, "text_input_ids": encoded["input_ids"], "text_attention_mask": encoded["attention_mask"]}


if __name__ == "__main__":
    raise SystemExit(main())
