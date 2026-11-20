#!/usr/bin/env python3
"""Rebuild ω-0's unpublished normalization stats from the released ω-HOME recordings.

The HF checkpoint ``keycharon/Omega-zero`` denormalizes its 66-D actions with
``wam_stat_no_human.json`` (``action_norm_type: std``) and normalizes its 45-D state
with ``wam_stat_state_no_human.json`` (``state_norm_type: bounds_q99``); neither file
is released.  This recomputes both from ω-HOME ``state_action.hdf5`` files of the
paper's 11 Stage-3 fine-tuning tasks, in the layouts the checkpoint consumes:

* action (66) = ``sonic/token_state`` (64) + ``action/left_trigger`` + ``action/right_trigger``
  (OMEGA ``tools/export_training_data.py``: ``latent = [token, triggers]``);
* state (45) = ``state/rt_lowstate[:29]`` joints + ``state/inspire_{left,right}_state[:5] / 1000``
  + 6-D pelvis rotation with the episode's first yaw removed (``real/transforms/state.py::sonic_state``,
  ``rotation.as_matrix()[:, :2].reshape(-1)``).

Every statistic any normalizer mode needs (mean/std/min/max/q01/q99) is written, under the
keys and file names the checkpoint's ``run_config.json`` names.  The result is an
approximation of the originals (task subset and the exact recording streams of the
authors' export are not published).

    python omega0_home_stats.py --home-root <ω-HOME download> --run-config <ckpt>/run_config.json \\
        --out-dir <external_models>/stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Paper Table 1 (Stage-3 fine-tuning set) -> ω-HOME folder names.
OMEGA_FT_TASKS = (
    "pick_and_place_apple",               # Pick an apple and place into a basket
    "arrange_fruit_in_the_closet",        # Arrange an apple on a shelf
    "put_clothes_into_bucket",            # Pick clothes from bed and throw into basket
    "put_towel_into_washing_machine",     # Move towel from basket to washing machine
    "wipe_table",                         # Wipe the table
    "mop_floor",                          # Mop the floor
    "pick_garbage",                       # Pick trash from different heights into handheld bin
    "put_apple_and_close_drawer",         # Pick apple from table, throw into drawer, close drawer with knee
    "clean_bed",                          # Sweep trash from bed, turn around, throw into bin
    "pick_clothes_from_washing_machine",  # Take clothes out of the washing machine
    "retrieve_from_the_upper_fridge",     # Retrieve a drink from the fridge
)


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def rot6d_yaw_removed(quat_wxyz: np.ndarray, reference_wxyz: np.ndarray) -> np.ndarray:
    """``sonic_state``: Rz(-yaw(reference)) * R, first two columns, row-major -> (N, 6)."""
    mats = quat_wxyz_to_matrix(np.asarray(quat_wxyz, np.float64))
    ref = quat_wxyz_to_matrix(np.asarray(reference_wxyz, np.float64)[None])[0]
    yaw = np.arctan2(ref[1, 0], ref[0, 0])
    c, s = np.cos(-yaw), np.sin(-yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (rz[None] @ mats)[:, :, :2].reshape(len(mats), 6)


def episode_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as f:
        token = f["sonic/token_state"][()].astype(np.float64)
        trig = np.concatenate([f["action/left_trigger"][()], f["action/right_trigger"][()]], axis=1).astype(np.float64)
        low = f["state/rt_lowstate"][()].astype(np.float64)
        left = f["state/inspire_left_state"][()].astype(np.float64)[:, :5] / 1000.0
        right = f["state/inspire_right_state"][()].astype(np.float64)[:, :5] / 1000.0
    n = min(len(token), len(trig), len(low), len(left), len(right))
    action = np.concatenate([token[:n], trig[:n]], axis=1)
    state = np.concatenate([low[:n, :29], left[:n], right[:n], rot6d_yaw_removed(low[:n, 29:33], low[0, 29:33])], axis=1)
    return action, state


def stats(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": values.mean(0).tolist(), "std": values.std(0).tolist(),
        "min": values.min(0).tolist(), "max": values.max(0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(), "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--home-root", required=True, help="local ω-HOME root (<task>/<episode>/state_action.hdf5)")
    parser.add_argument("--run-config", required=True, help="checkpoint run_config.json (stat file names / keys)")
    parser.add_argument("--out-dir", required=True, help="<external_models>/stats")
    parser.add_argument("--tasks", nargs="*", default=list(OMEGA_FT_TASKS))
    args = parser.parse_args()

    field = json.loads(Path(args.run_config).read_text())["data"]["transform"]["field"]
    actions, states, per_task, skipped = [], [], {}, []
    for task in args.tasks:
        files = sorted((Path(args.home_root) / task).glob("*/state_action.hdf5"))
        per_task[task] = 0
        for path in files:
            try:
                a, s = episode_arrays(path)
            except (KeyError, OSError) as exc:
                skipped.append(f"{path}: {exc}")
                continue
            actions.append(a.astype(np.float32))
            states.append(s.astype(np.float32))
            per_task[task] += 1
    if not actions:
        raise SystemExit(f"no usable state_action.hdf5 under {args.home_root} for {args.tasks}")
    action, state = np.concatenate(actions), np.concatenate(states)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, key, values in (
        (Path(field["stat_path"]).name, field["stat_action_key"], action),
        (Path(field["state_stat_path"]).name, field["stat_state_key"], state),
    ):
        (out / name).write_text(json.dumps({key: stats(values)}, indent=1))
        print(f"wrote {out / name} [{key}] from {values.shape}", flush=True)
    summary = {"episodes_per_task": per_task, "rows": int(len(action)), "skipped": skipped,
               "action_norm_type": field["action_norm_type"], "state_norm_type": field["state_norm_type"]}
    (out / "omega0_home_stats_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
