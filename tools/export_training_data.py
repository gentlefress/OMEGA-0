"""Convert recorded Sonic episodes into the retained latent-training dataset schema.

Resamples the latest complete image/state/action at a fixed cadence, like WAM's
recorder. Controller latents must be recorded; SMPL references are never used as
substitutes. The export contains annotation/*.hdf5 and first/*.mp4.
"""
import argparse
from itertools import groupby
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from omega_real.recording.episode import EpisodeReader, VideoWriter
from omega_real.sensors.camera import prepare_rgb


DEFAULT_STREAMS = {"image": "camera.rgb", "latent": "robot.latent", "joints": "robot.joints",
    "orientation": "robot.orientation", "angular_velocity": "robot.angular_velocity",
    "linear_acceleration": "robot.linear_acceleration", "left_hand": "robot.left_hand",
    "right_hand": "robot.right_hand", "triggers": "operator.triggers", "reference": "teleop.command.action"}
ACTION_STREAMS = {**{key: value for key, value in DEFAULT_STREAMS.items() if key not in {"triggers", "reference"}},
    **{"details." + name: "teleop.details." + name for name in
       ("smpl_pose", "body_quat_w", "left_trigger", "right_trigger")}}


def inverse_yaw(rotation):
    matrix = rotation.as_matrix()
    return Rotation.from_euler("z", -np.arctan2(matrix[1, 0], matrix[0, 0]))


def export_episode(episode, destination, *, instruction=None, fps=30, streams=None, hand_scale=None):
    fps = int(fps)
    if fps <= 0:
        raise ValueError("Training FPS must be positive")
    episode, destination = Path(episode), Path(destination)
    stem = episode.name
    annotation, video = destination / "annotation" / (stem + ".hdf5"), destination / "first" / (stem + ".mp4")
    index_path = destination / "alignment" / (stem + ".json")
    if any(path.exists() for path in (annotation, video, index_path)):
        raise FileExistsError(f"Training episode already exists: {stem}")
    destination.mkdir(parents=True, exist_ok=True)
    with EpisodeReader(episode) as reader, tempfile.TemporaryDirectory(prefix=".export-", dir=destination) as temp:
        metadata = reader.manifest.get("metadata", {})
        if streams is None:
            # Recorder port names describe training roles; its actual streams may
            # use discovered wire names such as g1_debug.body_q.
            if "recording_inputs" in metadata:
                candidates = [metadata["recording_inputs"]]
            else:
                # Compatibility for episodes recorded before port metadata.
                candidates = [module.get("inputs", {}) for module in metadata.get("runtime_config", {}).get("modules", [])
                              if module["type"] in {"omega_real.recording.recorder.Recorder",
                                                    "omega_real.recording.modules.Recorder"}]
            candidates = [inputs for inputs in candidates
                          if any(schema.keys() <= inputs.keys() for schema in (ACTION_STREAMS, DEFAULT_STREAMS))]
            if len(candidates) > 1:
                raise ValueError("Multiple training recorders; provide an explicit streams mapping")
            if candidates:
                schema = ACTION_STREAMS if ACTION_STREAMS.keys() <= candidates[0].keys() else DEFAULT_STREAMS
                streams = {key: candidates[0][key] for key in schema}
            else:
                streams = ACTION_STREAMS if set(ACTION_STREAMS.values()) <= reader.manifest["streams"].keys() else DEFAULT_STREAMS
        streams = dict(streams)
        if set(streams) not in (set(DEFAULT_STREAMS), set(ACTION_STREAMS)) or len(set(streams.values())) != len(streams):
            raise ValueError("Provide a distinct recorded stream for every training field")
        missing = set(streams.values()) - reader.manifest["streams"].keys()
        if missing:
            raise ValueError(f"Missing recorded training streams: {sorted(missing)}. Enable controller latent and hand/IMU telemetry before collection.")
        instruction = instruction or metadata.get("instruction")
        if not instruction:
            raise ValueError("An instruction is required in episode metadata or --instruction")
        hand_scale = float(hand_scale if hand_scale is not None else metadata.get("state_hand_scale", 1.))
        image_options = {"crop": metadata["training_image_crop"]} if "training_image_crop" in metadata else {}
        if not np.isfinite(hand_scale) or hand_scale <= 0:
            raise ValueError("state_hand_scale must be positive (0.001 for original Inspire raw units)")
        reverse = {stream: name for name, stream in streams.items()}
        latest, rows, data = {}, [], {name: [] for name in ("motion", "latent", "state")}
        next_tick = first_tick = None
        invalidated_at = None
        yaw_state = yaw_motion = None
        writer = VideoWriter(Path(temp) / "video.mp4", fps=fps)
        sampled = reader.manifest.get("sampling") == "latest"

        def append_tick(tick):
            nonlocal yaw_state, yaw_motion
            def vector(name, width):
                array = np.asarray(latest[name].value, np.float32).reshape(-1)
                if array.shape != (width,) or not np.isfinite(array).all():
                    raise ValueError(f"{name} must contain {width} finite values")
                return array
            latent, joints = vector("latent", 64), vector("joints", 29)
            orientation = Rotation.from_quat(vector("orientation", 4)[[1, 2, 3, 0]])
            if "reference" in latest:
                motion = vector("reference", 71).copy()
                from omega_real.transforms.geometry import rot6d_to_matrix
                root = Rotation.from_matrix(rot6d_to_matrix(motion[:6]))
                triggers = vector("triggers", 2)
            else:
                root = Rotation.from_quat(vector("details.body_quat_w", 4)[[1, 2, 3, 0]]) * Rotation.from_quat([.5, .5, .5, .5])
                triggers = np.r_[vector("details.left_trigger", 1), vector("details.right_trigger", 1)]
                motion = np.r_[root.as_matrix()[:, :2].reshape(-1), vector("details.smpl_pose", 63), triggers].astype(np.float32)
            if yaw_state is None:
                yaw_state, yaw_motion = inverse_yaw(orientation), inverse_yaw(root)
            state_rotation = (yaw_state * orientation).as_matrix()[:, :2].reshape(-1)
            motion[:6] = (yaw_motion * root).as_matrix()[:, :2].reshape(-1)
            motion[-2:] = triggers
            hands = []
            for side in ("left_hand", "right_hand"):
                hand = np.asarray(latest[side].value, np.float32).reshape(-1)
                if len(hand) < 5 or not np.isfinite(hand).all():
                    raise ValueError("Measured hands need at least five finite values")
                hands.append(hand[:5] * hand_scale)
            state = np.concatenate([joints, *hands, state_rotation, vector("angular_velocity", 3), vector("linear_acceleration", 3)])
            data["motion"].append(motion)
            data["latent"].append(np.concatenate([latent, triggers]))
            data["state"].append(state)
            writer.write(prepare_rgb(latest["image"].value, image_options))
            rows.append({"received_ns": tick, "sequences": {name: sample.sequence for name, sample in latest.items()},
                         "sample_received_ns": {name: sample.received_ns for name, sample in latest.items()}})

        try:
            for stamp, samples in groupby(reader, key=lambda sample: sample.received_ns):
                while not sampled and next_tick is not None and next_tick < stamp and invalidated_at is None:
                    append_tick(next_tick)
                    next_tick = first_tick + round(len(rows) * 1e9 / fps)
                for sample in samples:
                    if sample.stream not in reverse:
                        continue
                    name = reverse[sample.stream]
                    if sample.valid:
                        latest[name] = sample
                    else:
                        latest.pop(name, None)
                        if next_tick is not None and invalidated_at is None:
                            invalidated_at = stamp
                if invalidated_at is not None and set(latest) == set(streams):
                    raise ValueError("Training stream invalidated and resumed mid-episode; split the recording explicitly")
                if next_tick is None and set(latest) == set(streams):
                    next_tick = first_tick = stamp
                if sampled and set(latest) == set(streams) and invalidated_at is None:
                    append_tick(stamp)
                elif invalidated_at is None and next_tick is not None and next_tick == stamp:
                    append_tick(next_tick)
                    next_tick = first_tick + round(len(rows) * 1e9 / fps)
        finally:
            writer.close()
        if not rows:
            raise ValueError("Episode never contained a complete image/state/latent/reference")
        with h5py.File(Path(temp) / "annotation.hdf5", "w") as output:
            for name, values in data.items():
                output[name] = np.stack(values).astype(np.float32)
            output["instruction"], output["view"] = str(instruction), "first"
            output["received_ns"] = np.asarray([row["received_ns"] for row in rows], np.int64)
            output.attrs["fps"] = fps
            output.attrs["state_hand_scale"] = hand_scale
            output.attrs["source_episode"] = str(episode.resolve())
        for path in (annotation, video, index_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps({"source_episode": str(episode), "fps": fps,
            "selection": "recorded snapshots" if sampled else "latest sample at each fixed-rate tick", "trailing_invalidation_ns": invalidated_at,
            "image_preparation": image_options,
            "streams": streams, "rows": rows}, indent=2) + "\n")
        os.replace(Path(temp) / "video.mp4", video)
        os.replace(Path(temp) / "annotation.hdf5", annotation)
    return {"annotation": str(annotation), "video": str(video), "frames": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instruction")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hand-scale", type=float)
    args = parser.parse_args()
    print(json.dumps(export_episode(args.episode, args.output, instruction=args.instruction, fps=args.fps, hand_scale=args.hand_scale), indent=2))


if __name__ == "__main__":
    main()
