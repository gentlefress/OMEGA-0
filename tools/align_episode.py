"""Export an explicit receive-time alignment index and arrays for offline learning."""
import argparse
import bisect
import json
from pathlib import Path
import numpy as np
from omega_real.recording.episode import EpisodeReader


def align(episode, output, *, image_stream, action_stream, state_streams=(), max_skew_ms=100):
    streams = {name: [] for name in (image_stream, action_stream, *state_streams)}
    with EpisodeReader(episode) as reader:
        metadata = reader.manifest
        for sample in reader:
            if sample.stream in streams:
                streams[sample.stream].append(sample)
    times = {name: [s.received_ns for s in values] for name, values in streams.items()}
    rows, images, actions, states = [], [], [], []
    for image in streams[image_stream]:
        if not image.valid:
            continue
        selected = {}
        for name in (action_stream, *state_streams):
            index = bisect.bisect_right(times[name], image.received_ns) - 1
            if index < 0:
                break
            sample = streams[name][index]
            if not sample.valid or image.received_ns - sample.received_ns > max_skew_ms * 1e6:
                break
            selected[name] = sample
        if len(selected) != 1 + len(state_streams):
            continue
        sample = selected[action_stream]
        if sample.command is not None and (sample.command.stop or sample.command.valid_until_ns <= image.received_ns):
            continue
        action = sample.value
        images.append(image.value)
        actions.append(np.asarray(action, np.float32).reshape(-1))
        if state_streams:
            states.append(np.concatenate([np.asarray(selected[name].value, np.float32).reshape(-1) for name in state_streams]))
        rows.append({"received_ns": image.received_ns, "image_sequence": image.sequence,
                     "selected_sequences": {name: sample.sequence for name, sample in selected.items()},
                     "skew_ns": {name: image.received_ns - sample.received_ns for name, sample in selected.items()}})
    if not rows:
        raise ValueError("No valid aligned rows; check streams and skew tolerance")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    np.savez_compressed(output, images=np.stack(images), actions=np.stack(actions),
                        states=np.stack(states) if states else np.empty((len(rows), 0), np.float32),
                        received_ns=np.asarray([row["received_ns"] for row in rows], np.int64))
    output.with_suffix(".json").write_text(json.dumps({"schema_version": 1, "source_episode": str(episode),
        "source_manifest": metadata, "basis": "host_receive_time", "image_stream": image_stream,
        "action_stream": action_stream, "state_streams": list(state_streams), "rows": rows}, indent=2) + "\n")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-stream", required=True)
    parser.add_argument("--action-stream", required=True)
    parser.add_argument("--state-stream", action="append", default=[])
    parser.add_argument("--max-skew-ms", type=float, default=100)
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    print(align(args.episode, args.output, image_stream=args.image_stream, action_stream=args.action_stream,
                state_streams=args.state_stream, max_skew_ms=args.max_skew_ms))


if __name__ == "__main__":
    main()
