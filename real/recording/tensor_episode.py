"""Versioned HDF5 episodes with optional video and explicit completion state."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import replace
import json
import os
import re
from pathlib import Path
import uuid

import numpy as np

from ..core.protocol import Sample, dumps, loads
from .images import ImageWriter, image_directories, read_image


def hdf5_paths(layout):
    """Flatten nested dataset paths; leaves name recorder inputs or streams."""
    result = {}

    def visit(node, prefix=""):
        if not isinstance(node, Mapping):
            raise ValueError("hdf5 must map dataset paths to input names or nested groups")
        for name, value in node.items():
            if not isinstance(name, str):
                raise ValueError("HDF5 path components must be strings")
            path = prefix + name.removeprefix("/")
            parts = path.split("/")
            if any(part in {"", ".", ".."} or "\x00" in part for part in parts) or parts[0] == "_streams":
                raise ValueError(f"Invalid or reserved HDF5 path: {path!r}")
            if isinstance(value, Mapping):
                if not value:
                    raise ValueError(f"HDF5 group {path!r} is empty")
                visit(value, path + "/")
            else:
                if not isinstance(value, str) or not value:
                    raise ValueError(f"HDF5 dataset {path!r} must name an input")
                if any(path == other or path.startswith(other + "/") or other.startswith(path + "/") for other in result):
                    raise ValueError(f"Conflicting HDF5 dataset path: {path!r}")
                if value in result.values():
                    raise ValueError(f"HDF5 input {value!r} is mapped more than once")
                result[path] = value

    visit({} if layout is None else layout)
    return result


def video_files(layout):
    """Map safe episode-local MP4 filenames to input names or streams."""
    if layout is None:
        return {}
    if not isinstance(layout, Mapping):
        raise ValueError("layout.videos must map video names to input names")
    result = {}
    for name, stream in layout.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"Invalid video name: {name!r}")
        filename = name if name.endswith(".mp4") else name + ".mp4"
        if filename in result:
            raise ValueError(f"Duplicate video filename: {filename}")
        if not isinstance(stream, str) or not stream:
            raise ValueError(f"Video {name!r} must name an input")
        if stream in result.values():
            raise ValueError(f"Video input {stream!r} is mapped more than once")
        result[filename] = stream
    return result


class VideoWriter:
    def __init__(self, path, fps=30):
        import av
        self.av = av
        self.container = av.open(str(path), mode="w")
        self.stream = None
        self.fps = fps
        self.count = 0

    def write(self, image):
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Video requires uint8 HWC RGB images")
        if self.stream is None:
            self.stream = self.container.add_stream("libx264", rate=self.fps)
            self.stream.height, self.stream.width = image.shape[:2]
            self.stream.pix_fmt = "yuv444p"
            self.stream.options = {"crf": "0", "preset": "ultrafast"}
        if image.shape[:2] != (self.stream.height, self.stream.width):
            raise ValueError("Image size changed during episode")
        frame = self.av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        index = self.count
        self.count += 1
        return index

    def close(self):
        if self.container is not None:
            if self.stream is not None:
                for packet in self.stream.encode():
                    self.container.mux(packet)
            self.container.close()
            self.container = None


class EpisodeWriter:
    data_filename = "samples.hdf5"

    def __init__(self, root, *, metadata=None, videos=None, fps=30, hdf5=None, images=None):
        import h5py
        layout = hdf5_paths(hdf5)
        video_layout = video_files(videos)
        image_layout = image_directories(images)
        if set(image_layout.values()) & (set(layout.values()) | set(video_layout.values())):
            raise ValueError("An input cannot appear in layout.images and another destination")
        if set(layout.values()) & set(video_layout.values()):
            raise ValueError("An input cannot appear in both layout.hdf5 and layout.videos")
        self.image_names = {stream: name for name, stream in image_layout.items()}
        self.images = {}
        self.video_names = {stream: name for name, stream in video_layout.items()}
        self.dataset_paths = {stream: path for path, stream in layout.items()}
        self.datasets = {}
        version = 3 if layout else 2
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / ("episode-" + uuid.uuid4().hex + ".inprogress")
        self.path.mkdir()
        self.file = h5py.File(self.path / self.data_filename, "w")
        self.file.attrs["schema_version"] = version
        self.groups = {}
        self.group_datasets = {}
        self.videos = {}
        self.fps = fps
        self.events = []
        self.last_sequences = {}
        self.sequence_gaps = {}
        self.manifest = {"schema_version": version, "status": "incomplete", "metadata": metadata or {}, "streams": {}, "gaps": {}, "events": []}
        if layout:
            self.manifest["hdf5"] = layout
        if video_layout:
            self.manifest["videos"] = video_layout
        if image_layout:
            self.manifest["images"] = image_layout
        self.closed = False
        self._write_manifest()

    def _write_manifest(self):
        temp = self.path / "manifest.json.tmp"
        temp.write_text(json.dumps(self.manifest, indent=2, allow_nan=False) + "\n")
        os.replace(temp, self.path / "manifest.json")

    def _group(self, sample):
        import h5py
        if sample.stream not in self.groups:
            key = hashlib.sha256(sample.stream.encode()).hexdigest()[:24]
            if self.dataset_paths or self.data_filename == "state_action.hdf5":
                key = "_streams/" + key
            group = self.file.create_group(key)
            group.attrs["stream"] = sample.stream
            for name in ("sequence", "received_ns", "source_ns", "video_index", "image_index", "observed_sequence", "observed_received_ns"):
                group.create_dataset(name, shape=(0,), maxshape=(None,), dtype="i8", chunks=True)
            group.create_dataset("payload", shape=(0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
            group.create_dataset("valid", shape=(0,), maxshape=(None,), dtype="bool", chunks=True)
            group.create_dataset("held", shape=(0,), maxshape=(None,), dtype="bool", chunks=True)
            group.create_dataset("batch_id", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(), chunks=True)
            self.groups[sample.stream] = group
            self.group_datasets[sample.stream] = dict(group.items())
            self.manifest["streams"][sample.stream] = {"group": key, "count": 0, "spec_id": sample.spec_id}
            if sample.stream in self.dataset_paths:
                self.manifest["streams"][sample.stream]["dataset"] = self.dataset_paths[sample.stream]
            if sample.stream in self.image_names:
                self.manifest["streams"][sample.stream]["image_dir"] = self.image_names[sample.stream]
            self.file.flush()
            self._write_manifest()
        return self.groups[sample.stream]

    def _tensor(self, sample, group, row):
        """Store mapped values once as native HDF5 arrays, retaining invalid rows."""
        path = self.dataset_paths[sample.stream]
        dataset = self.datasets.get(sample.stream)
        if sample.valid:
            value = sample.value
            if dataset is None:
                parent, _, name = path.rpartition("/")
                group_parent = self.file.require_group(parent) if parent else self.file
                dataset = group_parent.create_dataset(name, shape=(row + 1, *value.shape),
                    maxshape=(None, *value.shape), dtype=value.dtype, chunks=True)
                dataset.attrs["stream"] = sample.stream
                dataset.attrs["metadata"] = group.name
                dataset.attrs["kind"] = "tensor"
                self.datasets[sample.stream] = dataset
            elif dataset.shape[1:] != value.shape or dataset.dtype != value.dtype:
                raise ValueError(f"HDF5 tensor layout changed for {path}: expected {dataset.shape[1:]} {dataset.dtype}, got {value.shape} {value.dtype}")
        if dataset is not None:
            dataset.resize(row + 1, axis=0)
            if sample.valid:
                dataset[row] = sample.value

    def append(self, sample, *, recorded_ns=None, recorded_sequence=None, held=False):
        if self.closed:
            raise RuntimeError("Episode already closed")
        observed = sample
        if recorded_ns is not None:
            sample = replace(sample, received_ns=recorded_ns, sequence=recorded_sequence)
        group = self._group(sample)
        previous = self.last_sequences.get(sample.stream)
        if previous is not None and sample.sequence != previous + 1:
            self.sequence_gaps[sample.stream] = self.sequence_gaps.get(sample.stream, 0) + max(1, sample.sequence - previous - 1)
        self.last_sequences[sample.stream] = sample.sequence
        payload, video_index, image_index = sample, -1, -1
        if sample.valid and sample.stream in self.video_names:
            if sample.stream not in self.videos:
                name = self.video_names[sample.stream]
                self.videos[sample.stream] = VideoWriter(self.path / name, self.fps)
                self.manifest["streams"][sample.stream]["video"] = name
            video_index = self.videos[sample.stream].write(sample.value)
            payload = replace(sample, value=np.asarray(video_index, np.int64))
        if sample.valid and sample.stream in self.image_names:
            if sample.stream not in self.images:
                self.images[sample.stream] = ImageWriter(self.path / self.image_names[sample.stream])
            image_index = self.images[sample.stream].write(sample.value)
            payload = replace(sample, value=np.asarray(image_index, np.int64))
        fields = self.group_datasets[sample.stream]
        row = len(fields["received_ns"])
        if sample.stream in self.dataset_paths:
            self._tensor(payload, group, row)
            if payload.valid:
                payload = replace(payload, value=np.asarray(row, np.int64))
        for dataset in fields.values():
            dataset.resize((row + 1,))
        fields["sequence"][row] = sample.sequence
        fields["received_ns"][row] = sample.received_ns
        fields["source_ns"][row] = -1 if sample.source_ns is None else sample.source_ns
        fields["video_index"][row] = video_index
        fields["image_index"][row] = image_index
        fields["observed_sequence"][row] = observed.sequence
        fields["observed_received_ns"][row] = observed.received_ns
        fields["valid"][row] = sample.valid
        fields["held"][row] = held
        fields["batch_id"][row] = sample.batch_id
        fields["payload"][row] = np.frombuffer(dumps(payload), dtype=np.uint8)
        self.manifest["streams"][sample.stream]["count"] += 1
        if row % 30 == 0:
            self.file.flush()
            self._write_manifest()

    def event(self, kind, timestamp_ns, payload=None):
        self.manifest["events"].append({"kind": kind, "timestamp_ns": timestamp_ns, "payload": payload})

    def close(self, *, complete=True, gaps=None, reason=""):
        if self.closed:
            return self.path
        errors = []
        for video in self.videos.values():
            try:
                video.close()
            except Exception as error:
                errors.append(str(error))
        try:
            self.file.flush()
        finally:
            self.file.close()
        self.closed = True
        self.manifest["gaps"] = {name: max((gaps or {}).get(name, 0), self.sequence_gaps.get(name, 0)) for name in set(gaps or {}) | self.sequence_gaps.keys()}
        self.manifest["status"] = "complete" if complete and not errors and not any(self.manifest["gaps"].values()) else "incomplete"
        self.manifest["reason"] = reason or "; ".join(errors)
        self._write_manifest()
        if self.manifest["status"] == "complete":
            final = self.path.with_suffix("")
            self.path.rename(final)
            self.path = final
        if errors:
            raise RuntimeError("Video finalization failed: " + "; ".join(errors))
        return self.path


class EpisodeReader:
    data_filename = "samples.hdf5"

    def _read_manifest(self):
        return json.loads((self.path / "manifest.json").read_text())

    def __init__(self, path, *, allow_incomplete=False):
        import h5py
        self.path = Path(path)
        self.manifest = self._read_manifest()
        if self.manifest["schema_version"] not in (2, 3):
            raise ValueError("Unsupported episode schema; expected tensor-stream version 2 or 3")
        if self.manifest["status"] != "complete" and not allow_incomplete:
            raise ValueError("Incomplete episode; explicitly opt in to recovery")
        self.file = h5py.File(self.path / self.data_filename, "r")
        self.index = []
        for entry in self.manifest["streams"].values():
            name = entry["group"]
            group = self.file[name]
            self.index.extend((int(stamp), name, row) for row, stamp in enumerate(group["received_ns"][:]))
        self.index.sort()
        self.video_readers = {}
        self.missing_images = set()
        self.image_row_limits = {}
        try:
            for entry in self.manifest['streams'].values():
                if 'image_dir' not in entry:
                    continue
                image_directories({entry['image_dir']: 'image'})
                group = self.file[entry['group']]
                for row, index in enumerate(group['image_index'][:]):
                    if not group['valid'][row]:
                        continue
                    if index < 0 or not (self.path / entry['image_dir'] / f'frame_{index:06d}.png').is_file():
                        self.missing_images.add((entry['group'], row))
                        self.image_row_limits.setdefault(entry['group'], row)
            if self.missing_images:
                if not allow_incomplete:
                    raise ValueError('Incomplete episode: PNG frames are missing')
                self.manifest['status'] = 'incomplete'
        except Exception:
            self.close()
            raise

    def _image(self, stream, index):
        directory = self.manifest['streams'][stream]['image_dir']
        return read_image(self.path / directory / f'frame_{index:06d}.png')

    def _frame(self, stream, index):
        import av
        video = self.manifest["streams"][stream]["video"]
        cached = self.video_readers.get(video)
        if cached is None or index < cached[2]:
            if cached:
                cached[0].close()
            container = av.open(str(self.path / video))
            cached = [container, iter(container.decode(video=0)), -1, None]
            self.video_readers[video] = cached
        while cached[2] < index:
            try:
                cached[3] = next(cached[1]).to_ndarray(format="rgb24")
            except StopIteration as error:
                raise ValueError(f"Video {video} ended before frame {index}") from error
            cached[2] += 1
        return cached[3]

    def publication_groups(self):
        """Index atomic publications without decoding media/tensor datasets.

        Sampled recordings replay each complete recorded row atomically, even
        when its inputs originated in different source batches. Older files
        without batch IDs replay unidentified samples individually.
        """
        sampled = self.manifest.get("sampling") == "latest"
        groups, occurrences = {}, {}
        for index, (stamp, group_name, row) in enumerate(self.index):
            group = self.file[group_name]
            if sampled:
                key = (stamp, "snapshot", int(group["sequence"][row]))
            else:
                batch = (group['batch_id'].asstr()[row] if 'batch_id' in group
                         else loads(group['payload'][row].tobytes()).batch_id)
                key = (stamp, "batch", batch) if batch else (stamp, "single", index)
            occurrence = occurrences.get((key, group_name), 0)
            occurrences[key, group_name] = occurrence + 1
            groups.setdefault((key, occurrence), []).append(index)
        return tuple(tuple(indices) for indices in groups.values())

    def sample(self, index):
        _, group_name, row = self.index[index]
        group = self.file[group_name]
        sample = loads(group["payload"][row].tobytes())
        dataset = self.manifest["streams"][sample.stream].get("dataset")
        if dataset is not None and sample.valid:
            sample = replace(sample, value=np.asarray(self.file[dataset][row]))
        if (group_name, row) in self.missing_images:
            return replace(sample, value=None, valid=False)
        if sample.valid and 'image_dir' in self.manifest['streams'][sample.stream]:
            sample = replace(sample, value=self._image(sample.stream, int(group['image_index'][row])))
        video_index = int(group["video_index"][row])
        if video_index >= 0:
            image = self._frame(sample.stream, video_index)
            sample = replace(sample, value=image)
        return sample

    def __iter__(self):
        for index in range(len(self.index)):
            yield self.sample(index)

    def close(self):
        self.file.close()
        for container, *_ in self.video_readers.values():
            container.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
