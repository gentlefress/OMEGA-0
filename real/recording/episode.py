"""Row-aligned state_action.hdf5 / MP4 episodes and automatic format reading."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from datetime import datetime
from fractions import Fraction
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import numpy as np

from ..core.protocol import Sample, TensorSpec
from .images import png_array, image_directories, sequence_length
from .tensor_episode import (EpisodeReader as TensorEpisodeReader,
                             EpisodeWriter as TensorEpisodeWriter,
                             VideoWriter, hdf5_paths, video_files)


class EpisodeWriter(TensorEpisodeWriter):
    """Write complete rows to native datasets and one frame to each video.

    ``hdf5`` maps dataset paths to input names; unmapped numeric inputs use their
    stream names with dots replaced by slashes. ``videos`` maps MP4 names to
    image inputs. ``images`` maps directories to per-frame PNG inputs.
    Floating depth is quantized to uint16 millimeters. The additional _streams HDF5 group and _recording session
    metadata retain provenance without changing the action/state/sonic arrays.
    """
    data_filename = "state_action.hdf5"

    def __init__(self, root, *, metadata=None, videos=None, fps=30, hdf5=None, images=None):
        fps = Fraction(str(fps))
        if fps <= 0:
            raise ValueError("Recording FPS must be positive")
        self.created_at = time.time()
        self.row_count = 0
        self.failed = False
        self.row_fields = None
        super().__init__(root, metadata=metadata, videos=videos, fps=fps, hdf5=hdf5, images=images)
        self.file.attrs['created_at'] = self.created_at
        self.manifest.update(format='state_action', schema_version=3, sampling='latest', fps=float(fps), rows=0)
        self.file.attrs['schema_version'] = 3
        self._write_manifest()

    def _write_manifest(self):
        created = datetime.fromtimestamp(self.created_at)
        metadata = dict(self.manifest['metadata'])
        metadata.setdefault('session_name', self.path.name.removesuffix('.inprogress'))
        metadata.setdefault('created_at', created.strftime('%Y%m%d_%H%M%S'))
        metadata.setdefault('recording_started_at', created.strftime('%Y-%m-%d %H:%M:%S'))
        metadata['_recording'] = self.manifest
        temporary = self.path / 'session_meta.json.tmp'
        temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
        os.replace(temporary, self.path / 'session_meta.json')

    def append(self, sample, **kwargs):
        raise TypeError("State/action episodes require append_row(); use tensor_episode.EpisodeWriter for individual samples")

    def append_row(self, values, *, timestamp_ns=None, held=()):
        """Append input-name -> tensor/Sample mapping, validating the entire row first."""
        if self.closed or self.failed:
            raise RuntimeError("Episode is closed or has a failed row")
        if not isinstance(values, Mapping) or not values:
            raise ValueError("A recording row must be a nonempty mapping")
        if self.row_fields is not None and set(values) != self.row_fields:
            raise ValueError("Recording row fields changed")
        declared = set(self.dataset_paths) | set(self.video_names) | set(self.image_names)
        if not declared <= values.keys():
            raise ValueError("Recording row is missing configured inputs")
        if timestamp_ns is None:
            timestamp_ns = round(self.row_count * 1_000_000_000 / self.fps)
        if type(timestamp_ns) is not int or timestamp_ns < 0:
            raise ValueError("Row timestamp must be a nonnegative integer")
        if self.row_count and timestamp_ns < self.last_timestamp:
            raise ValueError("Row timestamps must be nondecreasing")
        held = set(held)
        if not held <= values.keys():
            raise ValueError("Held fields must name row inputs")
        samples = {}
        paths = dict(self.dataset_paths)
        for name, value in values.items():
            sample = value if isinstance(value, Sample) else Sample(name, self.row_count+1, timestamp_ns, np.asarray(value))
            if sample.stream != name or not sample.valid:
                raise ValueError("Rows require valid samples matching their input names")
            array = sample.value
            if name in self.image_names:
                writer = self.images.get(name)
                sample = replace(sample, value=writer.validate(array) if writer else png_array(array))
            elif name in self.video_names:
                if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
                    raise ValueError("Video requires uint8 HWC RGB images")
                video = self.videos.get(name)
                if video and array.shape[:2] != (video.stream.height, video.stream.width):
                    raise ValueError("Image size changed during episode")
            else:
                paths.setdefault(name, name.replace('.', '/'))
                dataset = self.datasets.get(name)
                if dataset is not None and (dataset.shape[1:] != array.shape or dataset.dtype != array.dtype):
                    raise ValueError(f"HDF5 tensor layout changed for {paths[name]}")
            samples[name] = sample
        if len(set(paths.values())) != len(paths):
            raise ValueError("Recording dataset paths collide")
        layout = hdf5_paths({path:name for name,path in paths.items()})
        self.dataset_paths = paths
        self.manifest['hdf5'] = layout
        self.row_fields = set(samples)
        try:
            for name, sample in samples.items():
                super().append(sample, recorded_ns=timestamp_ns, recorded_sequence=self.row_count+1,
                               held=name in held)
        except Exception:
            self.failed = True
            raise
        self.row_count += 1
        self.last_timestamp = timestamp_ns
        self.manifest['rows'] = self.row_count

    def close(self, *, complete=True, gaps=None, reason=''):
        return super().close(complete=complete and not self.failed, gaps=gaps,
                             reason=reason or ('Row write failed' if self.failed else ''))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close(complete=exc is None, reason=str(exc) if exc else '')


class EpisodeReader(TensorEpisodeReader):
    """Open original state/action episodes, new recordings, or tensor versions 2/3.

    Original datasets are exposed by their HDF5 paths (e.g. action/smpl_pose),
    videos by their filename stem (ego/exo), PNG sequences by directory name
    (exo_depth). Their row cadence is taken from
    video FPS; source timestamps, where available, remain separate Unix clocks.
    """
    def __init__(self, path, *, allow_incomplete=False, fps=30):
        self.path = Path(path)
        self.original = False
        if (self.path/'state_action.hdf5').exists():
            self.data_filename = 'state_action.hdf5'
            self.session_metadata = json.loads((self.path/'session_meta.json').read_text())
            if '_recording' not in self.session_metadata:
                self.original = True
                self._open_original(allow_incomplete, fps)
                return
        super().__init__(path, allow_incomplete=allow_incomplete)
        self.session_metadata = getattr(self, 'session_metadata', self.manifest.get('metadata', {}))
        self.row_count = self.manifest.get('rows')
        if self.data_filename == 'state_action.hdf5':
            counts = [len(self.file[entry['group']]['received_ns']) for entry in self.manifest['streams'].values()]
            if type(self.row_count) is not int or self.row_count < 0:
                self.close()
                raise ValueError("Invalid recorded row count")
            if any(count != self.row_count for count in counts):
                if not allow_incomplete:
                    self.close()
                    raise ValueError("Incomplete episode: stream row counts differ")
                self.manifest['status'] = 'incomplete'
            self.row_count = min([self.row_count, *counts, *self.image_row_limits.values()])
            # A failed row may have reached some datasets but not all videos.
            self.index = [entry for entry in self.index if entry[2] < self.row_count]
            self.manifest['rows'] = self.row_count
        self._rows = None

    def _read_manifest(self):
        if self.data_filename == 'state_action.hdf5':
            return self.session_metadata['_recording']
        return super()._read_manifest()

    @property
    def metadata(self):
        return {key:value for key,value in self.session_metadata.items() if key != '_recording'}

    def _open_original(self, allow_incomplete, fps):
        import h5py
        self.file = h5py.File(self.path/'state_action.hdf5', 'r')
        self.video_readers = {}
        self._array_datasets = {}
        self._array_cache = OrderedDict()
        self._array_cache_bytes = 0
        self._array_cache_limit = 16 * 1024 * 1024
        try:
            streams, counts, rates = {}, [], []
            image_gaps = False
            def visit(name, dataset):
                if not isinstance(dataset, h5py.Dataset):
                    return
                if dataset.ndim < 1:
                    raise ValueError(f"Episode dataset must have a row dimension: {name}")
                TensorSpec(dataset.dtype.name, dataset.shape[1:])
                self._array_datasets[name] = dataset
                streams[name] = {'dataset':name, 'count':len(dataset)}
                counts.append(len(dataset))
            self.file.visititems(visit)
            for path in sorted(self.path.glob('*.mp4')):
                import av
                with av.open(str(path)) as container:
                    video = container.streams.video[0]
                    count = video.frames or sum(1 for _ in container.decode(video=0))
                    if video.average_rate:
                        rates.append(Fraction(video.average_rate))
                if path.stem in streams:
                    raise ValueError(f"Video and tensor names collide: {path.stem}")
                streams[path.stem] = {'video':path.name, 'count':count}
                counts.append(count)
            for directory in sorted(self.path.iterdir()):
                if not directory.is_dir() or not any(directory.glob('frame_*.png')):
                    continue  # Optional depth directories can be absent or empty.
                image_directories({directory.name: directory.name})
                if directory.name in streams:
                    raise ValueError(f'Image and tensor/video names collide: {directory.name}')
                count, gaps = sequence_length(directory)
                image_gaps |= gaps
                streams[directory.name] = {'image_dir': directory.name, 'count': count}
                counts.append(count)
            if rates and any(rate != rates[0] for rate in rates):
                raise ValueError("Episode videos have different frame rates")
            self.fps = rates[0] if rates else Fraction(str(self.session_metadata.get('fps', fps)))
            if self.fps <= 0:
                raise ValueError("Episode FPS must be positive")
            incomplete = self.path.name.endswith('.inprogress') or len(set(counts)) > 1 or image_gaps
            if incomplete and not allow_incomplete:
                raise ValueError("Incomplete episode: tensor/media row counts differ, PNG frames have gaps, or recording is in progress")
            self.row_count = min(counts) if counts else 0
            self.manifest = {'format':'state_action', 'sampling':'latest', 'streams':streams,
                'status':'incomplete' if incomplete else 'complete', 'metadata':self.metadata,
                'fps':float(self.fps), 'rows':self.row_count, 'events':[], 'gaps':{},
                'timing_basis':'video_frame_index' if rates else 'configured_frame_rate'}
            if incomplete:
                self.manifest['gaps'] = {name:entry['count']-self.row_count for name,entry in streams.items()
                                         if entry['count'] != self.row_count}
            self.index = [(round(row*1_000_000_000/self.fps), name, row)
                          for row in range(self.row_count) for name in streams]
            self._rows = None
        except Exception:
            self.close()
            raise

    def publication_groups(self):
        if not self.original:
            return super().publication_groups()
        width = len(self.manifest['streams'])
        return tuple(tuple(range(row*width, (row+1)*width)) for row in range(self.row_count))

    def close(self):
        try:
            super().close()
        finally:
            if self.original:
                self._array_cache.clear()
                self._array_datasets.clear()
                self._array_cache_bytes = 0

    def _array(self, name, row):
        # h5py serializes calls across files/threads. Read short numeric blocks
        # so replay does not compete with the recorder for every field/frame.
        cached = self._array_cache.get(name)
        if cached is not None:
            start, values = cached
            if start <= row < start + len(values):
                self._array_cache.move_to_end(name)
                return values[row - start]
            self._array_cache_bytes -= values.nbytes
            del self._array_cache[name]
        dataset = self._array_datasets[name]
        row_bytes = dataset.dtype.itemsize * int(np.prod(dataset.shape[1:]))
        if row_bytes > self._array_cache_limit:
            return dataset[row]
        share = self._array_cache_limit // len(self._array_datasets)
        count = max(1, min(64, share // max(1, row_bytes)))
        start = row // count * count
        values = dataset[start:min(start + count, self.row_count)]
        while self._array_cache_bytes + values.nbytes > self._array_cache_limit:
            _, (_, old) = self._array_cache.popitem(last=False)
            self._array_cache_bytes -= old.nbytes
        self._array_cache[name] = (start, values)
        self._array_cache_bytes += values.nbytes
        return values[row - start]

    def sample(self, index):
        if not self.original:
            return super().sample(index)
        stamp, name, row = self.index[index]
        entry = self.manifest['streams'][name]
        value = (self._frame(name, row) if 'video' in entry else self._image(name, row)
                 if 'image_dir' in entry else np.asarray(self._array(entry['dataset'], row)))
        source_ns, clock = None, 'unknown'
        timestamp_path = {'action':'action/timestamp_realtime', 'sonic':'sonic/ros_timestamp'}.get(name.split('/')[0])
        if timestamp_path in self._array_datasets:
            seconds = np.asarray(self._array(timestamp_path, row)).reshape(-1)
            if seconds.size == 1 and np.isfinite(seconds[0]) and 0 <= seconds[0] <= np.iinfo(np.int64).max/1e9:
                source_ns, clock = round(float(seconds[0])*1e9), 'unix'
        return Sample(name, row+1, stamp, value, source_ns, clock, batch_id=f'row:{row}')

    def read_row(self, row, *, streams=None):
        """Read one recorded row as a mapping of stream names to NumPy tensors."""
        if not self.original and self.manifest.get('sampling') != 'latest':
            raise ValueError("Row access requires a sampled episode")
        if self._rows is None:
            self._rows = self.publication_groups()
        if type(row) is not int or not 0 <= row < len(self._rows):
            raise IndexError("Episode row out of range")
        selected = set(self.manifest['streams'] if streams is None else (streams,) if isinstance(streams, str) else streams)
        if not selected <= self.manifest['streams'].keys():
            raise KeyError("Unknown recorded stream")
        result = {}
        for index in self._rows[row]:
            # Tensor-format indices use metadata group names, so consult the
            # manifest before loading only the requested fields/videos.
            key = self.index[index][1]
            name = key if self.original else self.file[key].attrs['stream']
            if name in selected:
                sample = self.sample(index)
                result[name] = sample.value if sample.valid else None
        return result
