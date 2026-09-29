# Derived from the original project; attribution is retained in LICENSE and NOTICE.md.
import logging
import random
from .common import context_indices, is_validation_episode
import numpy as np
import h5py
from pathlib import Path
from decord import VideoReader, cpu

class VLMPretrainDataset:
    """Whole-body WAM Dataset Loader for SMPL motion and Video"""

    def __init__(self, data_root, upsample_rate=3, val=False, chunk_size=1, img_history_size=1, use_delta_actions=True, data_downsample=1):
        self.DATASET_NAME = 'wam_smpl'
        self.data_root = Path(data_root)
        self.anno_dir = self.data_root / 'annotation_smpl'
        self.video_dir = self.data_root / 'video_smpl'
        self.upsample_rate = upsample_rate
        self.val = val
        self.use_delta_actions = use_delta_actions
        self.chunk_size = chunk_size
        self.img_history_size = img_history_size
        self.data_downsample = data_downsample
        self.data_files = self._load_file_list()
        split_name = 'test' if self.val else 'train'
        logging.getLogger(__name__).info('Loaded %d %s SMPL episodes', len(self.data_files), split_name)

    def get_dataset_name(self):
        return self.DATASET_NAME

    def _load_file_list(self):
        data_files = []
        for hdf5_path in sorted(self.anno_dir.glob('*.hdf5')):
            file_stem = hdf5_path.stem
            parts = file_stem.split('_')
            dataset_source = parts[0] if len(parts) > 0 else 'unknown'
            mp4_path = self.video_dir / f'{file_stem}.mp4'
            if hdf5_path.exists() and mp4_path.exists():
                data_files.append({'hdf5': hdf5_path, 'mp4': mp4_path, 'dataset': dataset_source, 'file_index': file_stem})
            else:
                logging.getLogger(__name__).warning('Missing paired video for %s', hdf5_path)
        if self.val:
            data_files = [f for (i, f) in enumerate(data_files) if is_validation_episode(f['file_index'], 10)]
        else:
            data_files = [f for (i, f) in enumerate(data_files) if not is_validation_episode(f['file_index'], 10)]
        return data_files[::self.data_downsample]

    def parse_img_data(self, mp4_path, idx, history_size=None, return_valid_mask=False):
        """
        Args:
            mp4_path: video path
            idx: current frame index
            history_size: number of frames to return (defaults to self.img_history_size)
            return_valid_mask: whether to return history validity mask
        Returns:
            frames: (T, H, W, 3) RGB
            valid_mask(optional): (T,), 1 for real frame, 0 for left-padded frame
        """
        history_size = int(self.img_history_size if history_size is None else history_size)
        history_size = max(1, history_size)
        try:
            vr = VideoReader(str(mp4_path), ctx=cpu(0), num_threads=1)
        except Exception:
            raise ValueError(f'Video decode failed: {mp4_path}')
        total_frames = len(vr)
        if total_frames <= 0:
            raise ValueError(f'Empty video: {mp4_path}')
        cur_idx = max(0, min(int(idx), total_frames - 1))
        frame_indices, valid_mask_np = context_indices(
            history_size - 1, self.upsample_rate, cur_idx, total_frames)
        frames = vr.get_batch(frame_indices).asnumpy()
        if return_valid_mask:
            return (frames, valid_mask_np)
        return frames

    def get_item(self, idx=None):
        if idx is None:
            idx = random.randint(0, len(self.data_files) - 1)
        file_info = self.data_files[idx % len(self.data_files)]
        with h5py.File(file_info['hdf5'], 'r') as f:
            total_frames = f['motion'].shape[0]
            max_index = total_frames - self.chunk_size * self.upsample_rate - 1
            if max_index < 0:
                raise ValueError(f"Not enough frames for the configured action horizon: {file_info['hdf5']}")
            index = random.randint(0, max_index)
            instruction = f['instruction'][()]
            if isinstance(instruction, bytes):
                instruction = instruction.decode('utf-8')
            view_flag = f['view'][()]
            view = 'first' if view_flag == b'first' else 'third'
            current_state = f['motion'][()][:, :-10][index:index + 1].astype(np.float32)
            chunk_size = self.chunk_size + 1 if self.use_delta_actions else self.chunk_size
            action_end = min(index + chunk_size * self.upsample_rate, max_index + 1)
            action_indices = list(range(index if self.use_delta_actions else index + self.upsample_rate, action_end, self.upsample_rate))
            while len(action_indices) < chunk_size:
                action_indices.append(action_indices[-1] if action_indices else index + self.upsample_rate)
            actions = f['motion'][()][:, :-10][action_indices[:chunk_size]].astype(np.float32)
            if self.use_delta_actions:
                actions = actions[1:] - actions[:-1]
        (context_frames, context_valid) = self.parse_img_data(file_info['mp4'], index, history_size=self.img_history_size, return_valid_mask=True)
        current_frame = context_frames[-1:]
        next_index = min(index + self.upsample_rate, total_frames - 1)
        next_image_frames = self.parse_img_data(file_info['mp4'], next_index, history_size=1)
        assert actions.shape[0] == self.chunk_size
        result = {'states': current_state, 'actions': actions, 'observations': current_frame, 'next_observations': next_image_frames, 'instruction': instruction.strip(), 'view': view, 'dataset': file_info['dataset'], 'current_images': current_frame, 'context_images': context_frames, 'context_valid': context_valid, 'file_info': {'hdf5_path': str(file_info['hdf5']), 'mp4_path': str(file_info['mp4']), 'frame_index': index}}
        return result

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = self.get_item(idx)
        return data
