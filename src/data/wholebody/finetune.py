# Derived from the original project; attribution is retained in LICENSE and NOTICE.md.
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple
from transformers import T5Tokenizer
import h5py
import numpy as np
import torch
from . import common
import logging

log = logging.getLogger(__name__)

class FinetuneDataset(torch.utils.data.Dataset):
    """
    WholeBody post-training dataset with HE-style indexing/sampling:
      - deterministic global index -> (episode, frame)
      - future video frames paired with each action chunk
      - boundary padding masks for observations/actions
    """

    def __init__(self, data_root: str, split: str='train', action_chunk_size: int=30, upsample_rate: int=1, use_delta_actions: bool=False, data_downsample: int=1, annotation_dir: str | None=None, ego_video_dir: str | None=None, *, text_model='google-t5/t5-base', local_files_only=True) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.action_chunk_size = int(action_chunk_size)
        self.upsample_rate = int(upsample_rate)
        self.use_delta_actions = bool(use_delta_actions)
        self.data_downsample = int(data_downsample)
        if min(self.action_chunk_size, self.upsample_rate, self.data_downsample) < 1:
            raise ValueError("Chunk size, frame stride, and downsampling must be positive")
        self.dataset_name = 'wholebody'
        self.anno_dir = Path(annotation_dir) if annotation_dir is not None else self.data_root / 'annotation'
        self.ego_video_dir = Path(ego_video_dir) if ego_video_dir is not None else self.data_root / 'first'
        self.max_instruction_length = 128
        self.tokenizer = T5Tokenizer.from_pretrained(text_model, use_fast=False, local_files_only=local_files_only)
        self.episode_list, self.episodes_lens = self._build_episode_list(split=split)
        if self.data_downsample > 1:
            self.episode_list = self.episode_list[::self.data_downsample]
            self.episodes_lens = self.episodes_lens[::self.data_downsample]
        self.cumsum_lens = np.cumsum(self.episodes_lens) if self.episodes_lens else np.array([], dtype=np.int64)

    def _build_episode_list(self, split: str) -> Tuple[List[Dict[str, Any]], List[int]]:
        episode_list: List[Dict[str, Any]] = []
        episode_idx = 0
        for hdf5_path in sorted(self.anno_dir.glob('*.hdf5'), key=lambda p: p.name):
            file_stem = hdf5_path.stem
            ego_mp4_path = self.ego_video_dir / f'{file_stem}.mp4'
            if not ego_mp4_path.exists():
                log.warning("Skipping %s: missing ego video %s", hdf5_path, ego_mp4_path)
                continue
            total_frames = common.latent_length(hdf5_path)
            if total_frames <= 0:
                continue
            is_val = common.is_validation_episode(file_stem, 20)
            if split == 'train' and is_val:
                episode_idx += 1
                continue
            if split != 'train' and (not is_val):
                episode_idx += 1
                continue
            parts = file_stem.split('_')
            dataset_source = parts[0] if len(parts) > 0 else self.dataset_name
            episode_list.append({'episode_index': episode_idx, 'hdf5_path': str(hdf5_path), 'ego_mp4_path': str(ego_mp4_path), 'len_episode': total_frames, 'dataset': dataset_source, 'file_index': file_stem})
            episode_idx += 1
        episodes_lens = [int(ep['len_episode']) for ep in episode_list]
        log.info("Selected %d %s episodes (%d frames) from %s", len(episode_list), split, sum(episodes_lens), self.anno_dir)
        return (episode_list, episodes_lens)

    def __len__(self) -> int:
        return int(np.sum(self.episodes_lens)) if self.episodes_lens else 0

    def _locate_frame(self, idx: int) -> Tuple[int, int]:
        return common.locate_frame(self.cumsum_lens, idx)


    def _read_frames(self, mp4_path: str, frame_indices: List[int]) -> np.ndarray:
        return common.read_frames(mp4_path, frame_indices)

    def _build_context_indices_future(self, frame_id: int, length: int) -> Tuple[List[int], np.ndarray]:
        return common.future_indices(self.action_chunk_size, self.upsample_rate, frame_id, length)

    def _build_action_indices(self, frame_id: int, length: int) -> Tuple[List[int], np.ndarray]:
        return common.action_indices(self.action_chunk_size, self.upsample_rate, self.use_delta_actions, frame_id, length)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        (episode_id, frame_id) = self._locate_frame(idx)
        ep = self.episode_list[episode_id]
        length = int(ep['len_episode'])
        with h5py.File(ep['hdf5_path'], 'r') as f:
            motion = f['latent'][()].astype(np.float32)
            state = f['state'][()].astype(np.float32)
            instruction = f['instruction'][()]
            view_flag = f['view'][()]
        if isinstance(instruction, bytes):
            instruction = instruction.decode('utf-8')
        instruction = str(instruction).strip().lower()
        view = 'first' if view_flag == b'first' else 'third'
        if view == 'first':
            view_token = 1
        else:
            view_token = 0
        (context_indices, obs_mask) = self._build_context_indices_future(frame_id, length)
        (action_indices, action_mask) = self._build_action_indices(frame_id, length)
        ego_context_images = self._read_frames(ep['ego_mp4_path'], context_indices)
        # not including linear acceleration and angular velocity in the state for now
        states = state[frame_id:frame_id + 1][:, :-6]
        actions = motion[action_indices]
        initial_pose = actions[0:1].copy() if self.use_delta_actions else None
        if self.use_delta_actions:
            actions = actions[1:] - actions[:-1]
            action_mask = action_mask[1:]
        encoded = self.tokenizer(instruction, max_length=self.max_instruction_length, padding='max_length', truncation=True, return_tensors='pt')
        return {'initial_pose': initial_pose, 'states': states.astype(np.float32), 'actions': actions.astype(np.float32), 'action_mask': action_mask.astype(np.float32), 'observations': ego_context_images[0:1], 'current_images': ego_context_images[0:1], 'context_images': ego_context_images, 'context_valid': obs_mask.astype(np.float32), 'next_observations': ego_context_images, 'view_token': view_token, 'instruction': instruction, 'view': view, 'dataset': ep.get('dataset', self.dataset_name), 'dataset_name': self.dataset_name, 'episode_index': int(ep['episode_index']), 'frame_id': int(frame_id), 'obs_mask': obs_mask.astype(np.float32), 'text_input_ids': encoded['input_ids'], 'text_attention_mask': encoded['attention_mask'], 'file_info': {'hdf5_path': ep['hdf5_path'], 'ego_mp4_path': ep['ego_mp4_path'], 'frame_index': int(frame_id)}}
