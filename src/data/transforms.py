"""Whole-body sample preparation for VLM pretraining and WAM finetuning.

Derived from the original project; attribution is retained in LICENSE and NOTICE.md.
"""
from __future__ import annotations
import copy
from typing import Any

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, Field
from torchvision.transforms import v2

from .augmentation import ResizeImage, CenterCrop, ColorJitter
from .utils import pad_to_len

IGNORE_INDEX = -100

class WholeBodyRepackTransform(BaseModel):
    dataset_name: str = 'wholebody'
    action_chunk_size: int | None = None
    pad_action_dim: int | None = None
    pad_state_dim: int | None = None

    def _fit_action_chunk(self, actions: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.action_chunk_size is None:
            return (actions, mask)
        target = int(self.action_chunk_size)
        if target <= 0:
            raise ValueError(f'action_chunk_size must be > 0, got {self.action_chunk_size}')
        t = int(actions.shape[0])
        if t == target:
            return (actions, mask)
        if t > target:
            return (actions[:target], mask[:target])
        pad_t = target - t
        if t > 0:
            pad_actions = np.repeat(actions[-1:], pad_t, axis=0)
        else:
            pad_actions = np.zeros((pad_t, *actions.shape[1:]), dtype=actions.dtype)
        pad_mask = np.zeros((pad_t, *mask.shape[1:]), dtype=mask.dtype)
        actions = np.concatenate([actions, pad_actions], axis=0)
        mask = np.concatenate([mask, pad_mask], axis=0)
        return (actions, mask)

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        actions = np.asarray(data['actions'], dtype=np.float32)
        if 'action_mask' in data:
            raw_mask = np.asarray(data['action_mask'], dtype=np.float32)
            if raw_mask.ndim == 1:
                raw_mask = np.repeat(raw_mask[:, None], actions.shape[1], axis=1)
            mask = raw_mask > 0.5
        else:
            mask = np.ones_like(actions, dtype=bool)
        if self.pad_action_dim is not None:
            (actions, _) = pad_to_len(actions, self.pad_action_dim)
            (mask, _) = pad_to_len(mask.astype(np.float32), self.pad_action_dim)
            mask = mask > 0.5
        (actions, mask) = self._fit_action_chunk(actions, mask)
        if 'states' in data and data['states'] is not None:
            (states, _) = pad_to_len(data['states'], self.pad_state_dim) if self.pad_state_dim is not None else (data['states'], None)
            state_valid = np.array([1.0], dtype=np.float32)
        else:
            state_dim = int(actions.shape[-1]) if len(actions.shape) > 1 else int(actions.shape[0])
            states = np.zeros((1, state_dim), dtype=np.float32)
            state_valid = np.array([0.0], dtype=np.float32)
        result = dict(observations=[Image.fromarray(img) for img in data['current_images']], states=states, actions=actions, instruction=data['instruction'].lower(), dataset=data.get('dataset', data.get('dataset_name', self.dataset_name)), actions_mask=mask, view=data['view'], view_token=data.get('view_token', int(data['view'] == 'first')), state_valid=state_valid)
        for key in ('text_input_ids', 'text_attention_mask'):
            if key in data:
                result[key] = data[key]
        if 'next_observations' in data:
            result['next_observations'] = [Image.fromarray(img) for img in data['next_observations']]
        if 'context_images' in data:
            result['context_observations'] = [Image.fromarray(img) for img in data['context_images']]
        if 'context_valid' in data:
            result['context_valid'] = np.array(data['context_valid'], dtype=np.float32)
        return result


class VLMModelTransform(BaseModel):
    resize: ResizeImage = Field(default_factory=lambda : ResizeImage(size=(270, 480)))
    color_jitter: ColorJitter
    img_aug: bool = True
    adaptive_resize: bool = False
    img_sizes: dict[str, Any] = Field(default_factory=lambda : {'wholebody': [270, 480], 'he': [240, 320]})

    def __call__(self, data: dict[str, Any], no_aug: bool=False, vlm_processor=None, action_tokenizer=None, **kwargs) -> dict[str, Any]:
        if vlm_processor is None or action_tokenizer is None:
            raise ValueError('VLM preprocessing requires a processor and action tokenizer')
        do_img_aug = False if no_aug else self.img_aug
        if self.adaptive_resize:
            assert data['dataset'] is not None
            match data['dataset']:
                case 'wholebody':
                    target_size = self.img_sizes['wholebody']
                case 'humanoid-everyday':
                    target_size = self.img_sizes['he']
                case _:
                    target_size = (256, 256)
            resizer = ResizeImage(size=tuple(target_size))()
        else:
            resizer = self.resize()
        t1 = v2.Compose([resizer, self.color_jitter() if do_img_aug else v2.Identity()])
        observations_raw = data['observations']
        if isinstance(observations_raw, np.ndarray) and observations_raw.ndim == 3:
            observations_raw = [observations_raw]
        images = [t1(img) for img in observations_raw]
        instruction = data['instruction']
        state = data['states']
        action = data['actions']
        view = data['view']
        (inputs, num_answer_tokens_list) = self.build_qwenvl_inputs(vlm_processor, action_tokenizer, [images], [instruction], [state], [action], [view])
        labels = copy.deepcopy(inputs['input_ids'])
        labels[:, :-(num_answer_tokens_list[0] + 2)] = IGNORE_INDEX
        inputs['labels'] = labels
        inputs['dataset_name'] = data.get('dataset', 'unknown')
        inputs['raw_actions'] = action
        inputs['raw_images'] = np.stack([np.array(img) for img in images])
        return inputs

    def build_qwenvl_inputs(self, vlm_processor, action_tokenizer, images, instructions, states, actions, views, **kwargs):
        """adapted from Qwen_VL_Interface.build_qwenvl_inputs"""
        messages = []
        num_answer_tokens_list = []
        assert len(images) == len(instructions), 'Images and instructions must have the same length'
        for (imgs, instruction, action, view) in zip(images, instructions, actions, views):
            tokenized_action = action_tokenizer(action)
            raw_action_tokens = vlm_processor.tokenizer(tokenized_action)['input_ids']
            num_answer_tokens = len(raw_action_tokens)
            num_answer_tokens_list.append(num_answer_tokens)
            view_token = '<EGO_VIEW>' if view == 'first' else '<EXO_VIEW>'
            content = [{'type': 'text', 'text': f'{view_token}\n'}]
            content.extend([{'type': 'image', 'image': img} for img in imgs])
            content.append({'type': 'text', 'text': instruction})
            user_msg = {'role': 'user', 'content': content}
            assistant_msg = {'role': 'assistant', 'content': [{'type': 'text', 'text': tokenized_action}]}
            messages.append([user_msg, assistant_msg])
        texts = [vlm_processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in messages]
        try:
            from qwen_vl_utils import process_vision_info
        except:
            raise ImportError('qwen_vl_utils not found, make sure to install it if using Qwen-VL model!')
        (image_inputs, video_inputs) = process_vision_info(messages, image_patch_size=16)
        inputs = vlm_processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt')
        return (inputs, num_answer_tokens_list)


class WAMModelTransform(BaseModel):
    include_state: bool = True
    resize: ResizeImage = Field(default_factory=lambda : ResizeImage(size=224))
    center_crop: CenterCrop = Field(default_factory=lambda : CenterCrop(size=224))
    jepa_resize: ResizeImage = Field(default_factory=lambda : ResizeImage(size=(256, 256)))
    jepa_center_crop: CenterCrop = Field(default_factory=lambda : CenterCrop(size=(256, 256)))
    color_jitter: ColorJitter = Field(default_factory=lambda : ColorJitter())
    img_aug: bool = False
    adaptive_resize: bool = False
    img_sizes: dict[str, Any] = Field(default_factory=lambda : {'egodex': [270, 480], 'he': [240, 320]})

    def __call__(self, data: dict[str, Any], vlm_processor=None, no_aug=False, **kwargs) -> dict[str, Any]:
        do_img_aug = False if no_aug else self.img_aug
        if self.adaptive_resize:
            assert data['dataset'] is not None
            match data['dataset']:
                case 'egodex':
                    target_size = self.img_sizes['egodex']
                case 'humanoid-everyday':
                    target_size = self.img_sizes['he']
                case _:
                    target_size = (256, 256)
            resizer = ResizeImage(size=tuple(target_size))()
            center_crop = CenterCrop(size=tuple(target_size))()
        else:
            resizer = self.resize()
            center_crop = self.center_crop()
        t1 = v2.Compose([resizer, center_crop, self.color_jitter() if do_img_aug else v2.Identity()])
        frame_to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        jepa_frame_transform = v2.Compose([self.jepa_resize(), self.jepa_center_crop()])
        observations_raw = data['observations']
        if isinstance(observations_raw, np.ndarray) and observations_raw.ndim == 3:
            observations_raw = [observations_raw]
        images = [t1(img) for img in observations_raw]
        context_images_raw = data.get('context_observations', observations_raw)
        context_images = [t1(img) for img in context_images_raw]
        jepa_context_images = [jepa_frame_transform(img) for img in context_images_raw]
        instruction = data['instruction']
        view = data['view']
        inputs: dict = self.build_qwenvl_inputs(vlm_processor, images, instruction, view)
        inputs['dataset_name'] = data.get('dataset', 'unknown')
        inputs['raw_actions'] = data.get('raw_actions', data['actions'])
        if 'actions_mask' in data:
            inputs['actions_mask'] = data['actions_mask']
        inputs['raw_images'] = images
        inputs['actions'] = data['actions']
        inputs['states'] = data['states'] if self.include_state else None
        inputs['view_token'] = data['view_token']
        inputs['instruction'] = data['instruction']
        inputs['text_input_ids'] = data['text_input_ids']
        inputs['text_attention_mask'] = data['text_attention_mask']
        if 'state_valid' in data:
            inputs['state_valid'] = data['state_valid']
        else:
            inputs['state_valid'] = np.array([1.0], dtype=np.float32)
        inputs['jepa_current_frame'] = frame_to_tensor(jepa_context_images[0])
        inputs['jepa_context_frames'] = torch.stack([frame_to_tensor(img) for img in jepa_context_images], dim=0)
        if 'context_valid' in data:
            inputs['jepa_context_valid'] = np.array(data['context_valid'], dtype=np.float32)
        else:
            inputs['jepa_context_valid'] = np.ones(len(context_images), dtype=np.float32)
        if 'next_observations' in data and len(data['next_observations']) > 0:
            next_img = jepa_frame_transform(data['next_observations'][1:])
            inputs['jepa_next_frame'] = frame_to_tensor(next_img)
        else:
            inputs['jepa_next_frame'] = inputs['jepa_current_frame'].clone()
        return inputs

    def build_qwenvl_inputs(self, vlm_processor, imgs, instruction, view, **kwargs) -> dict:
        from qwen_vl_utils import process_vision_info
        'adapted from Qwen_VL_Interface.build_qwenvl_inputs'
        messages = []
        view_token = '<EGO_VIEW>' if view == 'first' else '<EXO_VIEW>'
        content = [{'type': 'text', 'text': f'{view_token}\n'}]
        content.extend([{'type': 'image', 'image': img} for img in imgs])
        content.append({'type': 'text', 'text': instruction})
        user_msg = {'role': 'user', 'content': content}
        messages.append([user_msg])
        texts = [vlm_processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        (image_inputs, video_inputs) = process_vision_info(messages, image_patch_size=16)
        inputs = vlm_processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt')
        return inputs
