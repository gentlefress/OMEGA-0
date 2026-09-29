# Derived from the original project; attribution is retained in LICENSE and NOTICE.md.
from __future__ import annotations
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

class PaddedCollatorForTogether:

    def __init__(self, model_max_length, pad_token_id, t5_pad_token_id, padding_side='right', pixel_values_dtype=torch.float32):
        self.model_max_length = model_max_length
        self.pad_token_id = pad_token_id
        self.t5_pad_token_id = t5_pad_token_id
        self.padding_side = padding_side
        self.pixel_values_dtype = pixel_values_dtype

    def __call__(self, instances):
        input_ids = [instance['input_ids'].squeeze(0) if instance['input_ids'].dim() == 2 else instance['input_ids'] for instance in instances]
        text_input_ids = [instance['text_input_ids'].squeeze(0) if instance['text_input_ids'].dim() == 2 else instance['text_input_ids'] for instance in instances]
        pixel_values = [instance['pixel_values'] for instance in instances]
        dataset_names = [instance['dataset_name'] for instance in instances] if 'dataset_name' in instances[0] else None
        has_ego = [instance['has_ego'] for instance in instances] if 'has_ego' in instances[0] else None
        has_exo = [instance['has_exo'] for instance in instances] if 'has_exo' in instances[0] else None
        assert self.padding_side == 'right', f'Invalid Tokenizer padding_side={self.padding_side}'
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        text_input_ids = pad_sequence(text_input_ids, batch_first=True, padding_value=self.t5_pad_token_id)
        input_ids = input_ids[:, :self.model_max_length]
        attention_mask = input_ids.ne(self.pad_token_id)
        text_input_ids = text_input_ids[:, :128]
        text_attention_mask = text_input_ids.ne(self.t5_pad_token_id)
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values).to(dtype=self.pixel_values_dtype)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {k: torch.stack([pv[k] for pv in pixel_values]) for k in pixel_values[0]}
        else:
            raise ValueError(f'Unsupported pixel_values type: {type(pixel_values[0])}')
        image_grid_thw = torch.stack([instance['image_grid_thw'].squeeze(0) for instance in instances])
        output = {'input_ids': input_ids, 'attention_mask': attention_mask, 'text_input_ids': text_input_ids, 'text_attention_mask': text_attention_mask, 'pixel_values': pixel_values, 'image_grid_thw': image_grid_thw}
        if dataset_names is not None:
            output['dataset_name'] = dataset_names
        if has_ego is not None:
            output['has_ego'] = has_ego
        if has_exo is not None:
            output['has_exo'] = has_exo
        states_present = [instance.get('states') is not None for instance in instances]
        if any(states_present) and not all(states_present):
            raise ValueError('A batch must consistently include or omit state conditioning')
        if all(states_present):
            output['states'] = torch.stack([torch.as_tensor(instance['states']) for instance in instances])
        for key in (
            'raw_actions', 'actions_mask', 'raw_images', 'view_token', 'actions',
            'state_valid', 'jepa_current_frame', 'jepa_context_frames',
            'jepa_context_valid', 'jepa_next_frame',
        ):
            if key in instances[0]:
                values = [instance[key] for instance in instances]
                output[key] = (np.stack(values) if key == 'raw_actions'
                               else torch.stack([torch.as_tensor(np.asarray(value)) if not torch.is_tensor(value)
                                                 else value for value in values]))
        return output


class PaddedCollatorForActionPrediction:
    """Right-pad token answers and concatenate Qwen's flattened image patches."""
    def __init__(self, model_max_length, pad_token_id, pixel_values_dtype=torch.float32):
        if model_max_length <= 0 or pad_token_id is None:
            raise ValueError("Pretraining requires a positive token limit and a pad token")
        self.model_max_length = model_max_length
        self.pad_token_id = pad_token_id
        self.pixel_values_dtype = pixel_values_dtype

    def __call__(self, instances):
        inputs = [instance['input_ids'].reshape(-1) for instance in instances]
        labels = [instance['labels'].reshape(-1) for instance in instances]
        if any(len(tokens) > self.model_max_length for tokens in inputs):
            raise ValueError("Pretraining sequence exceeds model_max_length; reduce image size or action horizon")
        if any(len(tokens) != len(answer) for tokens, answer in zip(inputs, labels)):
            raise ValueError("Pretraining input_ids and labels must have matching lengths")
        masks = [torch.ones_like(tokens, dtype=torch.bool) for tokens in inputs]
        return {
            'input_ids': pad_sequence(inputs, batch_first=True, padding_value=self.pad_token_id),
            'attention_mask': pad_sequence(masks, batch_first=True, padding_value=False),
            'labels': pad_sequence(labels, batch_first=True, padding_value=-100),
            'pixel_values': torch.cat([instance['pixel_values'] for instance in instances], dim=0).to(self.pixel_values_dtype),
            'image_grid_thw': torch.cat([instance['image_grid_thw'].reshape(-1, 3) for instance in instances], dim=0),
        }
