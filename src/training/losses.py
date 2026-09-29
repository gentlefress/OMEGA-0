# Derived from the original project; attribution is retained in LICENSE and NOTICE.md.
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

class FinetuneObjective:
    """Active one-video loss, independent of optimizer, logging, and distributed runtime."""

    def __init__(self, model_cfg, device, *, dataset_action_is_delta=False):
        self.model_cfg = model_cfg
        self.device = torch.device(device)
        self.ac_dim = model_cfg.action_dim
        self.rtc_enabled = model_cfg.rtc
        self.max_delay = model_cfg.max_delay
        if self.rtc_enabled and (not 0 < self.max_delay <= model_cfg.action_chunk_size):
            raise ValueError('RTC max_delay must be in (0, action_chunk_size]')
        self.train_action_on_delta = False
        self.dataset_action_is_delta = dataset_action_is_delta
        self.jepa_loss_w = model_cfg.ac_vjepa_jepa_loss_weight
        self.action_loss_w = model_cfg.ac_vjepa_action_loss_weight
        self.action_temporal_loss_w = getattr(model_cfg, 'ac_vjepa_action_temporal_loss_weight', 0.0)
        self.action_dim_weights = self._build_action_dim_weights()

    def __call__(self, model, batch):
        return self.forward_and_loss_one_video(model, batch)

    def _build_action_dim_weights(self) -> torch.Tensor:
        """Build normalized per-dim action weights (legacy finetune_egodex behavior)."""
        action_dim = int(self.ac_dim)
        cfg_w = getattr(self.model_cfg, 'loss_w', None)
        weights: torch.Tensor | None = None
        if action_dim == 7 and isinstance(cfg_w, (list, tuple)) and (len(cfg_w) >= 3):
            (w_xyz, w_rpy, w_gripper) = (float(cfg_w[0]), float(cfg_w[1]), float(cfg_w[2]))
            weights = torch.tensor([w_xyz, w_xyz, w_xyz, w_rpy, w_rpy, w_rpy, w_gripper], dtype=torch.float32, device=self.device)
        elif isinstance(cfg_w, (list, tuple)) and len(cfg_w) == action_dim:
            weights = torch.tensor([float(v) for v in cfg_w], dtype=torch.float32, device=self.device)
        if weights is None:
            weights = torch.ones(action_dim, dtype=torch.float32, device=self.device)
        if not torch.isfinite(weights).all() or float(weights.sum().item()) <= 0.0:
            weights = torch.ones(action_dim, dtype=torch.float32, device=self.device)
        return weights / weights.sum().clamp_min(1e-06)

    def _align_action_mask(self, action_mask: torch.Tensor, ref_actions: torch.Tensor) -> torch.Tensor:
        """Normalize action mask shape to [B, T, D] and cast dtype/device to ref."""
        mask = action_mask.to(device=ref_actions.device, dtype=ref_actions.dtype)
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        if mask.ndim != 3:
            raise ValueError(f'Expected action mask with 2/3 dims, got shape={tuple(mask.shape)}')
        if mask.shape[-1] == 1 and ref_actions.shape[-1] > 1:
            mask = mask.expand(-1, -1, ref_actions.shape[-1])
        elif mask.shape[-1] != ref_actions.shape[-1]:
            raise ValueError(f'Action mask/action dim mismatch: mask D={mask.shape[-1]}, action D={ref_actions.shape[-1]}')
        if mask.shape[1] != ref_actions.shape[1]:
            raise ValueError(f'Action mask/time dim mismatch: mask T={mask.shape[1]}, action T={ref_actions.shape[1]}')
        return mask

    def _build_action_supervision(self, pred_actions: torch.Tensor, gt_actions: torch.Tensor, action_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns supervision tensors for action loss.
        By default uses first-order temporal deltas unless dataset already outputs deltas.
        """
        if self.train_action_on_delta and (not self.dataset_action_is_delta) and (pred_actions.shape[1] > 1):
            pred_target = pred_actions[:, 1:, :] - pred_actions[:, :-1, :]
            gt_target = gt_actions[:, 1:, :] - gt_actions[:, :-1, :]
            target_mask = action_mask[:, 1:, :] * action_mask[:, :-1, :]
            return (pred_target, gt_target, target_mask)
        return (pred_actions, gt_actions, action_mask)

    def _compute_chunk_dynamics_metrics(self, pred_actions: torch.Tensor, gt_actions: torch.Tensor, action_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Measure intra-chunk temporal variation using adjacent-step L1 distance.
        Returns scalar tensors:
          - pred_chunk_adj_l1
          - gt_chunk_adj_l1
          - pred_vs_gt_chunk_adj_ratio
        """
        if pred_actions.shape[1] <= 1:
            zero = pred_actions.new_zeros(())
            return {'pred_chunk_adj_l1': zero, 'gt_chunk_adj_l1': zero, 'pred_vs_gt_chunk_adj_ratio': zero}
        adj_mask = action_mask[:, 1:, :] * action_mask[:, :-1, :]
        denom = adj_mask.sum().clamp_min(1.0)
        pred_adj_l1 = ((pred_actions[:, 1:, :] - pred_actions[:, :-1, :]).abs() * adj_mask).sum() / denom
        gt_adj_l1 = ((gt_actions[:, 1:, :] - gt_actions[:, :-1, :]).abs() * adj_mask).sum() / denom
        pred_vs_gt_ratio = pred_adj_l1 / gt_adj_l1.clamp_min(1e-06)
        return {'pred_chunk_adj_l1': pred_adj_l1, 'gt_chunk_adj_l1': gt_adj_l1, 'pred_vs_gt_chunk_adj_ratio': pred_vs_gt_ratio}

    def _compute_temporal_delta_alignment_loss(self, pred_actions: torch.Tensor, gt_actions: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """Match intra-chunk temporal deltas to discourage flat chunk collapse."""
        if pred_actions.shape[1] <= 1:
            return pred_actions.new_zeros(())
        d_pred = pred_actions[:, 1:, :] - pred_actions[:, :-1, :]
        d_gt = gt_actions[:, 1:, :] - gt_actions[:, :-1, :]
        d_mask = action_mask[:, 1:, :] * action_mask[:, :-1, :]
        denom = d_mask.sum().clamp_min(1.0)
        loss = (d_pred - d_gt).abs() * d_mask
        return loss.sum() / denom

    def _weighted_reduce_l1(self, abs_error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply per-dim weights then reduce with valid-mask normalization."""
        dim_w = self.action_dim_weights.to(device=abs_error.device, dtype=abs_error.dtype).view(1, 1, -1)
        weighted_mask = mask * dim_w
        denom = weighted_mask.sum().clamp_min(1.0)
        return (abs_error * weighted_mask).sum() / denom

    def _weighted_zero_baseline(self, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        dim_w = self.action_dim_weights.to(device=target.device, dtype=target.dtype).view(1, 1, -1)
        mask_wo_transl = mask.clone()
        weighted_mask = mask_wo_transl * dim_w
        denom = weighted_mask.sum().clamp_min(1.0)
        return (target.abs() * weighted_mask).sum() / denom

    def forward_and_loss_one_video(self, model, batch) -> dict[str, torch.Tensor]:
        if 'jepa_current_frame' not in batch or 'jepa_next_frame' not in batch:
            raise KeyError("Batch is missing JEPA frames. Expected keys: 'jepa_current_frame' and 'jepa_next_frame'. Please use wholebody posttrain transform that provides next-frame supervision.")
        gt_actions = batch['actions']
        (B, T, _) = gt_actions.shape
        sigmas = torch.rand((B,), device=self.device)
        if self.rtc_enabled:
            delay = torch.randint(low=0, high=self.max_delay, size=(B,), device=self.device, dtype=torch.long)
            prefix_mask = torch.arange(T, device=self.device)[None, :] < delay[:, None]
            sigmas = torch.where(prefix_mask, torch.tensor(0.0, device=self.device, dtype=sigmas.dtype), torch.tensor(1.0, device=self.device, dtype=sigmas.dtype))
            sigmas_action = sigmas
            while len(sigmas_action.shape) < len(batch['actions'].shape):
                sigmas_action = sigmas_action.unsqueeze(-1)
        else:
            sigmas_action = None
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], t5_input_ids=batch['text_input_ids'], t5_attention_mask=batch['text_attention_mask'], pixel_values=batch['pixel_values'], image_grid_thw=batch['image_grid_thw'], states=batch['states'] if 'states' in batch else None, current_frames=batch['jepa_current_frame'], context_frames=batch['jepa_context_frames'] if 'jepa_context_frames' in batch else None, next_frames=batch['jepa_next_frame'], state_valid_mask=batch['state_valid'] if 'state_valid' in batch else None, context_valid_mask=batch['jepa_context_valid'] if 'jepa_context_valid' in batch else None, actions=batch['actions'], view_token=batch['view_token'] if 'view_token' in batch else None, sigmas=sigmas_action if self.rtc_enabled else None)
        if outputs.pred_actions is None:
            raise RuntimeError('Model returned pred_actions=None during training. Please check ACVJEPAModel.forward/predictor wiring.')
        pred_actions = outputs.pred_actions
        if 'actions_mask' in batch:
            action_mask = self._align_action_mask(batch['actions_mask'], pred_actions)
        else:
            action_mask = torch.ones_like(pred_actions)
        (pred_target, gt_target, target_mask) = self._build_action_supervision(pred_actions=pred_actions, gt_actions=gt_actions, action_mask=action_mask)
        if self.rtc_enabled:
            postfix_mask = (~prefix_mask)[:, :, None].float()
            action_mask = target_mask * postfix_mask
        action_l1 = (pred_target.float() - gt_target.float()).abs()
        loss_action_delta = self._weighted_reduce_l1(action_l1, action_mask.float())
        loss_action_zero_baseline = self._weighted_zero_baseline(gt_target.float(), action_mask.float())
        loss_action_vs_zero_ratio = loss_action_delta / loss_action_zero_baseline.clamp_min(1e-06)
        chunk_dyn = self._compute_chunk_dynamics_metrics(pred_actions=pred_actions, gt_actions=gt_actions, action_mask=action_mask)
        loss_action_temporal = self._compute_temporal_delta_alignment_loss(pred_actions=pred_actions, gt_actions=gt_actions, action_mask=action_mask)
        loss_jepa_ego = F.mse_loss(outputs.pred_next_latent.float(), outputs.target_next_latent.float(), reduction='mean')
        loss_jepa = loss_jepa_ego
        loss_action_total = self.action_loss_w * loss_action_delta
        if self.action_temporal_loss_w > 0.0:
            loss_action_total = loss_action_total + self.action_temporal_loss_w * loss_action_temporal
        loss_total = self.jepa_loss_w * loss_jepa + loss_action_total
        losses = {'loss': loss_total, 'loss_jepa': loss_jepa, 'loss_jepa_ego': loss_jepa_ego, 'loss_action': loss_action_delta, 'loss_action_total': loss_action_total, 'loss_action_zero_baseline': loss_action_zero_baseline, 'loss_action_vs_zero_ratio': loss_action_vs_zero_ratio, 'loss_action_temporal': loss_action_temporal, 'pred_chunk_adj_l1': chunk_dyn['pred_chunk_adj_l1'], 'gt_chunk_adj_l1': chunk_dyn['gt_chunk_adj_l1'], 'pred_vs_gt_chunk_adj_ratio': chunk_dyn['pred_vs_gt_chunk_adj_ratio'], 'right_trigger': pred_actions[:, :, -1], 'left_trigger': pred_actions[:, :, -2], 'gt_right_trigger': gt_actions[:, :, -1], 'gt_left_trigger': gt_actions[:, :, -2]}
        return losses
