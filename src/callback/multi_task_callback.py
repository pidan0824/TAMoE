"""
Multi-Task Reconstruction Pretraining Callback

Tasks: PM, MPM, RFM, SFM, HM, DM.
Each batch samples one task. view_meta encodes corruption strategy for TaskTokenGenerator.
"""

__all__ = ['MultiTaskReconCB', 'build_view_meta', 'build_view_meta_batch',
           'TASK_ID_MAP', 'VIEW_META_DIM']

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Any

from .core import Callback
from ..masking import (
    PatchMasking, BlockMasking,
    RandomFreqMasking, StructuredFreqMasking,
    DecompositionMasking,
    TaskSampler,
)


VIEW_META_DIM = 8

# Domain mapping (for view_meta): time=0, freq=1
DOMAIN_MAP = {
    'PM': 0,   # time domain
    'MPM': 0,  # time domain
    'RFM': 1,  # freq domain
    'SFM': 1,  # freq domain
    'DM': 0,   # time domain
    'HM': 0,   # time domain
}

# Full-sequence loss flag: 0=masked only, 1=full sequence
FULL_SEQ_LOSS_MAP = {
    'PM': 0,   # masked patches only
    'MPM': 0,  # masked patches only
    'RFM': 1,  # full sequence (freq corruption affects all)
    'SFM': 1,  # full sequence (freq corruption affects all)
    'DM': 0,   # masked patches only
    'HM': 1,   # full sequence (holistic reconstruction)
}

STRUCTURED_MAP = {
    'PM': 0,   # random patch masking
    'MPM': 1,  # structured block masking
    'RFM': 0,  # random frequency masking
    'SFM': 1,  # structured frequency band masking
    'DM': 1,   # structured decomposition masking
    'HM': 0,   # random patch masking
}


# Task ID mapping.
# Default mode (coarse): all recon tasks share task_emb[0] (1 slot).
# Fine-grained mode (use_fine_grained_task_id=True): each task uses its own index (7 slots, 0 reserved).
TASK_ID_MAP = {
    'recon': 0,
    'PM': 1, 'MPM': 2, 'RFM': 3, 'SFM': 4, 'DM': 5, 'HM': 6,
}


def build_view_meta(
    task_name: str,
    mask_ratio: float = 0.4,
    band_value: float = 0.5,
    trend_mask_ratio: float = 0.0,
    residual_mask_ratio: float = 0.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build view_meta [8] describing corruption strategy.
    
    Layout: [mask_ratio, time_flag, freq_flag, full_seq_flag,
             structured_flag, band_value, trend_ratio, residual_ratio]
    """
    view_meta = torch.zeros(VIEW_META_DIM, device=device)
    view_meta[0] = mask_ratio
    domain_idx = DOMAIN_MAP.get(task_name, 0)
    view_meta[1 + domain_idx] = 1.0
    view_meta[3] = FULL_SEQ_LOSS_MAP.get(task_name, 0)
    view_meta[4] = STRUCTURED_MAP.get(task_name, 0)
    if task_name == 'SFM':
        view_meta[5] = band_value
    if task_name == 'DM':
        view_meta[6] = trend_mask_ratio
        view_meta[7] = residual_mask_ratio
    
    return view_meta


def build_view_meta_batch(
    batch_size: int,
    task_name: str,
    mask_ratio: float = 0.4,
    band_value: float = 0.5,
    trend_mask_ratio: float = 0.0,
    residual_mask_ratio: float = 0.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build view_meta for a batch (all samples have same task).
    
    Returns:
        view_meta: [B, view_dim] tensor
    """
    single = build_view_meta(
        task_name=task_name,
        mask_ratio=mask_ratio,
        band_value=band_value,
        trend_mask_ratio=trend_mask_ratio,
        residual_mask_ratio=residual_mask_ratio,
        device=device,
    )
    return single.unsqueeze(0).expand(batch_size, -1).clone()


class MultiTaskReconCB(Callback):
    """
    Multi-Task Reconstruction Pretraining Callback.
    
    Samples one task per batch. Loss: PM/MPM/DM masked MSE, RFM/SFM/HM full MSE.
    
    Args:
        patch_len: Patch length
        stride: Stride between patches
        task_probs: Task sampling probabilities (default: PM/MPM 0.25, rest 0.15, HM 0.05)
        loss_weights: Loss type weights, keys 'time' and 'decomp' (both default 1.0)
        use_task_token: Enable TaskTokenGenerator view_meta injection
        use_fine_grained_task_id: Each recon task uses its own task_emb (default False)
    """
    
    def __init__(
        self,
        patch_len: int,
        stride: int,
        task_probs: Optional[Dict[str, float]] = None,
        pm_mask_ratio: float = 0.4,
        mpm_mask_ratio: float = 0.4,
        rfm_mask_ratio: float = 0.3,
        sfm_tau: float = 0.5,
        sfm_band: str = 'random',
        dm_trend_mask_ratio: float = 0.2,
        dm_residual_mask_ratio: float = 0.5,
        hm_time_mask_ratio: float = 0.4,
        loss_weights: Optional[Dict[str, float]] = None,
        use_task_token: bool = False,
        use_fine_grained_task_id: bool = False,
        seed: int = None,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.use_task_token = use_task_token
        self.use_fine_grained_task_id = use_fine_grained_task_id
        
        # Isolated torch RNG for masking reproducibility
        self._seed = seed
        self._torch_generator = None  
        self._step_counter = 0
        
        # Task sampler (owns probabilities + numpy RNG)
        default_probs = {
            'PM': 0.16, 'MPM': 0.16,
            'RFM': 0.18, 'SFM': 0.18,
            'DM': 0.18, 'HM': 0.14,
        }
        self._sampler = TaskSampler(task_probs or default_probs, seed=seed)
        self._task_names = self._sampler.task_names
        
        self.loss_weights = loss_weights or {
            'time': 1.0,
            'decomp': 1.0,
        }

        base_config = {'patch_len': patch_len, 'stride': stride}
        
        self.strategies = {
            'PM': PatchMasking(**base_config, mask_ratio=pm_mask_ratio),
            'MPM': BlockMasking(**base_config, mask_ratio=mpm_mask_ratio),
            'RFM': RandomFreqMasking(**base_config, mask_ratio=rfm_mask_ratio),
            'SFM': StructuredFreqMasking(**base_config, tau=sfm_tau, mask_band=sfm_band),
            'DM': DecompositionMasking(**base_config, 
                                       trend_mask_ratio=dm_trend_mask_ratio,
                                       residual_mask_ratio=dm_residual_mask_ratio),
            'HM': PatchMasking(**base_config, mask_ratio=hm_time_mask_ratio),
        }
        
        self._device = None
        self._current_task = None

        self._task_loss_buffers = {task: [] for task in self._task_names}
        self._task_counts = {task: 0 for task in self._task_names}
        self._last_epoch_loss = {task: float('nan') for task in self._task_names}
        self._last_epoch_count = {task: 0 for task in self._task_names}
    
    def before_fit(self):
        """Override loss function with multi-task loss."""
        self.learner.loss_func = self._multi_task_loss
        self._device = self.learner.device
        for strategy in self.strategies.values():
            strategy.to(self._device)
        if self._seed is not None:
            self._torch_generator = torch.Generator(device=self._device)
            self._torch_generator.manual_seed(self._seed)
    
    def before_forward(self):
        """Sample task and prepare masked view."""
        self._prepare_batch()
    
    def _prepare_batch(self):
        """Sample a task and create masked view."""
        self._current_task = self._sampler.sample()
        
        xb = self.xb
        B = xb.shape[0]
        if self._seed is not None:
            self._torch_generator.manual_seed(self._seed + self._step_counter)
        self._step_counter += 1
        strategy = self.strategies[self._current_task]
        ctx = {'generator': self._torch_generator if self._seed is not None else None}
        current_view = strategy.make_view(xb, ctx=ctx)

        # [B, N, C*patch_len] -> [B, N, C, patch_len]
        x_in = current_view['x_in']
        _, N, D = x_in.shape
        n_vars = D // self.patch_len
        x_in_reshaped = x_in.view(B, N, n_vars, self.patch_len)
        
        self.learner.xb = x_in_reshaped
        self.learner.yb = current_view['target_time']

        if self.use_fine_grained_task_id:
            task_id = TASK_ID_MAP.get(self._current_task, 1)
            task_labels = torch.full((B, N), task_id, dtype=torch.long, device=xb.device)
            self.learner.forward_kwargs = {'task': 'recon', 'task_labels': task_labels}
        else:
            self.learner.forward_kwargs = {'task': 'recon'}

        self.learner._task_state = {
            'view': current_view,
            'task_name': self._current_task,
            'view_meta': self._build_current_view_meta(B, xb.device) if self.use_task_token else None,
        }
    
    def _build_current_view_meta(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Build view_meta [B, VIEW_META_DIM] for current task."""
        task = self._current_task
        strategy = self.strategies[task]
        kwargs = {'batch_size': batch_size, 'task_name': task, 'device': device}

        if task == 'SFM':
            band_value = 0.5
            if hasattr(strategy, 'mask_band'):
                band_map = {'low': 0.0, 'high': 1.0, 'random': 0.5}
                band_value = band_map.get(strategy.mask_band, 0.5)
            kwargs.update(mask_ratio=0.5, band_value=band_value)
        elif task == 'DM':
            trend_mask = strategy.trend_mask_ratio
            residual_mask = strategy.residual_mask_ratio
            kwargs.update(
                mask_ratio=(trend_mask + residual_mask) / 2,
                trend_mask_ratio=trend_mask,
                residual_mask_ratio=residual_mask,
            )
        else:  # PM, MPM, RFM, HM
            kwargs['mask_ratio'] = strategy.mask_ratio

        return build_view_meta_batch(**kwargs)
    
    def _multi_task_loss(self, pred, target):
        """Compute task-specific reconstruction loss."""
        ts = getattr(self.learner, '_task_state', {})
        view = ts.get('view')
        task = ts.get('task_name', 'PM')
        
        if view is None:
            # Fallback to simple MSE
            return F.mse_loss(pred, target)
        
        losses = {}
        
        if pred.dim() == 4:
            # PretrainHead output: [B, N, C, patch_len] -> [B, N, C*patch_len]
            B, N, C, P = pred.shape
            pred = pred.reshape(B, N, C * P)
        
        if target.dim() == 4:
            B, N, C, P = target.shape
            target = target.reshape(B, N, C * P)
        
        if task in ['PM', 'MPM']:
            mask = view['mask']
            losses['time'] = self._masked_mse(pred, target, mask)
            
        elif task in ['RFM', 'SFM']:
            losses['time'] = F.mse_loss(pred, target)
            
        elif task == 'DM':
            losses['decomp'] = self._masked_mse(pred, target, view['mask'])
                
        elif task == 'HM':
            losses['time'] = F.mse_loss(pred, target)
        
        total_loss = 0.0
        for loss_type, loss_val in losses.items():
            weight = self.loss_weights.get(loss_type, 1.0)
            total_loss = total_loss + weight * loss_val
        
        if self.model.training and task in self._task_loss_buffers:
            self._task_loss_buffers[task].append(total_loss.item())
            self._task_counts[task] += 1
        
        return total_loss
    
    def _masked_mse(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.BoolTensor
    ) -> torch.Tensor:
        """Compute MSE only on masked positions."""
        mse = (pred - target) ** 2
        if mse.dim() > 2:
            mse = mse.mean(dim=-1)  # [B, N]
        
        num_masked = mask.sum()
        if num_masked > 0:
            loss = (mse * mask.float()).sum() / num_masked
        else:
            loss = mse.mean()
        
        return loss
    
    def after_epoch_train(self):
        """Store epoch-level task-wise losses and reset batch buffers."""
        for task in self._task_names:
            self._last_epoch_loss[task] = np.mean(self._task_loss_buffers[task]) if self._task_loss_buffers[task] else float('nan')
            self._last_epoch_count[task] = self._task_counts[task]
            self._task_loss_buffers[task] = []
            self._task_counts[task] = 0
    
    def after_epoch(self):
        """Print per-task loss summary."""
        loss_parts = []
        count_parts = []
        total_samples = 0
        for task in self._task_names:
            loss_val = self._last_epoch_loss[task]
            count_val = self._last_epoch_count[task]
            if not np.isnan(loss_val):
                loss_parts.append(f"{task}={loss_val:.3f}")
            else:
                loss_parts.append(f"{task}=N/A")
            count_parts.append(f"{task}={count_val:>3d}")
            total_samples += count_val

        if loss_parts and total_samples > 0:
            print(f"  [Tasks] Loss: {' | '.join(loss_parts)}")
            print(f"  [Tasks] Samples: {' | '.join(count_parts)}")
    
    def get_task_stats(self) -> Dict[str, Any]:
        """Get task sampling statistics."""
        return {
            'task_probs': self._sampler.task_probs,
            'task_counts': self._task_counts.copy(),
            'current_task': self._current_task,
        }

