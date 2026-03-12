"""
TensorBoard Logger for Multi-Task Pretraining.

Monitors:
1. Aggregate losses (train/valid) - total loss, MoE aux loss
2. Task token (z_task) statistics - discriminability, variance

All metrics are logged as scalars per epoch for trend analysis.
"""

__all__ = ['TensorBoardCB', 'MultiTaskPretrainTensorBoardCB']

import os
import json
from datetime import datetime
from typing import Dict, Optional, List
import torch
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TENSORBOARD = True
except Exception:
    SummaryWriter = None  # type: ignore
    _HAS_TENSORBOARD = False

from .core import Callback
from .multi_task_callback import TASK_ID_MAP


# =============================================================================
# Main TensorBoard Callback
# =============================================================================

class TensorBoardCB(Callback):
    """
    Base TensorBoard callback for loss logging.
    
    Logs:
    - loss/train, loss/valid (per epoch)
    - loss/train_step (per batch, optional)
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        run_name: Optional[str] = None,
        add_timestamp: bool = True,
        flush_secs: int = 30,
    ):
        super().__init__()
        self.log_dir = log_dir
        self.run_name = run_name
        self.add_timestamp = add_timestamp
        self.flush_secs = flush_secs
        self.writer: Optional[SummaryWriter] = None
        self._train_step = 0
        self._resolved_run_dir: Optional[str] = None

    @staticmethod
    def _to_float(val):
        if isinstance(val, torch.Tensor):
            return val.detach().cpu().item()
        return float(val)

    def before_fit(self):
        if not _HAS_TENSORBOARD:
            print("[TensorBoardCB] TensorBoard not installed, skipping")
            return
        
        base_log_dir = getattr(self.learner, "tensorboard_log_dir", None) or self.log_dir or "runs"
        if not os.path.isabs(base_log_dir):
            base_log_dir = os.path.abspath(base_log_dir)

        run_name = self.run_name or getattr(self.learner, "tensorboard_run_name", None)
        if run_name is None:
            save_path = getattr(self.learner, "save_path", None)
            save_name = getattr(self.learner, "save_finetuned_model", None)
            if save_path and save_name:
                run_name = os.path.join(save_path, save_name).replace("/", "_")
            else:
                for attr in ["save_model_name", "save_pretrained_model"]:
                    run_name = getattr(self.learner, attr, None)
                    if run_name:
                        break

        if not run_name:
            run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        run_prefix = getattr(self.learner, "tensorboard_run_prefix", None)
        run_path = os.path.join(str(run_prefix).strip("/"), run_name) if run_prefix else run_name

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if self.add_timestamp:
            run_path = f"{run_path}_{timestamp}"
        
        full_dir = os.path.join(base_log_dir, run_path)
        os.makedirs(full_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=full_dir, flush_secs=self.flush_secs)
        self._resolved_run_dir = full_dir
        self._train_step = 0

        run_config = getattr(self.learner, "tensorboard_config", None)
        if run_config:
            self.writer.add_text("config/hparams", json.dumps(run_config, ensure_ascii=False, indent=2, default=str))

        print(f"[TensorBoardCB] Log directory: {full_dir}")

        setattr(self.learner, "tensorboard_run_info", {
            "base_dir": base_log_dir,
            "abs_dir": full_dir,
            "timestamp": timestamp,
        })

    def after_epoch_train(self):
        if self.writer is None:
            return
        epoch = getattr(self.learner, "epoch", 0)
        rec = getattr(self.learner, "recorder", None)
        if isinstance(rec, dict) and "train_loss" in rec and len(rec["train_loss"]) > 0:
            self.writer.add_scalar("loss/train", self._to_float(rec["train_loss"][-1]), epoch)

    def after_epoch_valid(self):
        if self.writer is None:
            return
        epoch = getattr(self.learner, "epoch", 0)
        rec = getattr(self.learner, "recorder", None)
        if isinstance(rec, dict) and "valid_loss" in rec and len(rec["valid_loss"]) > 0:
            self.writer.add_scalar("loss/valid", self._to_float(rec["valid_loss"][-1]), epoch)

    def after_fit(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
        if hasattr(self.learner, "tensorboard_run_info"):
            delattr(self.learner, "tensorboard_run_info")


# =============================================================================
# Multi-Task Pretraining Callback (Main)
# =============================================================================

class MultiTaskPretrainTensorBoardCB(Callback):
    """
    TensorBoard callback for multi-task pretraining (supplements TensorBoardCB).
    
    Logs per epoch (metrics not covered by the base TensorBoardCB):
    - MoE aux loss: loss/{split}/moe_aux
    - Task token discriminability: z_task/{split}/{inter_task_l2, intra_task_var, discriminability}
    
    Reuses the SummaryWriter owned by TensorBoardCB.
    """
    
    _Z_TASK_MAX_SAMPLES = 50

    def __init__(self):
        super().__init__()
        self.writer: Optional[SummaryWriter] = None
        self._moe_aux_train: List[float] = []
        self._moe_aux_valid: List[float] = []
        self._z_task_train: Dict[int, List[torch.Tensor]] = {}
        self._z_task_valid: Dict[int, List[torch.Tensor]] = {}
    
    def _get_writer(self) -> Optional[SummaryWriter]:
        if self.writer is not None:
            return self.writer
        for cb in getattr(self.learner, "cbs", []) or []:
            if isinstance(cb, TensorBoardCB) and cb.writer is not None:
                self.writer = cb.writer
                return self.writer
        return None
    
    def _collect_batch_data(self, is_train: bool):
        backbone = self._get_backbone()
        if backbone is not None:
            aux_loss = backbone.get_moe_aux_loss()
            if aux_loss is not None:
                buf = self._moe_aux_train if is_train else self._moe_aux_valid
                buf.append(aux_loss.detach().cpu().item())

        for cb in getattr(self.learner, "cbs", []) or []:
            if not (hasattr(cb, "task_token_gen") and cb.task_token_gen is not None):
                continue
            z_task = getattr(cb, "_last_z_task", None)
            task_name = getattr(self.learner, '_task_state', {}).get('task_name')
            task_id = TASK_ID_MAP.get(task_name) if task_name else getattr(cb, "_last_task_id", None)
            if z_task is not None and task_id is not None:
                z_buf = self._z_task_train if is_train else self._z_task_valid
                z_buf.setdefault(task_id, [])
                if len(z_buf[task_id]) < self._Z_TASK_MAX_SAMPLES:
                    z_buf[task_id].append(z_task.detach().cpu())
            break
        
    def after_batch_train(self):
        self._collect_batch_data(is_train=True)
    
    def after_batch_valid(self):
        self._collect_batch_data(is_train=False)
    
    def _log_epoch_stats(self, is_train: bool):
        writer = self._get_writer()
        if writer is None:
            return
        epoch = getattr(self.learner, "epoch", 0)
        split = "train" if is_train else "valid"

        aux_buf = self._moe_aux_train if is_train else self._moe_aux_valid
        if aux_buf:
            writer.add_scalar(f"loss/{split}/moe_aux", np.mean(aux_buf), epoch)

        self._log_z_task_stats(writer, epoch, split, is_train)

    def _log_z_task_stats(self, writer, epoch: int, split: str, is_train: bool):
        """Log z_task centroid distances and Fisher discriminability."""
        z_buffer = self._z_task_train if is_train else self._z_task_valid
        if len(z_buffer) < 2:
            return
        
        centroids, intra_vars = {}, {}
        for task_id, z_list in z_buffer.items():
            if not z_list:
                continue
            z_all = torch.cat(z_list, dim=0)
            centroid = z_all.mean(dim=0)
            centroids[task_id] = centroid
            intra_vars[task_id] = ((z_all - centroid) ** 2).mean().item()
        
        if len(centroids) < 2:
            return
        
        task_ids = sorted(centroids.keys())
        inter_dists = [
            torch.norm(centroids[task_ids[i]] - centroids[task_ids[j]]).item()
            for i in range(len(task_ids)) for j in range(i + 1, len(task_ids))
        ]
        
        inter_dist_mean = np.mean(inter_dists)
        intra_var_mean = np.mean(list(intra_vars.values()))
        
        writer.add_scalar(f"z_task/{split}/inter_task_l2", inter_dist_mean, epoch)
        writer.add_scalar(f"z_task/{split}/intra_task_var", intra_var_mean, epoch)
        if intra_var_mean > 1e-8:
            writer.add_scalar(f"z_task/{split}/discriminability", inter_dist_mean ** 2 / intra_var_mean, epoch)
    
    def after_epoch_train(self):
        self._log_epoch_stats(is_train=True)
        self._moe_aux_train = []
        self._z_task_train = {}
    
    def after_epoch_valid(self):
        self._log_epoch_stats(is_train=False)
        self._moe_aux_valid = []
        self._z_task_valid = {}
    
    def after_fit(self):
        # Clear reference to borrowed writer (TensorBoardCB owns flush/close)
        self.writer = None
