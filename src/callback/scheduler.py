__all__ = ['OneCycleLR', 'LRFinderCB']

import os
import tempfile
import uuid
from math import inf

import torch
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

from ..basics import *
from .core import Callback


# ---------------------------------------------------------------------------
# LR Schedulers (used internally by LRFinderCB)
# ---------------------------------------------------------------------------

class _BaseLR(_LRScheduler):
    """Base class for LR finder schedulers that ramp LR over a fixed number of iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        if num_iter <= 1:
            raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter
        super().__init__(optimizer, last_epoch)


class LinearLR(_BaseLR):
    """Linearly increases the learning rate between two boundaries."""

    def get_lr(self):
        r = (self.last_epoch + 1) / (self.num_iter - 1)
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_BaseLR):
    """Exponentially increases the learning rate between two boundaries."""

    def get_lr(self):
        r = (self.last_epoch + 1) / (self.num_iter - 1)
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def valley(lrs: list, losses: list):
    "Suggests a learning rate from the longest valley"
    n = len(losses)
    max_start, max_end = 0, 0

    lds = [1] * n
    for i in range(1, n):
        for j in range(0, i):
            if (losses[i] < losses[j]) and (lds[i] < lds[j] + 1):
                lds[i] = lds[j] + 1
            if lds[max_end] < lds[i]:
                max_end = i
                max_start = max_end - lds[max_end]

    sections = (max_end - max_start) / 3
    idx = max_start + int(sections) + int(sections / 2)
    return float(lrs[idx])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class OneCycleLR(Callback):
    def __init__(self, lr_max=None,
                 total_steps=None,
                 steps_per_epoch=None,
                 pct_start=0.3,
                 anneal_strategy='cos',
                 base_momentum=0.85,
                 max_momentum=0.95,
                 div_factor=25.,
                 final_div_factor=1e4,
                 three_phase=False,
                 last_epoch=-1,
                 verbose=False):
        super().__init__()
        if lr_max is None:
            raise ValueError(
                "[OneCycleLR] lr_max is a required parameter and cannot be None. "
                "Please explicitly specify lr_max during initialization."
            )
        self.lr_max = lr_max
        self.total_steps, self.steps_per_epoch = total_steps, steps_per_epoch
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.base_momentum, self.max_momentum = base_momentum, max_momentum
        self.div_factor, self.final_div_factor = div_factor, final_div_factor
        self.three_phase = three_phase
        self.last_epoch = last_epoch
        self.verbose = verbose

    def before_fit(self):
        if not self.steps_per_epoch:
            self.steps_per_epoch = len(self.dls.train)
        self.lrs = []

        self.scheduler = lr_scheduler.OneCycleLR(
            optimizer=self.opt,
            max_lr=self.lr_max,
            total_steps=self.total_steps,
            epochs=self.n_epochs,
            steps_per_epoch=self.steps_per_epoch,
            pct_start=self.pct_start,
            anneal_strategy=self.anneal_strategy,
            cycle_momentum=False,
            base_momentum=self.base_momentum,
            max_momentum=self.max_momentum,
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor,
            three_phase=self.three_phase,
            last_epoch=self.last_epoch,
            verbose=self.verbose
        )

    def after_batch_train(self):
        num_param_groups = len(self.opt.param_groups)
        if not hasattr(self.scheduler, 'base_lrs'):
            self.scheduler.base_lrs = []
        num_base_lrs = len(self.scheduler.base_lrs)

        if num_param_groups > num_base_lrs:
            for i in range(num_base_lrs, num_param_groups):
                group = self.opt.param_groups[i]
                if 'initial_lr' not in group:
                    group['initial_lr'] = group.get('lr', self.lr_max / self.div_factor)
                self.scheduler.base_lrs.append(group['initial_lr'])
                if hasattr(self.scheduler, 'max_lrs'):
                    self.scheduler.max_lrs.append(self.lr_max)
                if hasattr(self.scheduler, 'min_lrs'):
                    self.scheduler.min_lrs.append(self.lr_max / self.final_div_factor)

        self.scheduler.step()
        self.lrs.append(self.scheduler.get_last_lr()[0])

    def after_fit(self):
        self.learner.scheduled_lrs = self.lrs


class LRFinderCB(Callback):
    def __init__(self, start_lr=1e-7, end_lr=10, num_iter=100, step_mode='exp', beta=0.98, suggestion='valley'):
        self.start_lr, self.end_lr = start_lr, end_lr
        self.num_iter = num_iter
        self.step_mode = step_mode
        if beta >= 1:
            raise ValueError("`beta` must be smaller than 1")
        self.beta = beta
        self.suggestion = suggestion

    def _save_tmp(self):
        """Use system temp directory + unique name to avoid concurrency conflicts"""
        tmp_dir = tempfile.gettempdir()
        fname = f"lrfind_{uuid.uuid4().hex}.pt"
        self.temp_path = os.path.join(tmp_dir, fname)

        state = {
            "model": {k: v.cpu() for k, v in self.learner.model.state_dict().items()},
            "opt": self.learner.opt.state_dict()
        }
        torch.save(state, self.temp_path)
        self._saved_task_token_state = None
        for cb in getattr(self.learner, "cbs", []) or []:
            tt_gen = getattr(cb, "task_token_gen", None)
            if tt_gen is None:
                continue
            self._saved_task_token_state = {
                k: v.detach().cpu().clone()
                for k, v in tt_gen.state_dict().items()
            }
            break

    def _load_tmp(self, device):
        """Load temporary model state, handling dynamic parameter group changes"""
        p = getattr(self, "temp_path", None)
        if not p or not os.path.isfile(p) or os.path.getsize(p) == 0:
            raise FileNotFoundError(
                f"[LRFinderCB._load_tmp] Temp checkpoint not found or empty: {p}. "
                f"Please check if _save_tmp executed correctly."
            )
        state = torch.load(p, map_location=device)
        self.learner.model.load_state_dict(state["model"])
        if self._saved_task_token_state is not None:
            for cb in getattr(self.learner, "cbs", []) or []:
                tt_gen = getattr(cb, "task_token_gen", None)
                if tt_gen is None:
                    continue
                tt_gen.load_state_dict(self._saved_task_token_state)
                break

        # Handle case where parameter groups were added during lr_finder (e.g., conditional encoder)
        saved_opt_state = state["opt"]
        current_num_groups = len(self.learner.opt.param_groups)
        saved_num_groups = len(saved_opt_state["param_groups"])

        if current_num_groups == saved_num_groups:
            self.learner.opt.load_state_dict(saved_opt_state)
        else:
            # Only restore state for original groups, keep new groups as-is
            for i, saved_group in enumerate(saved_opt_state["param_groups"]):
                if i < current_num_groups:
                    for key in saved_group:
                        if key != "params":
                            self.learner.opt.param_groups[i][key] = saved_group[key]
            # Restore optimizer state (momentum buffers, etc.) for parameters that exist
            if "state" in saved_opt_state:
                for param_id, param_state in saved_opt_state["state"].items():
                    if param_id in self.learner.opt.state:
                        self.learner.opt.state[param_id] = {
                            k: v.to(device) if hasattr(v, 'to') else v
                            for k, v in param_state.items()
                        }

    def before_fit(self):
        self.losses, self.lrs = [], []
        self.best_loss, self.aver_loss = inf, 0
        self.train_iter = 0

        self._save_tmp()
        self.set_lr(self.start_lr)

        if not self.num_iter: self.num_iter = len(self.dls.train)

        if self.step_mode.lower() == "exp":
            self.scheduler = ExponentialLR(self.opt, self.end_lr, self.num_iter)
        elif self.step_mode.lower() == "linear":
            self.scheduler = LinearLR(self.opt, self.end_lr, self.num_iter)

    def after_batch_train(self):
        self.train_iter += 1
        self.scheduler.step()
        self.lrs.append(self.scheduler.get_last_lr()[0])

        self.smoothing(self.beta)
        if self.smoothed_loss < self.best_loss: self.best_loss = self.smoothed_loss

        if self.smoothed_loss > 4 * self.best_loss:
            raise KeyboardInterrupt
        if self.train_iter > self.num_iter:
            raise KeyboardInterrupt

    def smoothing(self, beta):
        self.aver_loss = beta * self.aver_loss + (1 - beta) * self.loss.detach().item()
        self.smoothed_loss = self.aver_loss / (1 - beta ** self.train_iter)
        self.losses.append(self.smoothed_loss)

    def after_fit(self):
        self.learner.opt.zero_grad()
        if self.suggestion:
            self.suggested_lr = valley(self.lrs, self.losses)
        self._load_tmp(self.learner.device)
        os.remove(self.temp_path)

    def set_lr(self, lrs):
        if not isinstance(lrs, list): lrs = [lrs] * len(self.opt.param_groups)
        if len(lrs) != len(self.opt.param_groups):
            raise ValueError(
                "Length of `lrs` is not equal to the number of parameter groups "
                + "in the given optimizer")
        for param_group, lr in zip(self.opt.param_groups, lrs):
            param_group["lr"] = lr
