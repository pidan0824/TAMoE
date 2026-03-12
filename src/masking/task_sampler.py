"""
Task Sampler for Multi-Task Pretraining.

Weighted random sampling over reconstruction tasks (PM, MPM, RFM, SFM, DM, HM).
Separated from callback logic so the masking module owns both
*what* each task does and *how* tasks are selected.
"""

import numpy as np
from typing import Dict, Optional, List


class TaskSampler:
    """Weighted random task sampler with optional single-task fast path.

    Args:
        task_probs: {task_name: weight} dict.  Weights are normalised internally.
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        task_probs: Dict[str, float],
        seed: Optional[int] = None,
    ):
        total = sum(task_probs.values())
        self.task_probs = {k: v / total for k, v in task_probs.items()}
        self._task_names: List[str] = list(self.task_probs.keys())
        self._task_probs_list: List[float] = [self.task_probs[t] for t in self._task_names]
        self._rng = np.random.default_rng(seed)

        self._single_task: Optional[str] = None
        for t, p in self.task_probs.items():
            if p >= 1.0 - 1e-6:
                self._single_task = t
                break

    def sample(self) -> str:
        """Return one task name drawn according to the probability distribution."""
        if self._single_task is not None:
            return self._single_task
        return self._rng.choice(self._task_names, p=self._task_probs_list)

    @property
    def task_names(self) -> List[str]:
        return list(self._task_names)

    def __repr__(self) -> str:
        probs_str = ", ".join(f"{t}={p:.2f}" for t, p in self.task_probs.items())
        return f"TaskSampler({probs_str})"
