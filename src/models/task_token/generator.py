"""
Task Token Generator: produces z_task for MoE routing.

Pipeline:
    StateGate(global_desc) -> alpha_view, tau
    QueryBuilder(task_emb, view_meta, alpha_view) -> Q
    CR(Q, hidden_tokens) -> z_context
    VWR(Q, x_input) -> z_vise
    z_task = TaskNorm(z_context + z_vise)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .state_gate import StateGate
from .query_builder import QueryBuilder
from .contextual_repr import ContextualRepresentation
from .variable_wise_repr import VWR, LearnableVWR


class TaskNorm(nn.Module):
    """Normalize to unit norm, then scale by learnable gain."""

    def __init__(self, d_task: int, init_gain: Optional[float] = None):
        super().__init__()
        if init_gain is None:
            init_gain = d_task ** 0.5
        self.gain = nn.Parameter(torch.tensor(init_gain))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return (z / norm) * F.softplus(self.gain)


class TaskTokenGenerator(nn.Module):
    """
    z_task = TaskNorm(z_context + z_vise), with tau for router temperature.

    Each component can be ablated via use_cr / use_vwr flags.
    """

    def __init__(
        self,
        d_task: int,
        d_model: int,
        view_dim: int = 8,
        global_desc_dim: int = 16,
        use_fine_grained_task_id: bool = False,
        vwr_beta: float = 0.5,
        nhead: int = 4,
        dropout: float = 0.1,
        use_cr: bool = True,
        use_vwr: bool = True,
        use_periodic: bool = True,
        use_spectral: bool = True,
        use_stat: bool = True,
        use_global_desc: bool = True,
        use_learnable_vwr: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.d_task = d_task
        self.d_model = d_model
        self.view_dim = view_dim
        self.global_desc_dim = global_desc_dim
        num_tasks = 7 if use_fine_grained_task_id else 1
        self.num_tasks = num_tasks
        self.use_fine_grained_task_id = use_fine_grained_task_id
        self.use_cr = use_cr
        self.use_vwr = use_vwr
        self.use_global_desc = use_global_desc
        self.use_learnable_vwr = use_learnable_vwr

        self.task_emb = nn.Embedding(num_tasks, d_task)
        self._init_task_emb()

        self.state_gate = StateGate(global_desc_dim, use_global_desc=use_global_desc) if (use_cr or use_vwr) else None
        self.query_builder = QueryBuilder(d_task, view_dim) if (use_cr or use_vwr) else None
        self.ctx_repr = ContextualRepresentation(d_task, d_model, nhead) if use_cr else None

        if self.use_vwr:
            if self.use_learnable_vwr:
                self.vwr = LearnableVWR(
                    d_task,
                    nhead=nhead,
                    dropout=dropout,
                    beta=vwr_beta,
                )
            else:
                self.vwr = VWR(
                    d_task,
                    beta=vwr_beta,
                    use_periodic=use_periodic,
                    use_spectral=use_spectral,
                    use_stat=use_stat,
                )
        else:
            self.vwr = None

        self.task_norm = TaskNorm(d_task)

        self.config = {
            'd_task': d_task,
            'd_model': d_model,
            'view_dim': view_dim,
            'global_desc_dim': global_desc_dim,
            'use_fine_grained_task_id': use_fine_grained_task_id,
            'vwr_beta': vwr_beta,
            'nhead': nhead,
            'dropout': dropout,
            'use_cr': use_cr,
            'use_vwr': use_vwr,
            'use_periodic': use_periodic,
            'use_spectral': use_spectral,
            'use_stat': use_stat,
            'use_global_desc': use_global_desc,
            'use_learnable_vwr': use_learnable_vwr,
        }

    def _init_task_emb(self):
        nn.init.normal_(self.task_emb.weight, std=0.02)

    def forward(
        self,
        task_id: torch.Tensor,
        view_meta: torch.Tensor,
        global_desc: torch.Tensor,
        hidden_tokens: torch.Tensor,
        x_input: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_all: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Returns dict with z_task, tau (+ optional intermediates)."""
        B = task_id.shape[0]
        task_emb = self.task_emb(task_id)

        z_context = None
        z_vise = None
        alpha_view = None
        attn_weights = None
        tau = torch.ones(B, 1, device=task_emb.device)
        gate_info = {}

        if self.state_gate is not None:
            alpha_view, tau = self.state_gate(global_desc)

        if self.use_cr or self.use_vwr:
            Q = self.query_builder(task_emb, view_meta, alpha_view)

        if self.use_cr:
            z_context, attn_weights = self.ctx_repr(
                Q, hidden_tokens, padding_mask, return_weights=return_all
            )

        if self.use_vwr:
            z_vise, gate_info = self.vwr(Q, x_input)

        if self.use_cr and self.use_vwr:
            z_task = self.task_norm(z_context + z_vise)
        elif self.use_cr:
            z_task = self.task_norm(z_context)
        elif self.use_vwr:
            z_task = self.task_norm(z_vise)
        else:
            z_task = self.task_norm(task_emb)

        output = {
            'z_task': z_task,
            'tau': tau,
        }

        if return_all:
            if z_vise is not None:
                output['z_vise'] = z_vise
            if z_context is not None:
                output['z_context'] = z_context
            if gate_info:
                output['gate_info'] = gate_info
            if alpha_view is not None:
                output['alpha_view'] = alpha_view
            if attn_weights is not None:
                output['attn_weights'] = attn_weights

        return output
