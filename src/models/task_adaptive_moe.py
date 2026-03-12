"""
Task-Adaptive Mixture of Experts (MoE)
=======================================

Architecture (Shared Expert + Routed Experts):
  y = SharedExpert(x) + alpha * dropout(norm(MoE(x, task)))
  - This module provides Routed Experts for task-specific corrections
  - alpha controlled by MoEAlphaScheduleCB callback (0.0 -> target)

Task-Adaptive Router:
  router_score = W @ [x; task_proj(z_task)]   <- concat mode (default)
  scores = softmax(router_score / temperature)
  top_k_weights, top_k_indices = top_k(scores, k)

Router Fusion Modes:
  1. Concatenation (default): router_score = W @ [x; task_proj(z_task)]
  2. Additive Bias: router_score = W @ x + task_bias(z_task)
  3. Multiplicative Gate: router_score = W @ (x * task_gate(z_task))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Expert(nn.Module):
    """Single Expert FFN"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TaskAdaptiveRouter(nn.Module):
    """
    Task-Adaptive Router
    
    Fusion modes:
    - 'concat': router_score = W @ [x; task_proj] (default, most expressive)
    - 'additive': router_score = W @ x + task_bias
    - 'multiplicative': router_score = W @ (x * task_gate)
    """
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        d_task: int = 16,
        fusion_mode: str = 'concat',
        top_k: int = 2,
        use_aux_loss: bool = True,
        router_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.d_task = d_task
        self.fusion_mode = fusion_mode
        self.top_k = top_k
        self.use_aux_loss = use_aux_loss
        self.router_temperature = router_temperature
        
        if fusion_mode == 'concat':
            self.router = nn.Linear(d_model + d_task, num_experts, bias=False)
            self.task_proj = nn.Sequential(
                nn.Linear(d_task, d_task),
                nn.GELU(),
            )
            nn.init.xavier_uniform_(self.router.weight[:, :d_model])
            nn.init.normal_(self.router.weight[:, d_model:], std=0.02)
        else:
            self.router = nn.Linear(d_model, num_experts, bias=False)
        
        if fusion_mode == 'additive':
            self.task_to_bias = nn.Sequential(
                nn.Linear(d_task, d_task * 2),
                nn.GELU(),
                nn.Linear(d_task * 2, num_experts),
            )
            nn.init.zeros_(self.task_to_bias[-1].weight)
            nn.init.zeros_(self.task_to_bias[-1].bias)
        elif fusion_mode == 'multiplicative':
            self.task_to_gate = nn.Sequential(
                nn.Linear(d_task, d_task * 2),
                nn.GELU(),
                nn.Linear(d_task * 2, d_model),
            )
            nn.init.zeros_(self.task_to_gate[-1].weight)
            nn.init.zeros_(self.task_to_gate[-1].bias)
        
        self.expert_bias = nn.Parameter(torch.zeros(num_experts))
        self._aux_loss = None
    
    def forward(
        self,
        x: torch.Tensor,
        z_task: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, L, D] or [B*L, D]
            z_task: [B, d_task] task representation
            tau: [B, 1] dynamic temperature from TaskTokenGenerator (optional)

        Returns:
            top_k_weights: [B, L, top_k] normalized weights
            top_k_indices: [B, L, top_k] expert indices
            router_probs: [B, L, num_experts] full routing probabilities
        """
        original_shape = x.shape
        if x.dim() == 3:
            B, L, D = x.shape
            x_flat = x.reshape(B * L, D)
        else:
            B, L = x.shape[0], 1
            x_flat = x
        
        z_task_aligned = None
        if z_task is not None:
            B_task = z_task.shape[0]
            if B_task != B:
                assert B % B_task == 0, f"Batch size B={B} must be divisible by task batch size B_task={B_task}"
                n_vars = B // B_task
                z_task_aligned = z_task.repeat_interleave(n_vars, dim=0)
            else:
                z_task_aligned = z_task

        if self.fusion_mode == 'additive' and z_task_aligned is not None:
            router_logits = self.router(x_flat)
            task_bias = self.task_to_bias(z_task_aligned)
            task_bias_expanded = task_bias.unsqueeze(1).expand(B, L, -1).reshape(B * L, -1)
            router_logits = router_logits + task_bias_expanded
        elif self.fusion_mode == 'multiplicative' and z_task_aligned is not None:
            task_gate = torch.sigmoid(self.task_to_gate(z_task_aligned))
            task_gate_expanded = task_gate.unsqueeze(1).expand(B, L, -1).reshape(B * L, -1)
            x_gated = x_flat * task_gate_expanded
            router_logits = self.router(x_gated)
        elif self.fusion_mode == 'concat':
            if z_task_aligned is not None:
                task_proj = self.task_proj(z_task_aligned)
                task_proj_expanded = task_proj.unsqueeze(1).expand(B, L, -1).reshape(B * L, -1)
            else:
                task_proj_expanded = torch.zeros(B * L, self.d_task, device=x_flat.device, dtype=x_flat.dtype)
            x_concat = torch.cat([x_flat, task_proj_expanded], dim=-1)
            router_logits = self.router(x_concat)
        else:
            router_logits = self.router(x_flat)
        
        router_logits = router_logits + self.expert_bias
        
        if tau is not None:
            temperature = tau.mean().clamp(min=0.1)
        else:
            temperature = self.router_temperature
        router_probs = F.softmax(router_logits / temperature, dim=-1)
        
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        if self.use_aux_loss:
            aux_loss = self._compute_aux_loss(router_probs, top_k_indices)
            self._aux_loss = aux_loss if self.training else aux_loss.detach()
        else:
            self._aux_loss = None
        
        if len(original_shape) == 3:
            top_k_weights = top_k_weights.reshape(B, L, self.top_k)
            top_k_indices = top_k_indices.reshape(B, L, self.top_k)
            router_probs = router_probs.reshape(B, L, self.num_experts)
        
        return top_k_weights, top_k_indices, router_probs
    
    def _compute_aux_loss(self, router_probs: torch.Tensor, top_k_indices: torch.Tensor) -> torch.Tensor:
        """
        Load balancing auxiliary loss: L = N * sum_i(f_i * P_i)
        where f_i = fraction of tokens selecting expert i, P_i = mean routing probability.
        """
        expert_mask = F.one_hot(top_k_indices, self.num_experts).float()
        expert_mask = expert_mask.sum(dim=-2) / max(self.top_k, 1)
        f = expert_mask.mean(dim=0)
        P = router_probs.mean(dim=0)
        return self.num_experts * (f * P).sum()
    
    def get_aux_loss(self) -> Optional[torch.Tensor]:
        return self._aux_loss


class TaskAdaptiveMoE(nn.Module):
    """
    Task-Adaptive Mixture of Experts

    Provides Routed Experts with task-aware routing, designed for use with
    an external Shared Expert (Dense FFN) in TransformerLayer:
      y = SharedExpert(x) + alpha * dropout(norm(MoE(x, task)))

    Aggregation Modes (set via set_aggregation_mode()):
      - 'router': top-k routing with learned weights (default, used during pretrain)
      - 'shared_only': returns zeros -> y = SharedExpert(x) (used during finetune/linear probe)
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        d_task: int = 16,
        dropout: float = 0.1,
        activation: str = 'gelu',
        router_fusion_mode: str = 'concat',
        use_aux_loss: bool = True,
        router_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_task = d_task
        self.aggregation_mode = 'router'
        
        self.routed_experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout, activation)
            for _ in range(num_experts)
        ])
        
        self.router = TaskAdaptiveRouter(
            d_model=d_model,
            num_experts=num_experts,
            d_task=d_task,
            fusion_mode=router_fusion_mode,
            top_k=top_k,
            use_aux_loss=use_aux_loss,
            router_temperature=router_temperature,
        )
        
    
    def forward(
        self,
        x: torch.Tensor,
        z_task: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: [B, L, D] input tokens
            z_task: [B, d_task] task representation (optional)
            tau: [B, 1] dynamic temperature (optional)

        Returns:
            Output tensor [B, L, D]
        """
        B, L, D = x.shape

        # shared_only: return zeros -> final output = SharedExpert(x) only
        if self.aggregation_mode == 'shared_only':
            return torch.zeros_like(x)

        # Router-based expert selection
        top_k_weights, top_k_indices, router_probs = self.router(x, z_task, tau=tau)

        x_flat = x.reshape(B * L, D)
        flat_indices = top_k_indices.reshape(B * L, self.top_k)
        flat_weights = top_k_weights.reshape(B * L, self.top_k)
        
        moe_routed = torch.zeros(B * L, D, device=x.device, dtype=x.dtype)
        for expert_idx in range(self.num_experts):
            expert_mask = (flat_indices == expert_idx)
            if not expert_mask.any():
                continue
            token_indices = expert_mask.any(dim=1)
            expert_output = self.routed_experts[expert_idx](x_flat[token_indices])
            selected_weights = flat_weights[token_indices] * expert_mask[token_indices].float()
            total_weight = selected_weights.sum(dim=1, keepdim=True)
            moe_routed[token_indices] += expert_output * total_weight
        
        return moe_routed.reshape(B, L, D)
    
    def get_aux_loss(self) -> Optional[torch.Tensor]:
        return self.router.get_aux_loss()
    
    def set_aggregation_mode(self, mode: str):
        """
        Args:
            mode: 'router' or 'shared_only'
                - 'router': Use router to select top-k experts (pretrain)
                - 'shared_only': Returns zeros -> y = SharedExpert(x) only (finetune/linear probe)
        """
        assert mode in ('router', 'shared_only'), f"Invalid aggregation mode: {mode}"
        self.aggregation_mode = mode
