"""
Contextual Representation via cross-attention: Q x hidden_tokens -> z_context.

K,V are projected from d_model to d_task so the task-token dimension is
decoupled from the backbone hidden dimension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ContextualRepresentation(nn.Module):
    """
    Multi-head cross-attention: Query [B, d_task] x hidden_tokens [B, T, d_model] -> [B, d_task].

    Args:
        d_task: Task/Query dimension
        d_model: Backbone hidden dimension (K,V are projected d_model -> d_task)
        nhead: Number of attention heads (d_task must be divisible by nhead)
    """

    def __init__(
        self,
        d_task: int,
        d_model: int,
        nhead: int = 4,
    ):
        super().__init__()
        self.d_task = d_task
        self.d_model = d_model
        self.nhead = nhead

        assert d_task % nhead == 0, f"d_task ({d_task}) must be divisible by nhead ({nhead})"
        self.head_dim = d_task // nhead

        self.proj_k = nn.Linear(d_model, d_task)
        self.proj_v = nn.Linear(d_model, d_task)
        self.proj_q = nn.Linear(d_task, d_task)
        self.proj_out = nn.Linear(d_task, d_task)

        self.norm = nn.LayerNorm(d_task)

        self._init_weights()

    def _init_weights(self):
        for proj in [self.proj_q, self.proj_k, self.proj_v, self.proj_out]:
            nn.init.xavier_uniform_(proj.weight, gain=1.0 / math.sqrt(2))
            nn.init.zeros_(proj.bias)

    def forward(
        self,
        Q: torch.Tensor,
        hidden_tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            Q: [B, d_task] query vector
            hidden_tokens: [B, T, d_model] backbone hidden states
            padding_mask: [B, T] True for padded positions (optional)
            return_weights: whether to return attention weights

        Returns:
            context: [B, d_task]
            attn_weights: [B, nhead, 1, T] or None
        """
        B, T, _ = hidden_tokens.shape

        K = self.proj_k(hidden_tokens)                                       # [B, T, d_task]
        V = self.proj_v(hidden_tokens)                                       # [B, T, d_task]
        Q = self.proj_q(Q).unsqueeze(1)                                      # [B, 1, d_task]

        Q = Q.view(B, 1, self.nhead, self.head_dim).transpose(1, 2)         # [B, nhead, 1, head_dim]
        K = K.view(B, T, self.nhead, self.head_dim).transpose(1, 2)         # [B, nhead, T, head_dim]
        V = V.view(B, T, self.nhead, self.head_dim).transpose(1, 2)         # [B, nhead, T, head_dim]

        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale          # [B, nhead, 1, T]

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).unsqueeze(2)                    # [B, 1, 1, T]
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)                       # [B, nhead, 1, T]

        context = torch.matmul(attn_weights, V)                             # [B, nhead, 1, head_dim]
        context = context.transpose(1, 2).contiguous().view(B, 1, self.d_task)
        context = context.squeeze(1)                                         # [B, d_task]

        # No residual: context carries only hidden_tokens information, not Q
        context = self.norm(self.proj_out(context))

        if return_weights:
            return context, attn_weights
        return context, None
