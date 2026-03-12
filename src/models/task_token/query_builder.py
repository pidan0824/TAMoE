"""
Query Builder Module for Task Token Generator
================================================

Builds the Query vector for cross-attention:
    Q = LN(q_task + alpha_view * q_view)

where q_task = proj_task(task_emb), q_view = proj_view(view_meta).

Component Semantics:
    - task_emb: Base direction from task embedding (what objective)
    - view_meta: Corruption metadata like mask_ratio, domain type (how corrupted)
    - alpha_view: Fusion weight from StateGate, controls view contribution

Note: view_meta encodes corruption strategy (mask_ratio, domain, structured flags),
      alpha_view is computed externally by StateGate from global statistics.
"""

import torch
import torch.nn as nn


class QueryBuilder(nn.Module):
    """
    Builds Query vector: Q = q_task + alpha_view * q_view
    
    Args:
        d_task: Task embedding dimension
        view_dim: View metadata dimension (typically 8-10)
    """
    
    def __init__(self, d_task: int, view_dim: int):
        super().__init__()
        self.d_task = d_task
        self.view_dim = view_dim
        
        self.proj_task = nn.Linear(d_task, d_task)
        self.proj_view = nn.Linear(view_dim, d_task)
        self.norm = nn.LayerNorm(d_task)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable fusion."""
        nn.init.eye_(self.proj_task.weight)
        nn.init.zeros_(self.proj_task.bias)
        
        nn.init.xavier_uniform_(self.proj_view.weight, gain=0.1)
        nn.init.zeros_(self.proj_view.bias)
    
    def forward(
        self, 
        task_emb: torch.Tensor, 
        view_meta: torch.Tensor, 
        alpha_view: torch.Tensor
    ) -> torch.Tensor:
        """
        Build Query vector.
        
        Args:
            task_emb: [B, d_task] - Task embedding from nn.Embedding
            view_meta: [B, view_dim] - View/corruption metadata (NO task_id!)
            alpha_view: [B, 1] - Fusion strength from StateGate
        
        Returns:
            Q: [B, d_task] - Query vector for cross-attention
        """
        q_task = self.proj_task(task_emb)    # [B, d_task]
        q_view = self.proj_view(view_meta)   # [B, d_task]
        
        Q = q_task + alpha_view * q_view
        Q = self.norm(Q)
        
        return Q
