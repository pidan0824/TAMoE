"""
State Gate: global_desc -> (alpha_view, tau).

alpha_view ∈ [0,1] (Sigmoid) controls view contribution in QueryBuilder.
tau ∈ (0,+∞) (Softplus) controls MoE router temperature.

Ablation (use_global_desc=False): learnable scalar parameters replace the
MLP, isolating "input-dependent dynamics" from "having alpha/tau at all".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateGate(nn.Module):
    """
    Args:
        global_desc_dim: Dimension of global descriptor input
        hidden_dim: Hidden dimension for MLP
        use_global_desc: If False, use learnable scalar parameters instead of MLP

    Returns (from forward):
        alpha_view: [B, 1] view injection strength ∈ [0, 1]
        tau: [B, 1] router temperature ∈ (0, +∞)
    """

    def __init__(self, global_desc_dim: int, hidden_dim: int = 32, use_global_desc: bool = True):
        super().__init__()
        self.use_global_desc = use_global_desc

        if use_global_desc:
            self.alpha_net = nn.Sequential(
                nn.Linear(global_desc_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            self.tau_net = nn.Sequential(
                nn.Linear(global_desc_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Softplus()
            )
            self._init_weights()
        else:
            # sigmoid(0) = 0.5
            self.alpha_logit = nn.Parameter(torch.tensor(0.0))
            # softplus(0.5) ≈ 0.97
            self.tau_pre = nn.Parameter(torch.tensor(0.5))

    def _init_weights(self):
        """Initialize so that zero input → alpha≈0.5, tau≈1.0."""
        # [-2] is the last Linear (before Sigmoid/Softplus activation)
        # bias=0 → sigmoid(0) = 0.5
        nn.init.zeros_(self.alpha_net[-2].bias)
        nn.init.xavier_uniform_(self.alpha_net[-2].weight, gain=0.1)

        # bias=0.5 → softplus(0.5) ≈ 0.97 ≈ 1.0
        nn.init.constant_(self.tau_net[-2].bias, 0.5)
        nn.init.xavier_uniform_(self.tau_net[-2].weight, gain=0.1)

    def forward(self, global_desc: torch.Tensor):
        B = global_desc.shape[0]

        if not self.use_global_desc:
            alpha_view = torch.sigmoid(self.alpha_logit).expand(B, 1)
            tau = F.softplus(self.tau_pre).expand(B, 1)
            return alpha_view, tau

        alpha_view = self.alpha_net(global_desc)   # [B, 1]
        tau = self.tau_net(global_desc)             # [B, 1]
        return alpha_view, tau

    def get_stats(self, tensor: torch.Tensor, prefix: str) -> dict:
        """Get statistics for monitoring a tensor distribution.

        Args:
            tensor: [B, 1] tensor (e.g. alpha_view or tau)
            prefix: Key prefix, e.g. 'alpha' or 'tau'
        """
        return {
            f'{prefix}_mean': tensor.mean().item(),
            f'{prefix}_std': tensor.std().item(),
            f'{prefix}_min': tensor.min().item(),
            f'{prefix}_max': tensor.max().item(),
        }

    def get_learned_params(self) -> dict:
        """Get learned alpha/tau values (only meaningful when use_global_desc=False)."""
        if not self.use_global_desc:
            return {
                'alpha_learned': torch.sigmoid(self.alpha_logit).item(),
                'tau_learned': F.softplus(self.tau_pre).item(),
            }
        return {}
