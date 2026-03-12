"""
Feature extraction utilities: Extract statistical features from backbone layers

"""
import torch
from typing import Optional


def extract_global_desc_from_layer(
    hidden: Optional[torch.Tensor] = None,
    attention: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    global_desc_dim: int = 16,
) -> Optional[torch.Tensor]:
    """
    Extract global statistical features from backbone layer's hidden state and attention weights

    Args:
        hidden: [B, L, D] - Layer hidden state (averaged over variables before passing in)
        attention: [B, H, L, L] or [B, L, L] - Attention weights
        padding_mask: [B, L] - True indicates padding positions
        global_desc_dim: Output feature dimension (zero-padded if > 8)

    Returns:
        [B, global_desc_dim] - Global statistical features, containing:
            - Hidden features (4 dims): mean_norm, std_norm, delta_ht, slope
            - Attention features (4 dims): entropy_mean, entropy_std, future_mass, Lb_norm
            - Zero-padded to global_desc_dim if global_desc_dim > 8
    """
    if hidden is None and attention is None:
        return None

    if attention is not None:
        device = attention.device
        B = attention.shape[0]
    else:
        device = hidden.device
        B = hidden.shape[0]

    # Reduce multi-head attention to [B, L, L]
    A = None
    L = None
    if attention is not None:
        if attention.dim() == 4:       # [B, H, L, L]
            A = attention.mean(dim=1)  # [B, L, L]
        elif attention.dim() == 3:
            A = attention
        if A is not None:
            L = A.shape[1]

    if L is None and hidden is not None:
        L = hidden.shape[1]

    if L is None:
        return None

    # Align padding_mask length to L
    if padding_mask is not None:
        padding_mask = padding_mask.to(device)
        if padding_mask.shape[1] > L:
            padding_mask = padding_mask[:, :L]
        elif padding_mask.shape[1] < L:
            pad = torch.ones(B, L - padding_mask.shape[1], device=device, dtype=torch.bool)
            padding_mask = torch.cat([padding_mask, pad], dim=1)

    eps = 1e-8

    mean_norm = torch.zeros(B, device=device)
    std_norm = torch.zeros(B, device=device)
    delta_ht = torch.zeros(B, device=device)
    slope = torch.zeros(B, device=device)

    if hidden is not None:
        h_norm = hidden.norm(dim=-1)  # [B, L]

        if padding_mask is not None:
            h_norm = h_norm.masked_fill(padding_mask, 0.0)
            valid_counts = (~padding_mask).sum(dim=1, keepdim=True).float().clamp_min(1.0)  # [B, 1]
        else:
            valid_counts = torch.full((B, 1), L, device=device, dtype=torch.float)

        mean_norm = h_norm.sum(dim=1) / valid_counts.squeeze(1)  # [B]
        h_norm_centered = h_norm - mean_norm.unsqueeze(1)  # [B, L]
        if padding_mask is not None:
            h_norm_centered = h_norm_centered.masked_fill(padding_mask, 0.0)
        std_norm = ((h_norm_centered ** 2).sum(dim=1) / valid_counts.squeeze(1)).sqrt()  # [B]

        # Head vs tail norm difference, padding-aware
        K = max(L // 4, 1)
        if padding_mask is not None:
            head_mask = ~padding_mask[:, :K]  # [B, K], True = valid
            tail_mask = ~padding_mask[:, -K:]
            head_sum = (h_norm[:, :K] * head_mask).sum(dim=1)
            tail_sum = (h_norm[:, -K:] * tail_mask).sum(dim=1)
            head_count = head_mask.sum(dim=1).float().clamp_min(1.0)
            tail_count = tail_mask.sum(dim=1).float().clamp_min(1.0)
            delta_ht = tail_sum / tail_count - head_sum / head_count  # [B]
        else:
            delta_ht = h_norm[:, -K:].mean(dim=1) - h_norm[:, :K].mean(dim=1)

        # Least-squares slope of h_norm over positions
        pos = torch.arange(L, device=device, dtype=torch.float).unsqueeze(0).expand(B, -1)  # [B, L]
        if padding_mask is not None:
            pos = pos.masked_fill(padding_mask, 0.0)
        pos_mean = pos.sum(dim=1) / valid_counts.squeeze(1)  # [B]

        pos_centered = pos - pos_mean.unsqueeze(1)  # [B, L]
        if padding_mask is not None:
            pos_centered = pos_centered.masked_fill(padding_mask, 0.0)

        pos_var = (pos_centered ** 2).sum(dim=1) / valid_counts.squeeze(1)  # [B]
        cov = (pos_centered * h_norm_centered).sum(dim=1) / valid_counts.squeeze(1)  # [B]
        slope = torch.where(pos_var > eps, cov / (pos_var + eps), torch.zeros_like(cov))  # [B]

    entropy_mean = torch.zeros(B, device=device)
    entropy_std = torch.zeros(B, device=device)
    future_mass = torch.zeros(B, device=device)
    Lb_norm = torch.ones(B, device=device)

    if A is not None:
        if padding_mask is not None:
            valid = ~padding_mask  # [B, L]
        else:
            valid = torch.ones(B, L, device=device, dtype=torch.bool)

        valid_f = valid.float()
        valid_lengths = valid_f.sum(dim=1).clamp_min(1.0)  # [B]

        valid_expanded = valid_f.unsqueeze(-1) * valid_f.unsqueeze(1)  # [B, L, L]
        A_masked = A * valid_expanded
        A_norm = (A_masked / (A_masked.sum(dim=-1, keepdim=True) + eps)).clamp_min(0)

        ent_batch = -(A_norm * (A_norm + eps).log()).sum(dim=-1)  # [B, L]
        ent_masked = ent_batch * valid_f
        entropy_mean = ent_masked.sum(dim=1) / valid_lengths  # [B]
        ent_centered = (ent_batch - entropy_mean.unsqueeze(-1)) * valid_f
        entropy_std = (ent_centered.pow(2).sum(dim=1) / valid_lengths).clamp_min(eps).sqrt()  # [B]

        # Upper-triangle attention mass — measures how much each position looks ahead
        future_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        future_mask = future_mask.unsqueeze(0) & valid.unsqueeze(-1) & valid.unsqueeze(1)  # [B, L, L]
        future_count = future_mask.sum(dim=(1, 2)).float().clamp_min(1.0)
        future_mass = (A_norm * future_mask).sum(dim=(1, 2)) / future_count  # [B]

        Lb_norm = valid_lengths / max(L, 1)  # [B]

    elif hidden is not None:
        if padding_mask is not None:
            valid_counts_1d = (~padding_mask).sum(dim=1).float().clamp_min(1.0)  # [B]
        else:
            valid_counts_1d = torch.full((B,), L, device=device, dtype=torch.float)
        Lb_norm = valid_counts_1d / max(L, 1)  # [B]

    feats = torch.stack([mean_norm, std_norm, delta_ht, slope,
                         entropy_mean, entropy_std, future_mass, Lb_norm], dim=1)  # [B, 8]

    if feats.shape[1] < global_desc_dim:
        pad_size = global_desc_dim - feats.shape[1]
        feats = torch.cat([feats, torch.zeros(B, pad_size, device=device)], dim=1)
    elif feats.shape[1] > global_desc_dim:
        feats = feats[:, :global_desc_dim]

    return feats  # [B, global_desc_dim]
