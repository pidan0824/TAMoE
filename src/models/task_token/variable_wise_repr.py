"""
Variable-wise Representation (VWR)
==========================================

Computes per-variable representations from input time series for task token generation.

Two implementations:
    - VWR (deterministic): FFT periodic features + spectral band features + time-domain stats
    - LearnableVWR: Conv1d encoder + cross-attention (Q=task_query, K=V=variable_tokens)

Output scaled by learnable beta in [0,1] to prevent VWR from bypassing content attention.

Input data note:
    During pretrain, x_input is the MASKED (corrupted) input. Different masking tasks
    produce different corruptions -> different features -> VWR implicitly carries
    task-related information. Combined with view_meta (which describes the corruption
    strategy), the model can leverage both signals for task-aware routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import math


# ---------------------------------------------------------------------------
# GatedFeatureRead
# ---------------------------------------------------------------------------

class GatedFeatureRead(nn.Module):
    """
    Gated cross-attention read from per-variable features.

    Reads from per-variable features [B, C, d_feat] via cross-attention,
    then gates the result with g in [0,1] computed from (Q, feature_scale).

    Args:
        d_task: Query/output dimension
        d_feat: Feature dimension per variable
        hidden_dim: Hidden dimension for gate MLP
    """

    def __init__(self, d_task: int, d_feat: int, hidden_dim: int = 32):
        super().__init__()
        self.d_task = d_task
        self.d_feat = d_feat

        self.Wk = nn.Linear(d_feat, d_task)
        self.Wv = nn.Linear(d_feat, d_task)

        # gate input = concat(Q, feature_scale) so gate responds to both
        # query content and feature magnitude
        self.gate = nn.Sequential(
            nn.Linear(d_task + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.Wk.weight, gain=0.1)
        nn.init.zeros_(self.Wk.bias)
        nn.init.xavier_uniform_(self.Wv.weight, gain=0.1)
        nn.init.zeros_(self.Wv.bias)

        # gate[-2] bias=0 → sigmoid(0)=0.5
        nn.init.zeros_(self.gate[-2].bias)
        nn.init.xavier_uniform_(self.gate[-2].weight, gain=0.1)

    def forward(
        self,
        Q: torch.Tensor,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Q: [B, d_task] query vector
            features: [B, C, d_feat] per-variable features

        Returns:
            out: [B, d_task] gated feature-read output
            gate: [B, 1] gate value for monitoring
        """
        B, C, _ = features.shape

        K = self.Wk(features)                                                # [B, C, d_task]
        V = self.Wv(features)                                                # [B, C, d_task]

        scores = torch.einsum('bd,bcd->bc', Q, K) / math.sqrt(self.d_task)  # [B, C]
        weights = F.softmax(scores, dim=-1)                                  # [B, C]
        feat_read = torch.einsum('bc,bcd->bd', weights, V)                   # [B, d_task]

        feat_scale = features.abs().mean(dim=(1, 2)).unsqueeze(-1)           # [B, 1]
        gate = self.gate(torch.cat([Q, feat_scale], dim=-1))                 # [B, 1]

        return gate * feat_read, gate


# ---------------------------------------------------------------------------
# Feature extractors 
# ---------------------------------------------------------------------------

class PeriodicFeatureExtractor:
    """
    FFT-based frequency features: top-k frequencies with magnitudes, phases, energy ratios.
    Features per frequency: freq, log_magnitude, sin(phase), cos(phase), power_ratio, relative_power.
    Output: [B, C, k*6]
    """

    def __init__(self, k: int = 4):
        self.k = k
        self.d_periodic = k * 6

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, C]
        Returns:
            [B, C, d_periodic]
        """
        B, S, C = x.shape
        device = x.device

        if S < 2:
            return torch.zeros(B, C, self.d_periodic, device=device)

        x_t = x.transpose(1, 2)                                                # [B, C, S]
        x_centered = x_t - x_t.mean(dim=-1, keepdim=True)

        fft_vals = torch.fft.rfft(x_centered, dim=-1)                          # [B, C, F]
        freqs = torch.fft.rfftfreq(S, d=1.0, device=device)

        magnitudes = fft_vals.abs()[:, :, 1:]                                  # [B, C, F-1], skip DC
        phases = torch.angle(fft_vals)[:, :, 1:]
        freqs = freqs[1:]

        if magnitudes.shape[-1] == 0:
            return torch.zeros(B, C, self.d_periodic, device=device)

        k_eff = min(self.k, magnitudes.shape[-1])
        top_vals, top_idx = torch.topk(magnitudes, k=k_eff, dim=-1)            # [B, C, k]

        top_freqs = freqs[top_idx]                                              # [B, C, k]
        top_phases = torch.gather(phases, dim=-1, index=top_idx)                # [B, C, k]

        total_power = magnitudes.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        mean_power = magnitudes.mean(dim=-1, keepdim=True).clamp_min(1e-8)

        features = torch.stack([
            top_freqs,
            torch.log1p(top_vals),
            torch.sin(top_phases),
            torch.cos(top_phases),
            top_vals / total_power,
            top_vals / mean_power,
        ], dim=-1)                                                              # [B, C, k, 6]

        features = features.view(B, C, -1)                                     # [B, C, k*6]

        if k_eff < self.k:
            features = F.pad(features, (0, (self.k - k_eff) * 6))

        return features


class SpectralFeatureExtractor:
    """
    Spectral band energy + distribution statistics.
    Output: [B, C, num_bands + 8]
    """

    def __init__(self, num_bands: int = 4):
        self.num_bands = num_bands
        self.d_spectral = num_bands + 8

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, C]
        Returns:
            [B, C, d_spectral]
        """
        B, S, C = x.shape
        device = x.device

        if S < 4:
            return torch.zeros(B, C, self.d_spectral, device=device)

        x_t = x.transpose(1, 2)                                                # [B, C, S]
        x_centered = x_t - x_t.mean(dim=-1, keepdim=True)

        fft_vals = torch.fft.rfft(x_centered, dim=-1)
        power = (fft_vals.conj() * fft_vals).real[:, :, 1:]                    # [B, C, F-1], skip DC

        F_dim = power.shape[-1]
        if F_dim == 0:
            return torch.zeros(B, C, self.d_spectral, device=device)

        band_size = F_dim // self.num_bands
        band_energies = []
        for i in range(self.num_bands):
            start = i * band_size
            end = (i + 1) * band_size if i < self.num_bands - 1 else F_dim
            band_energies.append(power[:, :, start:end].sum(dim=-1))
        band_energies = torch.stack(band_energies, dim=-1)                      # [B, C, num_bands]

        total_power = power.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        band_energies_norm = band_energies / total_power

        freqs = torch.arange(1, F_dim + 1, device=device, dtype=power.dtype)
        power_norm = power / total_power
        centroid = (power_norm * freqs).sum(dim=-1) / F_dim                     # [B, C]

        spread = torch.sqrt(
            ((freqs - centroid.unsqueeze(-1) * F_dim) ** 2 * power_norm).sum(dim=-1)
        ) / F_dim

        log_power = torch.log(power.clamp_min(1e-10))
        geo_mean = torch.exp(log_power.mean(dim=-1))
        arith_mean = power.mean(dim=-1).clamp_min(1e-8)
        flatness = geo_mean / arith_mean

        max_band_ratio = band_energies_norm.max(dim=-1).values

        power_std = power.std(dim=-1).clamp_min(1e-6)
        power_mean = power.mean(dim=-1, keepdim=True)
        power_skew = ((power - power_mean) ** 3).mean(dim=-1) / (power_std ** 3)
        power_kurt = ((power - power_mean) ** 4).mean(dim=-1) / (power_std ** 4) - 3

        spectral_stats = torch.stack([
            centroid,
            spread,
            flatness,
            max_band_ratio,
            power_std / (total_power.squeeze(-1) + 1e-8),
            power_skew.clamp(-5, 5),
            power_kurt.clamp(-5, 5),
            torch.log1p(total_power.squeeze(-1)),
        ], dim=-1)                                                              # [B, C, 8]

        return torch.cat([band_energies_norm, spectral_stats], dim=-1)          # [B, C, num_bands+8]


class StatFeatureExtractor:
    """
    Time-domain statistical features (12 dims):
        mean, std, min, max, range, normalized_range, slope, autocorr,
        zero_crossing_rate, log_energy, cv, upper_deviation.
    Output: [B, C, 12]
    """

    def __init__(self):
        self.d_stat = 12

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, C]
        Returns:
            [B, C, d_stat]
        """
        B, S, C = x.shape
        device = x.device

        if S < 2:
            return torch.zeros(B, C, self.d_stat, device=device)

        x_t = x.transpose(1, 2)                                                # [B, C, S]

        mean = x_t.mean(dim=-1)                                                # [B, C]
        std = x_t.std(dim=-1).clamp_min(1e-6)
        x_min = x_t.min(dim=-1).values
        x_max = x_t.max(dim=-1).values
        x_range = x_max - x_min
        range_norm = x_range / std

        t = torch.arange(S, device=device, dtype=x.dtype).view(1, 1, -1)
        t_centered = t - t.mean()
        x_centered = x_t - mean.unsqueeze(-1)
        slope = (x_centered * t_centered).sum(dim=-1) / (t_centered ** 2).sum().clamp_min(1e-8)

        x_lag = x_t[:, :, 1:]
        x_base = x_t[:, :, :-1]
        autocorr = ((x_lag - x_lag.mean(dim=-1, keepdim=True)) *
                    (x_base - x_base.mean(dim=-1, keepdim=True))).sum(dim=-1)
        autocorr = autocorr / (x_lag.std(dim=-1) * x_base.std(dim=-1) * (S - 1) + 1e-8)

        signs = torch.sign(x_centered)
        zero_crossings = (signs[:, :, 1:] != signs[:, :, :-1]).float().sum(dim=-1) / (S - 1)

        energy = (x_t ** 2).mean(dim=-1)
        cv = std / (mean.abs() + 1e-8)

        return torch.stack([
            mean,
            std,
            x_min,
            x_max,
            x_range,
            range_norm.clamp(-10, 10),
            slope.clamp(-10, 10),
            autocorr.clamp(-1, 1),
            zero_crossings,
            torch.log1p(energy),
            cv.clamp(0, 10),
            (x_max - mean) / std,
        ], dim=-1)                                                              # [B, C, 12]


# ---------------------------------------------------------------------------
# LearnableVWR
# ---------------------------------------------------------------------------

class LearnableVWR(nn.Module):
    """
    Learnable Variable-wise Representation via Conv1d encoding + cross-attention.

    Each variable's time series is encoded into a token [d_task] via Conv1d,
    then cross-attention reads: Q=task_query, K=V=variable_tokens.

    Args:
        d_task: Query/output dimension
        nhead: Number of attention heads
        beta: Output scale factor
    """

    def __init__(
        self,
        d_task: int,
        nhead: int = 4,
        dropout: float = 0.1,
        beta: float = 0.5,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.d_task = d_task
        self._beta = nn.Parameter(torch.tensor(beta))
        self.hidden_dim = hidden_dim

        self.var_encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.var_proj = nn.Sequential(
            nn.Linear(hidden_dim, d_task),
            nn.LayerNorm(d_task),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_task,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.out_proj = nn.Linear(d_task, d_task)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @property
    def beta(self) -> torch.Tensor:
        return self._beta.clamp(0.0, 1.0)

    def encode_variables(self, x_input: torch.Tensor) -> torch.Tensor:
        """Encode each variable's time series into a token. Returns [B, C, d_task]."""
        if x_input.dim() == 4:
            B, num_patch, C, patch_len = x_input.shape
            x = x_input.transpose(1, 2).reshape(B, C, -1)                      # [B, C, S]
        else:
            B, S, C = x_input.shape
            x = x_input.permute(0, 2, 1).contiguous()                          # [B, C, S]

        B, C, S = x.shape
        x_flat = x.view(B * C, 1, S)                                           # [B*C, 1, S]
        encoded = self.var_encoder(x_flat)                                      # [B*C, hidden_dim, 1]
        var_tokens = self.var_proj(encoded.squeeze(-1))                         # [B*C, d_task]
        return var_tokens.view(B, C, self.d_task)                               # [B, C, d_task]

    def forward(
        self,
        Q: torch.Tensor,
        x_input: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            Q: [B, d_task]
            x_input: [B, num_patch, C, patch_len] or [B, S, C]
        Returns:
            vwr_out: [B, d_task]
            gate_info: dict
        """
        var_tokens = self.encode_variables(x_input)                             # [B, C, d_task]
        num_vars = var_tokens.shape[1]

        query = Q.unsqueeze(1)                                                  # [B, 1, d_task]
        attn_out, attn_weights = self.cross_attn(
            query=query, key=var_tokens, value=var_tokens,
        )                                                                       # [B, 1, d_task], [B, 1, C]

        vwr_out = self.beta * self.out_proj(attn_out.squeeze(1))                # [B, d_task]

        gate_info = {
            'beta': self.beta,
            'attn_weights': attn_weights.squeeze(1).detach(),                   # [B, C]
            'attn_entropy': -(attn_weights * (attn_weights + 1e-8).log()).sum(dim=-1).mean().detach(),
            'num_vars': num_vars,
        }

        return vwr_out, gate_info


# ---------------------------------------------------------------------------
# VWR 
# ---------------------------------------------------------------------------

class VWR(nn.Module):
    """
    Variable-wise Representation with deterministic feature extraction.

    Three feature slots (each gated independently):
        - Periodic: FFT top-k features  (d=k*6, default 24)
        - Spectral: Band energy + stats  (d=num_bands+8, default 12)
        - Stat: Time-domain statistics   (d=12)

    Output scaled by learnable beta in [0,1].

    Args:
        d_task: Query/output dimension
        d_periodic: Periodic feature dim (default 24)
        d_spectral: Spectral feature dim (default 12)
        d_stat: Stat feature dim (default 12)
        beta: Initial output scale (learnable)
        hidden_dim: Gate MLP hidden dim
        use_periodic/use_spectral/use_stat: Enable/disable slots
    """

    def __init__(
        self,
        d_task: int,
        d_periodic: int = 24,
        d_spectral: int = 12,
        d_stat: int = 12,
        beta: float = 0.5,
        hidden_dim: int = 32,
        use_periodic: bool = True,
        use_spectral: bool = True,
        use_stat: bool = True,
    ):
        super().__init__()
        self.d_task = d_task
        self._beta = nn.Parameter(torch.tensor(beta))
        self.use_periodic = use_periodic
        self.use_spectral = use_spectral
        self.use_stat = use_stat

        self.periodic_extractor = PeriodicFeatureExtractor(k=d_periodic // 6) if use_periodic else None
        self.spectral_extractor = SpectralFeatureExtractor(num_bands=4) if use_spectral else None
        self.stat_extractor = StatFeatureExtractor() if use_stat else None

        self.d_periodic = self.periodic_extractor.d_periodic if use_periodic else 0
        self.d_spectral = self.spectral_extractor.d_spectral if use_spectral else 0
        self.d_stat = self.stat_extractor.d_stat if use_stat else 0

        self.read_periodic = GatedFeatureRead(d_task, self.d_periodic, hidden_dim) if use_periodic else None
        self.read_spectral = GatedFeatureRead(d_task, self.d_spectral, hidden_dim) if use_spectral else None
        self.read_stat = GatedFeatureRead(d_task, self.d_stat, hidden_dim) if use_stat else None

    @property
    def beta(self) -> torch.Tensor:
        return self._beta.clamp(0.0, 1.0)

    def extract_features(self, x_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns dict of {slot_name: [B, C, d_slot]} feature tensors."""
        if x_input.dim() == 4:
            B, num_patch, C, patch_len = x_input.shape
            x = x_input.transpose(2, 3).reshape(B, -1, C)                      # [B, S, C]
        else:
            x = x_input                                                         # [B, S, C]

        features = {}
        if self.use_periodic:
            features['periodic'] = self.periodic_extractor(x)
        if self.use_spectral:
            features['spectral'] = self.spectral_extractor(x)
        if self.use_stat:
            features['stat'] = self.stat_extractor(x)
        return features

    def forward(
        self,
        Q: torch.Tensor,
        x_input: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            Q: [B, d_task]
            x_input: [B, num_patch, C, patch_len] or [B, S, C]
        Returns:
            vwr_out: [B, d_task]  (beta-scaled)
            gate_info: dict with per-slot gate values
        """
        features = self.extract_features(x_input)

        parts = []
        gate_info = {}

        if self.use_periodic:
            out_p, g_p = self.read_periodic(Q, features['periodic'])
            parts.append(out_p)
            gate_info['g_periodic'] = g_p

        if self.use_spectral:
            out_s, g_s = self.read_spectral(Q, features['spectral'])
            parts.append(out_s)
            gate_info['g_spectral'] = g_s

        if self.use_stat:
            out_t, g_t = self.read_stat(Q, features['stat'])
            parts.append(out_t)
            gate_info['g_stat'] = g_t

        gate_info['beta'] = self.beta

        if parts:
            vwr_out = self.beta * sum(parts)
        else:
            vwr_out = torch.zeros(Q.shape[0], self.d_task, device=Q.device, dtype=Q.dtype)

        return vwr_out, gate_info
