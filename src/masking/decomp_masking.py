"""Decomposition Masking Strategy (DM)."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Literal

from .base import MaskingStrategy, MaskedView


class DecompositionModule(nn.Module):
    """Time series decomposition via moving average: x = trend + residual."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        self.avg_pool = nn.AvgPool1d(
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose [B, T, C] -> (trend, residual), both [B, T, C]."""
        x_t = x.transpose(1, 2)          # [B, C, T]
        trend = self.avg_pool(x_t).transpose(1, 2)  # [B, T, C]
        residual = x - trend
        return trend, residual

class DecompositionMasking(MaskingStrategy):
    """Decomposition-aware Masking (DM): separate mask ratios for trend/residual components."""
    
    name = "DM"
    requires_freq = False
    requires_decomp = True
    
    def __init__(
        self,
        patch_len: int,
        stride: int,
        trend_mask_ratio: float = 0.2,
        residual_mask_ratio: float = 0.5,
        decomp_kernel_size: int = 25,
        mask_value: float = 0.0,
        recon_target: Literal['original', 'trend', 'residual'] = 'original',
    ):
        # Note: mask_ratio is not used in make_view, but required by base class
        super().__init__(patch_len, stride, mask_ratio=0.0, mask_value=mask_value)

        self.trend_mask_ratio = trend_mask_ratio
        self.residual_mask_ratio = residual_mask_ratio
        self.recon_target = recon_target

        self.decomp = DecompositionModule(kernel_size=decomp_kernel_size)
    
    def make_view(
        self, 
        x: torch.Tensor, 
        ctx: Optional[Dict[str, Any]] = None
    ) -> MaskedView:
        """Create masked view with decomposition masking. x: [B, T, C]."""
        ctx = ctx or {}
        generator = ctx.get('generator', None)

        B = x.shape[0]

        trend, residual = self.decomp(x)  # Both [B, T, C]
        
        x_patch, num_patch = self.patchify(x)  # [B, N, C, patch_len]
        trend_patch, _ = self.patchify(trend)
        residual_patch, _ = self.patchify(residual)
        
        trend_mask = self.generate_random_mask(
            B, num_patch, x.device, mask_ratio=self.trend_mask_ratio, generator=generator
        )
        residual_mask = self.generate_random_mask(
            B, num_patch, x.device, mask_ratio=self.residual_mask_ratio, generator=generator
        )

        combined_mask = trend_mask | residual_mask

        # summing independently masked components
        trend_masked = self.apply_mask_to_patches(trend_patch, trend_mask)
        residual_masked = self.apply_mask_to_patches(residual_patch, residual_mask)
        x_masked = trend_masked + residual_masked

        # Mask original patches with the combined mask.
        # x_masked = self.apply_mask_to_patches(x_patch, combined_mask)

        x_in = self.flatten_patches(x_masked)  # [B, N, C*patch_len]

        # Select reconstruction target
        target_patches = {
            'trend': trend_patch,
            'residual': residual_patch,
            'original': x_patch
        }
        target_time = self.flatten_patches(target_patches[self.recon_target])
        
        return MaskedView(
            x_in=x_in,
            mask=combined_mask,
            target_time=target_time,
            target_freq=None,
            aux={
                'decomp': {
                    'trend_mask': trend_mask,
                    'residual_mask': residual_mask,
                    'trend': trend_patch,
                    'residual': residual_patch,
                },
            }
        )
