"""Patch Masking Strategy (PM)."""

import torch
from typing import Optional, Dict, Any

from .base import MaskingStrategy, MaskedView


class PatchMasking(MaskingStrategy):
    """Random Patch Masking (PM): masks random patches, loss on masked only."""

    name = "PM"
    requires_freq = False
    requires_decomp = False

    def make_view(
        self,
        x: torch.Tensor,
        ctx: Optional[Dict[str, Any]] = None
    ) -> MaskedView:
        ctx = ctx or {}
        generator = ctx.get('generator', None)

        B, T, C = x.shape

        x_patch, num_patch = self.patchify(x)
        mask = self.generate_random_mask(B, num_patch, x.device, generator=generator)
        x_masked = self.apply_mask_to_patches(x_patch, mask)

        x_in = self.flatten_patches(x_masked)
        target_time = self.flatten_patches(x_patch)

        return MaskedView(
            x_in=x_in,
            mask=mask,
            target_time=target_time,
            target_freq=None,
            aux=None
        )

