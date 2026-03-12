"""
Base classes and interfaces for masking strategies.

Defines the unified interface that all masking strategies must implement,
ensuring consistent input/output contracts for the multi-task reconstruction framework.
"""

import torch
import torch.nn as nn
from typing import TypedDict, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod


def create_patch(xb, patch_len, stride):
    """Convert time series to patches, aligned from the end of the sequence.

    Args:
        xb: [bs, seq_len, n_vars]
        patch_len: patch length
        stride: stride between patches

    Returns:
        (xb_patch [bs, num_patch, n_vars, patch_len], num_patch)
    """
    seq_len = xb.shape[1]
    if seq_len < patch_len:
        raise ValueError(f"seq_len ({seq_len}) must be >= patch_len ({patch_len})")
    num_patch = (seq_len - patch_len) // stride + 1
    tgt_len = patch_len + stride * (num_patch - 1)
    s_begin = seq_len - tgt_len
    xb = xb[:, s_begin:, :]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)
    return xb, num_patch


class MaskedView(TypedDict, total=False):
    """
    Unified output format for all masking strategies.

    Required fields:
        x_in:  Input to encoder after masking.  [B, N, d_model] or [B, N, C*patch_len]
        mask:  Patch-level boolean mask.         [B, N], True = masked
        target_time: Time-domain reconstruction target. [B, N, C*patch_len] or [B, T, C]

    Optional fields:
        target_freq: Frequency-domain target.    [B, F, C] or [B, N, ...]
        aux: Auxiliary dict (e.g., freq_band for SFM, decomp targets for DM)
    """
    x_in: torch.Tensor
    mask: torch.BoolTensor
    target_time: torch.Tensor
    target_freq: Optional[torch.Tensor]
    aux: Optional[Dict[str, Any]]


class MaskingStrategy(nn.Module, ABC):
    """
    Abstract base class for all masking strategies.

    Subclasses must implement ``make_view`` to transform raw input into a MaskedView.

    Attributes:
        name: Strategy identifier (e.g., 'PM', 'MPM', 'RFM', 'SFM', 'HM', 'DM')
        requires_freq: Whether this strategy operates in frequency domain
        requires_decomp: Whether this strategy requires decomposition
    """

    name: str = "base"
    requires_freq: bool = False
    requires_decomp: bool = False

    def __init__(
        self,
        patch_len: int,
        stride: int,
        mask_ratio: float = 0.4,
        mask_value: float = 0.0,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value

    @abstractmethod
    def make_view(
        self,
        x: torch.Tensor,
        ctx: Optional[Dict[str, Any]] = None
    ) -> MaskedView:
        """
        Create a masked view from input time series.

        Args:
            x: Input time series [B, T, C].
            ctx: Optional context dict (e.g., pre-computed FFT, decomposition, epoch).

        Returns:
            MaskedView with x_in, mask, target_time, and optional fields.
        """
        pass

    def forward(
        self,
        x: torch.Tensor,
        ctx: Optional[Dict[str, Any]] = None
    ) -> MaskedView:
        return self.make_view(x, ctx)

    def patchify(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Convert time series to patches, aligned from the end of the sequence.

        Args:
            x: [B, T, C]

        Returns:
            (x_patch [B, N, C, patch_len], num_patch)
        """
        return create_patch(x, self.patch_len, self.stride)

    def apply_mask_to_patches(
        self,
        x_patch: torch.Tensor,
        mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Apply mask to patches: masked positions filled with ``mask_value``.

        Args:
            x_patch: [B, N, C, patch_len]
            mask: [B, N], True = masked

        Returns:
            x_masked: [B, N, C, patch_len]
        """
        x_masked = x_patch.clone()
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand_as(x_masked)
        x_masked.masked_fill_(mask_expanded, self.mask_value)
        return x_masked

    def generate_random_mask(
        self,
        batch_size: int,
        num_patch: int,
        device: torch.device,
        mask_ratio: Optional[float] = None,
        generator: Optional[torch.Generator] = None
    ) -> torch.BoolTensor:
        """Generate random patch mask [B, N] (True = masked), vectorized.

        Args:
            batch_size: Batch size B
            num_patch: Number of patches N
            device: Target device
            mask_ratio: Override self.mask_ratio if provided
            generator: Optional random generator for reproducibility

        Returns:
            mask: Boolean tensor [B, N], True = masked
        """
        ratio = mask_ratio if mask_ratio is not None else self.mask_ratio
        num_masked = int(num_patch * ratio)

        noise = torch.rand(batch_size, num_patch, device=device, generator=generator)

        ids_shuffle = torch.argsort(noise, dim=1)
        masked_indices = ids_shuffle[:, :num_masked]

        mask = torch.zeros(batch_size, num_patch, dtype=torch.bool, device=device)
        mask.scatter_(1, masked_indices, True)

        return mask

    def flatten_patches(self, x_patch: torch.Tensor) -> torch.Tensor:
        """Flatten [B, N, C, patch_len] -> [B, N, C*patch_len]."""
        B, N, C, P = x_patch.shape
        return x_patch.reshape(B, N, C * P)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"name={self.name}, "
                f"patch_len={self.patch_len}, "
                f"stride={self.stride}, "
                f"mask_ratio={self.mask_ratio})")
