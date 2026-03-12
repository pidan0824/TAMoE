"""
Frequency Domain Masking Strategies (RFM & SFM)

RFM - Random Frequency Masking:
    Randomly masks frequency components in the frequency domain.

SFM - Structured Frequency Masking:
    Masks specific frequency bands (high/low frequency) based on thresholds.

"""

import torch
from abc import abstractmethod
from typing import Optional, Dict, Any, Literal, Tuple

from .base import MaskingStrategy, MaskedView


class FreqMaskingBase(MaskingStrategy):
    """
    Base class for frequency-domain masking strategies.

    Implements the shared FFT pipeline:
        x -> FFT -> subclass mask -> iFFT -> patchify -> MaskedView

    Subclasses only need to implement ``_build_freq_mask``.
    """

    requires_freq = True
    requires_decomp = False

    def __init__(
        self,
        patch_len: int,
        stride: int,
        mask_ratio: float = 0.4,
        mask_value: float = 0.0,
        preserve_dc: bool = True,
    ):
        super().__init__(patch_len, stride, mask_ratio, mask_value)
        self.preserve_dc = preserve_dc

    @abstractmethod
    def _build_freq_mask(
        self,
        batch_size: int,
        num_freq: int,
        num_channels: int,
        device: torch.device,
        generator: Optional[torch.Generator],
    ) -> torch.BoolTensor:
        """Return boolean mask [B, F, C], True = masked."""

    def make_view(
        self,
        x: torch.Tensor,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> MaskedView:
        ctx = ctx or {}
        generator = ctx.get('generator', None)

        B, T, C = x.shape
        x_freq = torch.fft.rfft(x, dim=1)
        F = x_freq.shape[1]

        freq_mask = self._build_freq_mask(B, F, C, x.device, generator)

        x_freq_masked = x_freq.clone()
        x_freq_masked[freq_mask] = 0
        x_corrupted = torch.fft.irfft(x_freq_masked, n=T, dim=1)

        x_patch_corrupted, num_patch = self.patchify(x_corrupted)
        x_patch_original, _ = self.patchify(x)
        x_in = self.flatten_patches(x_patch_corrupted)
        target_time = self.flatten_patches(x_patch_original)

        # All-False mask: frequency corruption is the pretext, loss on every patch
        patch_mask = torch.zeros(B, num_patch, dtype=torch.bool, device=x.device)

        return MaskedView(
            x_in=x_in,
            mask=patch_mask,
            target_time=target_time,
            target_freq=x_freq,
            aux=None,
        )


class RandomFreqMasking(FreqMaskingBase):
    """
    Random Frequency Masking (RFM) Strategy - FRAUG Style.

    Randomly masks frequency components, then reconstructs in time domain.
    This encourages the model to learn robust frequency representations.

    Pipeline: x -> FFT -> random mask freq bins -> iFFT -> patchify -> encoder
    """

    name = "RFM"

    def __init__(
        self,
        patch_len: int,
        stride: int,
        mask_ratio: float = 0.3,
        mask_value: float = 0.0,
        preserve_dc: bool = True,
    ):
        super().__init__(patch_len, stride, mask_ratio, mask_value, preserve_dc)

    def _build_freq_mask(
        self,
        batch_size: int,
        num_freq: int,
        num_channels: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.BoolTensor:
        """Generate random frequency mask [B, F, C] (True = masked)."""
        start_idx = 1 if self.preserve_dc else 0
        maskable_bins = num_freq - start_idx

        mask_2d = self.generate_random_mask(batch_size, maskable_bins, device, self.mask_ratio, generator)

        if start_idx > 0:
            dc = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
            mask_2d = torch.cat([dc, mask_2d], dim=1)

        return mask_2d.unsqueeze(-1).expand(-1, -1, num_channels)


class StructuredFreqMasking(FreqMaskingBase):
    """
    Structured Frequency Masking (SFM) Strategy

    Masks a contiguous frequency band (low or high) delimited by threshold tau.
    tau controls the boundary; mask_ratio is unused (kept for interface compat).
    """

    name = "SFM"

    def __init__(
        self,
        patch_len: int,
        stride: int,
        mask_ratio: float = 0.5,
        mask_band: Literal['low', 'high', 'random'] = 'random',
        tau: float = 0.5,
        tau_range: Optional[Tuple[float, float]] = None,
        mask_value: float = 0.0,
        preserve_dc: bool = True,
    ):
        super().__init__(patch_len, stride, mask_ratio, mask_value, preserve_dc)
        self.mask_band = mask_band
        self.tau = tau
        self.tau_range = tau_range

    def _build_freq_mask(
        self,
        batch_size: int,
        num_freq: int,
        num_channels: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.BoolTensor:
        """Generate structured frequency mask [B, F, C] based on tau cutoff."""
        # Sample tau and band
        need_tau_rand = self.tau_range is not None
        need_band_rand = self.mask_band == 'random'

        num_rand = int(need_tau_rand) + int(need_band_rand)
        if num_rand > 0:
            rand_vals = torch.rand(num_rand, device=device, generator=generator)
            rand_idx = 0

            if need_tau_rand:
                tau = self.tau_range[0] + rand_vals[rand_idx].item() * (self.tau_range[1] - self.tau_range[0])
                rand_idx += 1
            else:
                tau = self.tau

            if need_band_rand:
                band = 'low' if rand_vals[rand_idx].item() < 0.5 else 'high'
            else:
                band = self.mask_band
        else:
            tau = self.tau
            band = self.mask_band

        # Generate mask
        start_idx = 1 if self.preserve_dc else 0
        cutoff_idx = int(tau * (num_freq - start_idx)) + start_idx
        cutoff_idx = max(start_idx, min(cutoff_idx, num_freq))

        freq_mask = torch.zeros(batch_size, num_freq, num_channels, dtype=torch.bool, device=device)

        if band == 'low':
            freq_mask[:, start_idx:cutoff_idx, :] = True
        else:
            freq_mask[:, cutoff_idx:, :] = True

        return freq_mask
