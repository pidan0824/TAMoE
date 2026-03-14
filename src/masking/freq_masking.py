"""Frequency Domain Masking Strategies (RFM & SFM)."""

import torch
from abc import abstractmethod
from typing import Optional, Dict, Any, Literal, Tuple

from .base import MaskingStrategy, MaskedView


class FreqMaskingBase(MaskingStrategy):
    """Base for frequency-domain masking: x -> FFT -> subclass mask -> iFFT -> patchify -> MaskedView."""

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
    ) -> Tuple[torch.BoolTensor, Dict[str, Any]]:
        """Return (boolean mask [B, F, C] True=masked, extra aux info dict)."""

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

        freq_mask, extra_aux = self._build_freq_mask(B, F, C, x.device, generator)

        x_freq_masked = x_freq.clone()
        x_freq_masked[freq_mask] = 0
        x_corrupted = torch.fft.irfft(x_freq_masked, n=T, dim=1)

        x_patch_corrupted, num_patch = self.patchify(x_corrupted)
        x_patch_original, _ = self.patchify(x)
        x_in = self.flatten_patches(x_patch_corrupted)
        target_time = self.flatten_patches(x_patch_original)

        aux = {'freq_mask': freq_mask, **extra_aux}

        return MaskedView(
            x_in=x_in,
            target_time=target_time,
            target_freq=x_freq,
            aux=aux,
        )


class RandomFreqMasking(FreqMaskingBase):
    """Random Frequency Masking (RFM): randomly masks frequency bins."""

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
    ) -> Tuple[torch.BoolTensor, Dict[str, Any]]:
        """Generate random frequency mask [B, F, C] (True = masked)."""
        start_idx = 1 if self.preserve_dc else 0
        maskable_bins = num_freq - start_idx

        mask_2d = self.generate_random_mask(batch_size, maskable_bins, device, self.mask_ratio, generator)

        if start_idx > 0:
            dc = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
            mask_2d = torch.cat([dc, mask_2d], dim=1)

        return mask_2d.unsqueeze(-1).expand(-1, -1, num_channels), {}


class StructuredFreqMasking(FreqMaskingBase):
    """Structured Frequency Masking (SFM): masks a contiguous band (low/high) delimited by tau."""

    name = "SFM"

    def __init__(
        self,
        patch_len: int,
        stride: int,
        mask_band: Literal['low', 'high', 'random'] = 'random',
        tau: float = 0.5,
        tau_range: Optional[Tuple[float, float]] = None,
        mask_value: float = 0.0,
        preserve_dc: bool = True,
    ):
        # mask_ratio is not used by SFM (tau controls the cutoff), pass 0 to base
        super().__init__(patch_len, stride, 0.0, mask_value, preserve_dc)
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
    ) -> Tuple[torch.BoolTensor, Dict[str, Any]]:
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

        return freq_mask, {'band': band}
