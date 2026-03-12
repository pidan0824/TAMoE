"""Block Masking Strategy (MPM)."""

import torch
from typing import Optional, Dict, Any, List, Tuple

from .base import MaskingStrategy, MaskedView


class BlockMasking(MaskingStrategy):
    """
    Multi-patch Block Masking (MPM): masks contiguous blocks of patches.
    
    Args:
        patch_len: Length of each patch
        stride: Stride between patches
        mask_ratio: Target ratio of patches to mask (0.0-1.0)
        num_blocks: Number of blocks to mask (default: auto based on mask_ratio)
        min_block_len: Minimum length of each block (in patches)
        max_block_len: Maximum length of each block (in patches)
        min_gap: Minimum gap between blocks (in patches)
        mask_value: Value to fill masked patches (default: 0)
    """
    
    name = "MPM"
    requires_freq = False
    requires_decomp = False
    
    def __init__(
        self,
        patch_len: int,
        stride: int,
        mask_ratio: float = 0.4,
        num_blocks: Optional[int] = None,
        min_block_len: int = 2,
        max_block_len: Optional[int] = None,
        min_gap: int = 1,
        mask_value: float = 0.0,
    ):
        super().__init__(patch_len, stride, mask_ratio, mask_value)
        self.num_blocks = num_blocks
        self.min_block_len = min_block_len
        self.max_block_len = max_block_len
        self.min_gap = min_gap
    
    def make_view(
        self, 
        x: torch.Tensor, 
        ctx: Optional[Dict[str, Any]] = None
    ) -> MaskedView:
        """
        Create masked view with block masking.
        
        Args:
            x: Input time series. Shape: [B, T, C]
            ctx: Optional context dict with 'generator' for reproducibility
        
        Returns:
            MaskedView with:
                - x_in: Masked patches [B, N, C*patch_len]
                - mask: Boolean mask [B, N], True = masked
                - target_time: Original patches [B, N, C*patch_len]
        """
        ctx = ctx or {}
        generator = ctx.get('generator', None)

        B = x.shape[0]

        x_patch, num_patch = self.patchify(x)  # [B, N, C, patch_len]
        mask = self._generate_block_mask(B, num_patch, x.device, generator)  # [B, N]
        x_masked = self.apply_mask_to_patches(x_patch, mask)  # [B, N, C, patch_len]
        x_in = self.flatten_patches(x_masked)  # [B, N, C*patch_len]
        target_time = self.flatten_patches(x_patch)  # [B, N, C*patch_len]
        
        return MaskedView(
            x_in=x_in,
            mask=mask,
            target_time=target_time,
            target_freq=None,
            aux=None
        )
    
    def _generate_block_mask(
        self, 
        batch_size: int, 
        num_patch: int, 
        device: torch.device,
        generator: Optional[torch.Generator] = None
    ) -> torch.BoolTensor:
        """
        Generate block-structured mask.
        
        Strategy:
        1. Determine number of blocks and their lengths
        2. Sample block start positions with minimum gap constraint
        3. Create contiguous mask regions
        
        Args:
            batch_size: Batch size B
            num_patch: Number of patches N
            device: Target device
            generator: Optional random generator
        
        Returns:
            mask: Boolean tensor [B, N], True = masked
        """
        num_masked_target = int(num_patch * self.mask_ratio)
        max_block_len = self.max_block_len or max(self.min_block_len, num_patch // 4)
        
        if self.num_blocks is not None:
            num_blocks = self.num_blocks
        else:
            # Auto-determine: aim for reasonable block sizes
            avg_block_len = (self.min_block_len + max_block_len) // 2
            num_blocks = max(1, num_masked_target // avg_block_len)
        
        mask = torch.zeros(batch_size, num_patch, dtype=torch.bool, device=device)
        
        for b in range(batch_size):
            blocks = self._sample_blocks(
                num_patch, num_masked_target, num_blocks,
                self.min_block_len, max_block_len, self.min_gap,
                device, generator
            )
            for start, length in blocks:
                mask[b, start:start + length] = True
        
        return mask
    
    def _sample_blocks(
        self,
        num_patch: int,
        num_masked_target: int,
        num_blocks: int,
        min_block_len: int,
        max_block_len: int,
        min_gap: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None
    ) -> List[Tuple[int, int]]:
        """
        Sample block positions and lengths.
        
        Args:
            num_patch: Total number of patches
            num_masked_target: Target number of masked patches
            num_blocks: Number of blocks to create
            min_block_len: Minimum block length
            max_block_len: Maximum block length
            min_gap: Minimum gap between blocks
            device: Device for random generation
            generator: Optional random generator
        
        Returns:
            List of (start_position, length) tuples
        """
        blocks = []
        remaining_patches = num_masked_target
        available_positions = list(range(num_patch))
        
        for i in range(num_blocks):
            if remaining_patches <= 0 or len(available_positions) < min_block_len:
                break

            # Build contiguous available segments so a sampled block never jumps across gaps.
            segments = []
            seg_start = available_positions[0]
            seg_end = seg_start
            for pos in available_positions[1:]:
                if pos == seg_end + 1:
                    seg_end = pos
                else:
                    segments.append((seg_start, seg_end))
                    seg_start = seg_end = pos
            segments.append((seg_start, seg_end))

            segment_candidates = []
            for seg_start, seg_end in segments:
                seg_len = seg_end - seg_start + 1
                candidate_len = min(max_block_len, remaining_patches, seg_len)
                if candidate_len >= min_block_len:
                    segment_candidates.append((seg_start, seg_end, candidate_len))

            if not segment_candidates:
                break

            segment_idx = int(torch.randint(0, len(segment_candidates), (1,), device=device, generator=generator).item())

            seg_start, seg_end, candidate_len = segment_candidates[segment_idx]
            block_len = int(torch.randint(min_block_len, candidate_len + 1, (1,), device=device, generator=generator).item())

            max_start = seg_end - block_len + 1
            start_pos = int(torch.randint(seg_start, max_start + 1, (1,), device=device, generator=generator).item())

            blocks.append((start_pos, block_len))
            
            # Remove used positions and gap neighborhood in index space.
            masked_range = set(range(start_pos - min_gap, start_pos + block_len + min_gap))
            available_positions = [pos for pos in available_positions if pos not in masked_range]
            
            remaining_patches -= block_len
        
        return blocks

