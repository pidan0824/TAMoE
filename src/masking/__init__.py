"""
Multi-Task Reconstruction Masking Strategies

Unified interface for masking strategies in time series self-supervised pretraining.

Strategies:
- PatchMasking (PM): Random patch masking (also used for HM with full-sequence loss)
- BlockMasking (MPM): Contiguous multi-patch block masking
- RandomFreqMasking (RFM): Random frequency component masking
- StructuredFreqMasking (SFM): Structured frequency band masking
- DecompositionMasking (DM): Trend/residual decomposition-aware masking
"""

from .base import MaskingStrategy, MaskedView
from .patch_masking import PatchMasking
from .block_masking import BlockMasking
from .freq_masking import RandomFreqMasking, StructuredFreqMasking
from .decomp_masking import DecompositionMasking
from .task_sampler import TaskSampler

__all__ = [
    'MaskingStrategy',
    'MaskedView',
    'PatchMasking',
    'BlockMasking',
    'RandomFreqMasking',
    'StructuredFreqMasking',
    'DecompositionMasking',
    'TaskSampler',
]
