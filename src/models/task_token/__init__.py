"""Task Token subsystem: produces z_task for MoE routing."""

from .generator import TaskTokenGenerator, TaskNorm
from .feature_extractors import extract_global_desc_from_layer
