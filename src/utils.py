"""
Shared utilities: reproducibility, path helpers, config loading, weight save/load/transfer.
"""
import os
import random
import json
import numpy as np
import torch
from torch import nn
from pathlib import Path
from typing import Optional, Dict, Any


# ─── Path utilities ───────────────────────────────────────────────────────────

def _ensure_pth_ext(name: str) -> str:
    return name if name.endswith('.pth') else name + '.pth'


def join_path_file(file, path, ext=''):
    "Return `path/file` if file is a string or a `Path`, file otherwise"
    if not isinstance(file, (str, Path)): return file
    if not isinstance(path, Path): path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path/f'{file}{ext}'


def build_weight_path(pretrained_model: Optional[str], dset: str, model_type: str) -> Optional[str]:
    """Resolve pretrained model path. Returns None if pretrained_model is falsy."""
    if not pretrained_model:
        return None
    if os.path.isabs(pretrained_model) or pretrained_model.startswith('saved_models/'):
        resolved = _ensure_pth_ext(pretrained_model)
    else:
        resolved = f'saved_models/{dset}/{model_type}/{_ensure_pth_ext(pretrained_model)}'
    return resolved


# ─── State dict utilities ─────────────────────────────────────────────────────

def _extract_state_dict(checkpoint):
    """Extract state_dict from checkpoint (handles common wrapper keys)."""
    if isinstance(checkpoint, dict):
        for key in ('model', 'model_state_dict', 'state_dict'):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


# ─── Config loading ───────────────────────────────────────────────────────────

def load_model_config(weight_path: Optional[str], verbose: bool = False) -> Optional[Dict[str, Any]]:
    """Load model config from JSON sidecar next to weight file.

    Tries ``<stem>.json`` then ``<stem>_config.json``.
    Returns None if no sidecar found (caller decides whether that's an error).
    """
    if not weight_path:
        return None
    stem = Path(weight_path).with_suffix('')
    for candidate in (stem.with_suffix('.json'), Path(f'{stem}_config.json')):
        if candidate.exists():
            with open(candidate, 'r') as f:
                cfg = json.load(f)
            if verbose:
                print(f"[Config] Loaded: {candidate}")
            return cfg
    return None


# ─── Save ─────────────────────────────────────────────────────────────────────

def save_model(path, model, opt, with_opt=True, pickle_protocol=2, model_config=None):
    "Save model (and optionally optimizer) state to `path`, plus config JSON sidecar."
    if opt is None: with_opt = False
    state = model.state_dict()
    if with_opt: state = {'model': state, 'opt': opt.state_dict()}
    torch.save(state, path, pickle_protocol=pickle_protocol)

    if model_config is not None:
        config_path = str(Path(path).with_suffix('')) + '_config.json'
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        print(f"[Save] Config -> {config_path}")


# ─── Load (for testing / resuming -- strict) ─────────────────────────────────

def load_model(path, model, opt=None, with_opt=False, device='cpu', strict=True):
    """Load model weights from checkpoint. Strict by default -- errors on mismatch."""
    state = torch.load(path, map_location=device)
    if not opt: with_opt = False
    model_state = state['model'] if with_opt else _extract_state_dict(state)

    target = model
    missing, unexpected = target.load_state_dict(model_state, strict=strict)
    if missing:
        raise RuntimeError(f"Missing keys when loading {path}: {missing}")
    if unexpected:
        print(f"[Load] Unexpected keys (ignored): {unexpected}")

    if with_opt: opt.load_state_dict(state['opt'])
    model.to(device)


# ─── Transfer (pretrain -> finetune -- skip head, strict on backbone) ─────────

def transfer_weights(weights_path, model, exclude_head=True, device='cpu', verbose: bool = True):
    """Load pretrained weights into model.

    When exclude_head=True, head.* keys from the checkpoint are skipped
    (finetune creates a new head). Errors if backbone keys are missing.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)
    src_state = _extract_state_dict(checkpoint)

    if exclude_head:
        src_state = {k: v for k, v in src_state.items() if 'head' not in k}

    missing, unexpected = model.load_state_dict(src_state, strict=False)

    if exclude_head:
        missing = [k for k in missing if 'head' not in k]

    if missing:
        raise RuntimeError(
            f"Missing backbone keys when loading pretrained weights from {weights_path}:\n"
            f"  {missing}"
        )
    if unexpected and verbose:
        print(f"[Transfer] Unexpected keys (ignored): {unexpected}")

    if verbose:
        print(f"[Transfer] Weights loaded from {weights_path}")
    model.to(device)
    return model


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed, deterministic=True):
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
