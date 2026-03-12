#!/usr/bin/env python
"""
Visualize Dynamic Task Token (z_task) with t-SNE

Analyzes the dynamic z_task representation computed by TaskTokenGenerator.
z_task is the task token that is actually input to the MoE router.

Usage:
    python scripts/visualization/task_token_tsne.py \
        --checkpoint_path saved_models/etth2/.../model.pth \
        --output_dir analysis/etth2_tamoe/task_token_tsne \
        --dataset etth2 \
        --num_batches -1
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

# Add project root to path (3 levels up from TAMoE/scripts/visualization/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.models.tamoe_backbone import TAMoE, TransformerLayer
from src.models.task_token import TaskTokenGenerator
from src.callback.multi_task_callback import build_view_meta_batch, TASK_ID_MAP
from src.masking.base import create_patch
from src.utils import load_model_config
from datautils import get_dls

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False
    _HAS_VIS = True
except ImportError:
    _HAS_VIS = False
    print("Warning: matplotlib/seaborn/sklearn not available, visualization will be skipped")


# Task configurations — matches analyze_task_token_components.py
# Note: In fine-grained mode, task_id 0 is reserved, tasks use IDs 1-6
TASK_ID_MAP_VIS = {
    0: "Recon",  # Reserved (coarse mode uses this for all recon tasks)
    1: "PM",     # Patch Masking
    2: "MPM",    # Multi-scale Patch Masking
    3: "RFM",    # Random Frequency Masking
    4: "SFM",    # Structured Frequency Masking
    5: "DM",     # Decomposition Masking
    6: "HM"      # Holistic Masking
}
TASK_COLORS = {
    0: '#1f77b4',  # Recon - blue
    1: '#ff7f0e',  # PM - orange
    2: '#2ca02c',  # MPM - green
    3: '#d62728',  # RFM - red
    4: '#9467bd',  # SFM - purple
    5: '#8c564b',  # DM - brown
    6: '#e377c2',  # HM - pink
}

# Task-specific view_meta parameters
TASK_PARAMS = {
    'PM':  {'mask_ratio': 0.4},
    'MPM': {'mask_ratio': 0.4},
    'RFM': {'mask_ratio': 0.3},
    'SFM': {'mask_ratio': 0.3, 'band_value': 0.5},
    'DM':  {'mask_ratio': 0.35, 'trend_mask_ratio': 0.2, 'residual_mask_ratio': 0.5},
    'HM':  {'mask_ratio': 0.4},
}


def _extract_hidden_tokens_all_layers(
    model: TAMoE,
    xb_patched: torch.Tensor,
    task_id: int,
    num_patch: int,
    n_vars: int,
    d_model: int,
    device: str,
    layer_indices: Optional[List[int]] = None,
) -> Dict[int, torch.Tensor]:
    """Extract hidden tokens from specified TransformerLayers via forward hooks.

    Args:
        layer_indices: List of layer indices to extract (0-indexed). If None, extract all layers.

    Returns:
        Dict mapping layer_idx -> hidden_tokens [B, num_patch, d_model]
    """
    B = xb_patched.shape[0]
    hidden_states_by_layer = {}

    layers = []
    for name, module in model.named_modules():
        if isinstance(module, TransformerLayer):
            layers.append(module)

    if layer_indices is None:
        target_layers = list(enumerate(layers))
    else:
        target_layers = [(idx, layers[idx]) for idx in layer_indices if idx < len(layers)]

    hooks = []
    for layer_idx, layer_module in target_layers:
        def make_hook_fn(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states_by_layer[idx] = output[0].detach()
                else:
                    hidden_states_by_layer[idx] = output.detach()
            return hook_fn

        hook = layer_module.register_forward_hook(make_hook_fn(layer_idx))
        hooks.append(hook)

    try:
        task_labels_batch = torch.full(
            (B, num_patch), task_id, dtype=torch.long, device=device
        )
        _ = model(xb_patched, task='recon', task_labels=task_labels_batch)
    except Exception as e:
        print(f"    Warning: Forward pass failed for task {task_id}: {e}")
        return {idx: torch.zeros(B, num_patch, d_model, device=device) for idx, _ in target_layers}
    finally:
        for hook in hooks:
            hook.remove()

    # Process hidden states: [B*n_vars, L, D] -> [B, num_patch, d_model]
    result = {}
    for layer_idx, hidden_flat in hidden_states_by_layer.items():
        B_flat, L, D = hidden_flat.shape
        B_real = B_flat // n_vars
        hidden_tokens = hidden_flat.reshape(B_real, n_vars, L, D).mean(dim=1)
        result[layer_idx] = hidden_tokens

    # Fill missing layers with zeros
    for idx, _ in target_layers:
        if idx not in result:
            result[idx] = torch.zeros(B, num_patch, d_model, device=device)

    return result


def _compute_ztask_for_task(
    model: TAMoE,
    tt_gen: TaskTokenGenerator,
    xb: torch.Tensor,
    xb_patched: torch.Tensor,
    task_id: int,
    task_name: str,
    config: dict,
    device: str,
    layer_indices: Optional[List[int]] = None,
) -> Dict[int, np.ndarray]:
    """Compute z_task for a single task on a batch, for specified layers.

    Args:
        layer_indices: List of layer indices (0-indexed). If None, only extract layer 0.

    Returns:
        Dict mapping layer_idx -> z_task [B, d_task] numpy array
    """
    B = xb.shape[0]
    num_patch = xb_patched.shape[1]
    n_vars = config.get('c_in', 7)
    d_model = config.get('d_model', 128)

    # Default to first layer only
    if layer_indices is None:
        layer_indices = [0]

    # Extract hidden tokens from specified layers
    hidden_tokens_by_layer = _extract_hidden_tokens_all_layers(
        model, xb_patched, task_id, num_patch, n_vars, d_model, device, layer_indices
    )

    # Build view_meta with task-specific parameters (same for all layers)
    params = TASK_PARAMS.get(task_name, {'mask_ratio': 0.4})
    view_meta = build_view_meta_batch(
        batch_size=B,
        task_name=task_name,
        mask_ratio=params.get('mask_ratio', 0.4),
        band_value=params.get('band_value', 0.5),
        trend_mask_ratio=params.get('trend_mask_ratio', 0.0),
        residual_mask_ratio=params.get('residual_mask_ratio', 0.0),
        device=device,
    )

    # All recon tasks share task_emb[0] (coarse mode).
    # z_task differences across tasks come from view_meta (corruption strategy),
    # not from task embedding. We label samples by task_id only for visualization grouping.
    task_id_tensor = torch.full((B,), 0, dtype=torch.long, device=device)

    # Compute z_task for each layer
    result = {}
    for layer_idx, hidden_tokens in hidden_tokens_by_layer.items():
        # Extract global_desc from hidden_tokens for StateGate (used in CR/VWR).
        global_desc_dim = tt_gen.global_desc_dim
        hidden_mean = hidden_tokens.mean(dim=1)  # [B, d_model]
        if hidden_mean.shape[-1] >= global_desc_dim:
            global_desc = hidden_mean[:, :global_desc_dim]
        else:
            # Pad if d_model < global_desc_dim (should not happen in practice)
            global_desc = torch.nn.functional.pad(
                hidden_mean, (0, global_desc_dim - hidden_mean.shape[-1])
            )

        # Generate z_task
        outputs = tt_gen(
            task_id=task_id_tensor,
            view_meta=view_meta,
            global_desc=global_desc,
            hidden_tokens=hidden_tokens,
            x_input=xb,
            padding_mask=None,
            return_all=True,
        )

        result[layer_idx] = outputs['z_task'].cpu().numpy()

    return result


def load_model_and_tt_gen(
    checkpoint_path: str,
    tt_gen_path: Optional[str] = None,
    device: str = 'cuda'
) -> Tuple[TAMoE, TaskTokenGenerator, dict]:
    """Load TAMoE model and TaskTokenGenerator."""
    print(f"[Analysis] Loading checkpoint: {checkpoint_path}")

    # Load config from JSON sidecar (src.utils.load_model_config)
    config = load_model_config(checkpoint_path, verbose=True)
    if config is None:
        raise FileNotFoundError(
            f"No config JSON found next to checkpoint: {checkpoint_path}"
        )

    # Calculate num_patch
    context_points = config.get('context_points', 512)
    patch_len = config.get('patch_len', 12)
    stride = config.get('stride', 12)
    num_patch = (context_points - patch_len) // stride + 1

    # Build TAMoE model
    model = TAMoE(
        c_in=config.get('c_in', 7),
        target_dim=config.get('target_points', 96),
        patch_len=patch_len,
        stride=stride,
        num_patch=num_patch,
        n_layers=config.get('n_layers', 3),
        d_model=config.get('d_model', 128),
        n_heads=config.get('n_heads', 4),
        d_ff=config.get('d_ff', 256),
        norm=config.get('norm', 'BatchNorm'),
        attn_dropout=config.get('attn_dropout', 0.0),
        dropout=config.get('dropout', 0.1),
        head_dropout=config.get('head_dropout', 0.0),
        head_type='pretrain',
        use_routed_expert=config.get('use_routed_expert', False),
        use_shared_expert=config.get('use_shared_expert', False),
        num_experts=config.get('num_experts', 8),
        moe_top_k=config.get('moe_top_k', 2),
        d_task=config.get('d_task', 16),
    )

    # Load model weights via src.utils.load_model (handles common checkpoint wrappers)
    from src.utils import load_model
    load_model(checkpoint_path, model, device=device, strict=False)
    model.eval()
    print(f"[Analysis] TAMoE model loaded")

    # Load TaskTokenGenerator
    if tt_gen_path is None:
        tt_gen_path = checkpoint_path.replace('.pth', '_task_token.pth')

    print(f"[Analysis] Loading TaskTokenGenerator from: {tt_gen_path}")

    if not os.path.exists(tt_gen_path):
        raise FileNotFoundError(f"TaskTokenGenerator checkpoint not found: {tt_gen_path}")

    # Load tt_gen config from JSON sidecar, fall back to defaults derived from model config
    tt_gen_config = {
        'd_task': config.get('d_task', 16),
        'd_model': config.get('d_model', 128),
        'view_dim': 8,
        'global_desc_dim': 16,
        'use_fine_grained_task_id': True,
        'vwr_beta': 0.5,
        'nhead': config.get('n_heads', 4),
        'dropout': 0.1,
        'use_cr': True,
        'use_vwr': True,
        'use_periodic': True,
        'use_spectral': True,
        'use_stat': True,
        'use_global_desc': True,
        'use_learnable_vwr': False,
    }
    saved_tt_gen_config = load_model_config(tt_gen_path, verbose=True)
    if saved_tt_gen_config is not None:
        tt_gen_config.update(saved_tt_gen_config)

    tt_gen = TaskTokenGenerator(
        d_task=tt_gen_config['d_task'],
        d_model=tt_gen_config['d_model'],
        view_dim=tt_gen_config['view_dim'],
        global_desc_dim=tt_gen_config['global_desc_dim'],
        use_fine_grained_task_id=tt_gen_config['use_fine_grained_task_id'],
        vwr_beta=tt_gen_config['vwr_beta'],
        nhead=tt_gen_config['nhead'],
        dropout=tt_gen_config['dropout'],
        use_cr=tt_gen_config['use_cr'],
        use_vwr=tt_gen_config['use_vwr'],
        use_periodic=tt_gen_config['use_periodic'],
        use_spectral=tt_gen_config['use_spectral'],
        use_stat=tt_gen_config['use_stat'],
        use_global_desc=tt_gen_config['use_global_desc'],
        use_learnable_vwr=tt_gen_config['use_learnable_vwr'],
    )

    load_model(tt_gen_path, tt_gen, device=device, strict=False)
    tt_gen.eval()
    print(f"[Analysis] TaskTokenGenerator loaded")

    return model, tt_gen, config


def collect_ztask(
    model: TAMoE,
    tt_gen: TaskTokenGenerator,
    dataloader,
    task_ids: List[int],
    config: dict,
    num_batches: int,
    device: str = 'cuda',
    layer_indices: Optional[List[int]] = None,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Collect z_task samples for each task and each layer.

    Args:
        layer_indices: List of layer indices to extract. If None, only extract layer 0.

    Returns:
        Nested dict: {layer_idx: {task_id: z_task_samples}}
    """
    print(f"\n[Analysis] Collecting z_task samples ({num_batches} batches)...")
    if layer_indices is None:
        layer_indices = [0]
        print(f"[Analysis] Extracting from layer 0 (first layer) only")
    else:
        print(f"[Analysis] Extracting from layers: {layer_indices}")

    patch_len = config.get('patch_len', 12)
    stride = config.get('stride', 12)

    # Initialize nested dict: {layer_idx: {task_id: [samples]}}
    data: Dict[int, Dict[int, list]] = {
        layer_idx: {tid: [] for tid in task_ids}
        for layer_idx in layer_indices
    }

    model.eval()
    tt_gen.eval()

    batch_count = 0
    max_batches = num_batches if num_batches > 0 else float('inf')

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_count >= max_batches:
                break

            if len(batch) == 3:
                xb, _, _ = batch
            else:
                xb, _ = batch

            xb = xb.to(device)
            B = xb.shape[0]

            xb_patched, _ = create_patch(xb, patch_len, stride)

            for task_id in task_ids:
                if task_id not in TASK_ID_MAP_VIS:
                    print(f"    Warning: Unknown task_id {task_id}, skipping")
                    continue
                task_name = TASK_ID_MAP_VIS[task_id]

                z_task_by_layer = _compute_ztask_for_task(
                    model, tt_gen, xb, xb_patched, task_id, task_name, config, device,
                    layer_indices=layer_indices
                )

                for layer_idx, z_task in z_task_by_layer.items():
                    if z_task is not None:
                        data[layer_idx][task_id].append(z_task)

            batch_count += 1
            if batch_count % 10 == 0:
                print(f"    Processed {batch_count}/{num_batches} batches")

    # Concatenate: {layer_idx: {task_id: np.ndarray}}
    result: Dict[int, Dict[int, np.ndarray]] = {}
    for layer_idx in layer_indices:
        result[layer_idx] = {}
        for tid in task_ids:
            if data[layer_idx][tid]:
                result[layer_idx][tid] = np.concatenate(data[layer_idx][tid], axis=0)
            else:
                result[layer_idx][tid] = np.array([])

    total_per_layer = {
        layer_idx: sum(len(v) for v in layer_data.values())
        for layer_idx, layer_data in result.items()
    }
    print(f"[Analysis] Collected samples per layer: {total_per_layer}")
    return result


def plot_tsne_only(
    z_task_by_task: Dict[int, np.ndarray],
    task_ids: List[int],
    output_dir: str,
    layer_idx: int = 0,
):
    """Plot t-SNE visualization of z_task distribution for a single layer.

    Args:
        layer_idx: Layer index (used for filename and title).
    """
    if not _HAS_VIS:
        print("[Analysis] Skipping t-SNE (matplotlib/sklearn not available)")
        return

    print(f"\n[Vis] Generating t-SNE for layer {layer_idx}...")

    all_z_task = []
    all_labels = []

    for tid in task_ids:
        if len(z_task_by_task[tid]) > 0:
            all_z_task.append(z_task_by_task[tid])
            all_labels.extend([tid] * len(z_task_by_task[tid]))

    if not all_z_task:
        print(f"[Vis] No data to plot for layer {layer_idx}.")
        return

    all_z_task = np.concatenate(all_z_task, axis=0)
    all_labels = np.array(all_labels)

    # Subsample if too many points
    max_points = 3000
    if len(all_z_task) > max_points:
        idx = np.random.choice(len(all_z_task), max_points, replace=False)
        all_z_task = all_z_task[idx]
        all_labels = all_labels[idx]

    # t-SNE: perplexity must be < n_samples and at least 5
    perplexity = max(5, min(30, len(all_z_task) // 4))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    z_task_2d = tsne.fit_transform(all_z_task)

    fig, ax = plt.subplots(figsize=(10, 8))

    for tid in task_ids:
        mask = all_labels == tid
        if mask.sum() > 0:
            ax.scatter(
                z_task_2d[mask, 0], z_task_2d[mask, 1],
                c=TASK_COLORS.get(tid, 'gray'),
                label=TASK_ID_MAP_VIS.get(tid, f'T{tid}'),
                alpha=0.7,
                s=30,
                edgecolors='white',
                linewidth=0.3,
            )

    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title(f'z_task Distribution (t-SNE) — Layer {layer_idx}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'ztask_tsne_layer{layer_idx}.png'
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Vis] Saved {fname}")


def main():
    parser = argparse.ArgumentParser(description="Visualize dynamic z_task with t-SNE (TAMoE)")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to TAMoE model .pth checkpoint')
    parser.add_argument('--tt_gen_path', type=str, default=None,
                        help='Path to TaskTokenGenerator .pth (default: <checkpoint>_task_token.pth)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for plots')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., etth2)')
    parser.add_argument('--num_batches', type=int, default=30,
                        help='Number of batches to collect z_task from (0 = all)')
    parser.add_argument('--context_points', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--split', type=str, default='train', choices=['train', 'valid'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--layers', type=str, default='first', choices=['first', 'all'],
                        help='Which layers to visualize: "first" (layer 0 only) or "all" (all layers)')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else 'cpu'

    model, tt_gen, config = load_model_and_tt_gen(args.checkpoint_path, args.tt_gen_path, device)

    config['context_points'] = args.context_points

    # Determine which layers to extract
    n_layers = config.get('n_layers', 3)
    if args.layers == 'all':
        layer_indices = list(range(n_layers))
    else:
        layer_indices = [0]

    # Load data
    print(f"\n[Analysis] Loading dataset: {args.dataset} ({args.split} split)")
    params = argparse.Namespace(
        dset=args.dataset,
        batch_size=args.batch_size,
        context_points=args.context_points,
        target_points=96,
        features='M',
        num_workers=0,
    )
    dls = get_dls(params)
    dataloader = dls.train if args.split == 'train' else dls.valid

    # Collect z_task for all reconstruction tasks and specified layers
    task_ids = [1, 2, 3, 4, 5, 6]
    z_task_by_layer_and_task = collect_ztask(
        model=model,
        tt_gen=tt_gen,
        dataloader=dataloader,
        task_ids=task_ids,
        config=config,
        num_batches=args.num_batches,
        device=device,
        layer_indices=layer_indices,
    )

    # Plot t-SNE for each layer
    for layer_idx in layer_indices:
        z_task_by_task = z_task_by_layer_and_task[layer_idx]

        # Filter tasks with no data
        valid_ids = [tid for tid in task_ids if len(z_task_by_task[tid]) > 0]
        if not valid_ids:
            print(f"[Analysis] Warning: No z_task collected for layer {layer_idx}. Skipping.")
            continue

        plot_tsne_only(z_task_by_task, valid_ids, args.output_dir, layer_idx=layer_idx)

    # Save basic stats (aggregate across all layers)
    stats = {
        'layers': layer_indices,
        'task_ids': task_ids,
        'task_names': [TASK_ID_MAP_VIS[tid] for tid in task_ids],
        'd_task': int(config.get('d_task', 16)),
        'num_samples_per_layer_and_task': {
            f'layer_{layer_idx}': {tid: len(z_task_by_layer_and_task[layer_idx][tid]) for tid in task_ids}
            for layer_idx in layer_indices
        },
    }
    stats_path = os.path.join(args.output_dir, 'ztask_tsne_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[Analysis] Stats saved: {stats_path}")
    print(f"[Analysis] All outputs in: {args.output_dir}")


if __name__ == '__main__':
    main()
