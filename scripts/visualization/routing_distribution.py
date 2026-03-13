#!/usr/bin/env python
"""
Analyze TAMoE (Task-Adaptive MoE) Routing Distribution

Collect task-expert routing statistics from a TAMoE checkpoint and visualize:
  1. Expert usage stacked bar chart (per task)
  2. Expert usage grouped bar chart (per task)

Routing is collected by hooking into TaskAdaptiveMoE.router.forward() to record
router_probs per layer. Statistics are averaged across all layers and batches.

Usage:
    python scripts/visualization/routing_distribution.py \
        --checkpoint_path saved_models/etth2/tamoe_pretrain/tamoe_multitask_cw512_patch12_stride12_d16_epochs100_all_moe8_tasktoken_model1.pth \
        --task_token_path saved_models/etth2/tamoe_pretrain/tamoe_multitask_cw512_patch12_stride12_d16_epochs100_all_moe8_tasktoken_model1_task_token.pth \
        --output_dir analysis/etth2_tamoe/routing_analysis \
        --dataset etth2 \
        --num_batches -1 \
        --split train

Note: If --task_token_path is not provided, will auto-detect <checkpoint>_task_token.pth
"""

import os
import sys
import argparse
import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.models.tamoe_backbone import TAMoE, TransformerLayer
from src.models.task_adaptive_moe import TaskAdaptiveMoE
from src.models.task_token.generator import TaskTokenGenerator
from src.models.task_token.feature_extractors import extract_global_desc_from_layer
from src.utils import load_model_config, load_model
from src.callback.multi_task_callback import build_view_meta_batch
from src.masking import (
    PatchMasking, BlockMasking,
    RandomFreqMasking, StructuredFreqMasking,
    DecompositionMasking,
)
from datautils import get_dls

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False
    _HAS_VIS = True
except ImportError:
    _HAS_VIS = False
    print("Warning: matplotlib not available, visualization will be skipped")


TASK_ID_MAP = {1: "PM", 2: "MPM", 3: "RFM", 4: "SFM", 5: "DM", 6: "HM"}


# ─── Masking ──────────────────────────────────────────────────────────────────

def create_mask_strategies(patch_len: int, stride: int, device: str = 'cuda'):
    base = {'patch_len': patch_len, 'stride': stride}
    strategies = {
        'PM':  PatchMasking(**base, mask_ratio=0.4),
        'MPM': BlockMasking(**base, mask_ratio=0.4),
        'RFM': RandomFreqMasking(**base, mask_ratio=0.3),
        'SFM': StructuredFreqMasking(**base, tau=0.5, mask_band='random'),
        'DM':  DecompositionMasking(**base, trend_mask_ratio=0.2, residual_mask_ratio=0.5),
        'HM':  PatchMasking(**base, mask_ratio=0.4),
    }
    for s in strategies.values():
        s.to(device)
    return strategies


def apply_task_masking(xb_patched: torch.Tensor, task_id: int, strategies: dict,
                       patch_len: int, stride: int) -> torch.Tensor:
    """Apply task-specific masking. xb_patched: [B, num_patch, n_vars, patch_len]"""
    task_name = TASK_ID_MAP.get(task_id, 'PM')
    if task_name not in strategies:
        return xb_patched

    B, num_patch, n_vars, pl = xb_patched.shape
    seq_len = (num_patch - 1) * stride + patch_len
    device = xb_patched.device

    # Reconstruct time series from patches (average overlapping regions)
    xb_recon = torch.zeros(B, seq_len, n_vars, device=device, dtype=xb_patched.dtype)
    overlap = torch.zeros(B, seq_len, n_vars, device=device)
    for i in range(num_patch):
        s, e = i * stride, i * stride + patch_len
        xb_recon[:, s:e, :] += xb_patched[:, i, :, :].transpose(1, 2)
        overlap[:, s:e, :] += 1
    xb_recon = xb_recon / (overlap + 1e-8)

    masked_view = strategies[task_name].make_view(xb_recon)
    return masked_view['x_in'].view(B, num_patch, n_vars, patch_len)


def create_patches(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """[B, seq_len, n_vars] -> [B, num_patch, n_vars, patch_len]"""
    return x.transpose(1, 2).unfold(2, patch_len, stride).permute(0, 2, 1, 3)


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_task_token_generator(tt_gen_path: str, config: dict,
                               device: str = 'cuda') -> Optional[TaskTokenGenerator]:
    if tt_gen_path is None or not os.path.exists(tt_gen_path):
        print(f"[TAMoE Analysis] TaskTokenGenerator not found: {tt_gen_path}")
        return None

    # Load config from JSON sidecar if available
    config_path = tt_gen_path.replace('.pth', '.json')
    tt_cfg = {
        'd_task': config.get('d_task', 16),
        'd_model': config.get('d_model', 16),
        'view_dim': 8,
        'global_desc_dim': 16,
        'nhead': config.get('n_heads', 4),
        'dropout': 0.1,
        'use_cr': True,
        'use_vwr': True,
        'vwr_beta': 0.5,
    }
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            tt_cfg.update(json.load(f))

    encoder = TaskTokenGenerator(**tt_cfg)
    state = torch.load(tt_gen_path, map_location='cpu')
    encoder.load_state_dict(state, strict=False)
    encoder = encoder.to(device).eval()
    print(f"[TAMoE Analysis] TaskTokenGenerator loaded from {tt_gen_path}")
    return encoder


def inject_task_token_generator(model: TAMoE, tt_gen: TaskTokenGenerator,
                                device: str = 'cuda', n_vars: int = 7):
    """Inject task_token_provider into all TransformerLayer modules."""
    current_task_id_ref = [None]
    current_x_input_ref = [None]

    def layer_provider(layer_module, task_id=None, current_layer_hidden=None,
                       current_layer_attn=None, cls_attn_vec=None):
        task_id = current_task_id_ref[0] if current_task_id_ref[0] is not None else task_id
        if task_id is None or current_layer_hidden is None:
            return None
        if isinstance(task_id, str):
            task_id = {v: k for k, v in TASK_ID_MAP.items()}.get(task_id, 1)

        B_flat, L, D = current_layer_hidden.shape
        B = B_flat // n_vars
        hidden_tokens = current_layer_hidden.view(B, n_vars, L, D).mean(dim=1)

        attn_weights = None
        if current_layer_attn is not None:
            H = current_layer_attn.shape[1]
            attn_weights = current_layer_attn.view(B, n_vars, H, L, L).mean(dim=1)

        global_desc = extract_global_desc_from_layer(
            hidden=hidden_tokens, attention=attn_weights,
            padding_mask=None, global_desc_dim=tt_gen.global_desc_dim,
        )

        task_name = TASK_ID_MAP.get(task_id, 'PM')
        view_meta = build_view_meta_batch(
            batch_size=B, task_name=task_name, mask_ratio=0.4, device=device,
        )
        task_id_tensor = torch.full((B,), task_id, dtype=torch.long, device=device)
        outputs = tt_gen(
            task_id=task_id_tensor, view_meta=view_meta, global_desc=global_desc,
            hidden_tokens=hidden_tokens, x_input=current_x_input_ref[0],
            padding_mask=None, return_all=True,
        )
        return {'z_task': outputs['z_task'], 'tau': outputs['tau']}

    injected = 0
    for name, module in model.named_modules():
        if isinstance(module, TransformerLayer):
            m = re.search(r"transformer\.layers\.(\d+)", name)
            if m:
                module._layer_idx = int(m.group(1))
            module.store_attn = True
            module.attn = None
            module.task_token_provider = layer_provider
            injected += 1

    print(f'[TAMoE Analysis] Injected task_token_provider to {injected} TransformerLayer(s)')
    return current_task_id_ref, current_x_input_ref


def load_model_and_checkpoint(checkpoint_path: str, tt_gen_path: Optional[str] = None,
                               device: str = 'cuda'):
    print(f"[TAMoE Analysis] Loading checkpoint: {checkpoint_path}")
    config = load_model_config(checkpoint_path)
    if config is None:
        raise ValueError(f"Cannot infer config from checkpoint: {checkpoint_path}")

    patch_len = config.get('patch_len', 12)
    stride = config.get('stride', 12)
    context_points = config.get('context_points', 512)
    num_patch = config.get('num_patch') or (context_points - patch_len) // stride + 1

    model = TAMoE(
        c_in=config.get('c_in', 7),
        target_dim=config.get('target_points', 96),
        patch_len=patch_len, stride=stride, num_patch=num_patch,
        n_layers=config.get('n_layers', 3), n_heads=config.get('n_heads', 4),
        d_model=config.get('d_model', 16), d_ff=config.get('d_ff', 128),
        head_type='pretrain',
        use_routed_expert=bool(config.get('use_routed_expert', True)),
        num_experts=config.get('num_experts', 8),
        use_shared_expert=bool(config.get('use_shared_expert', False)),
        moe_top_k=config.get('moe_top_k', 2),
        moe_router_fusion_mode=config.get('moe_router_fusion_mode', 'concat'),
        d_task=config.get('d_task', 16),
    )

    load_model(checkpoint_path, model, device=device, strict=False)
    model.eval()

    n_vars = config.get('c_in', 7)
    print(f"[TAMoE Analysis] Model loaded: {config.get('n_layers', 3)} layers, "
          f"{config.get('num_experts', 8)} experts, d_model={config.get('d_model', 16)}")

    current_task_id_ref = current_x_input_ref = None
    if tt_gen_path:
        tt_gen = load_task_token_generator(tt_gen_path, config, device)
        if tt_gen is not None:
            current_task_id_ref, current_x_input_ref = inject_task_token_generator(
                model, tt_gen, device, n_vars=n_vars)

    return model, config, current_task_id_ref, current_x_input_ref, n_vars


# ─── Routing Statistics Collection ───────────────────────────────────────────

def collect_routing_statistics(
    model: nn.Module, dataloader, task_ids: List[int],
    num_batches: int = -1, device: str = 'cuda',
    patch_len: int = 12, stride: int = 12,
    current_task_id_ref: Optional[list] = None,
    current_x_input_ref: Optional[list] = None,
) -> Dict[int, Dict]:
    """
    Collect expert routing statistics per task by hooking TaskAdaptiveMoE.router.forward().

    For each forward pass, records router_probs [B*n_vars, L, num_experts] from all
    MoE layers and averages them to get per-task expert usage distribution.
    """
    model.eval()
    task_stats = {tid: {'expert_usage_sum': None, 'count': 0} for tid in task_ids}
    batches_per_task = {tid: 0 for tid in task_ids}
    use_full_epoch = num_batches <= 0
    strategies = create_mask_strategies(patch_len=patch_len, stride=stride, device=device)

    # Hook storage: list of router_probs tensors captured during one forward pass
    _hook_buffer: List[torch.Tensor] = []

    def _router_hook(module, inputs, output):
        if isinstance(output, tuple) and len(output) == 3:
            router_probs = output[2]  # [B*n_vars, L, num_experts]
            _hook_buffer.append(router_probs.detach().mean(dim=(0, 1)).cpu().numpy())

    hooks = []
    for module in model.modules():
        if isinstance(module, TaskAdaptiveMoE):
            hooks.append(module.router.register_forward_hook(_router_hook))

    if not hooks:
        print("[TAMoE Analysis] Warning: No TaskAdaptiveMoE modules found in model!")

    with torch.no_grad():
        for xb, _ in dataloader:
            if not use_full_epoch and all(c >= num_batches for c in batches_per_task.values()):
                break

            xb = xb.to(device)
            xb_patched = create_patches(xb, patch_len=patch_len, stride=stride)
            num_patch = xb_patched.shape[1]

            for task_id in task_ids:
                if not use_full_epoch and batches_per_task[task_id] >= num_batches:
                    continue

                xb_masked = apply_task_masking(xb_patched, task_id, strategies,
                                               patch_len=patch_len, stride=stride)
                B = xb_masked.shape[0]
                task_labels = torch.full((B, num_patch), task_id, dtype=torch.long, device=device)

                if current_task_id_ref is not None:
                    current_task_id_ref[0] = task_id
                if current_x_input_ref is not None:
                    current_x_input_ref[0] = xb_masked

                _hook_buffer.clear()
                try:
                    model(xb_masked, task='recon', task_labels=task_labels)
                except Exception as e:
                    print(f"[TAMoE Analysis] Forward failed for task {task_id}: {e}")
                    continue

                if _hook_buffer:
                    # Average across all MoE layers: [num_experts]
                    avg_usage = np.mean(_hook_buffer, axis=0)
                    if task_stats[task_id]['expert_usage_sum'] is None:
                        task_stats[task_id]['expert_usage_sum'] = np.zeros_like(avg_usage)
                    task_stats[task_id]['expert_usage_sum'] += avg_usage
                    task_stats[task_id]['count'] += 1

                batches_per_task[task_id] += 1

    for h in hooks:
        h.remove()

    for tid in task_stats:
        c = task_stats[tid]['count']
        task_stats[tid]['expert_usage'] = (
            task_stats[tid]['expert_usage_sum'] / c if c > 0 else None
        )
        if c == 0:
            print(f"[TAMoE Analysis] Warning: no stats for task {tid}")

    return task_stats


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_expert_usage_stacked(task_expert_matrix: np.ndarray, task_ids: List[int],
                               output_path: str, title: str = "Expert Usage Distribution per Task"):
    if not _HAS_VIS:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    task_names = [TASK_ID_MAP.get(tid, f"Task{tid}") for tid in task_ids]
    num_experts = task_expert_matrix.shape[1]
    x = np.arange(len(task_names))
    bottom = np.zeros(len(task_names))
    colors = plt.cm.Set3(np.linspace(0, 1, num_experts))

    for ei in range(num_experts):
        ax.bar(x, task_expert_matrix[:, ei], bottom=bottom, label=f'Expert {ei}',
               color=colors[ei], edgecolor='white', linewidth=0.5)
        bottom += task_expert_matrix[:, ei]

    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Expert Usage Ratio', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(task_names)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[TAMoE Analysis] Saved: {output_path}")


def plot_expert_usage_grouped(task_expert_matrix: np.ndarray, task_ids: List[int],
                               output_path: str, title: str = "Expert Usage by Task (Grouped)"):
    if not _HAS_VIS:
        return
    fig, ax = plt.subplots(figsize=(14, 6))
    task_names = [TASK_ID_MAP.get(tid, f"Task{tid}") for tid in task_ids]
    num_experts = task_expert_matrix.shape[1]
    x = np.arange(len(task_names))
    bar_width = 0.8 / num_experts
    colors = plt.cm.Set3(np.linspace(0, 1, num_experts))

    for ei in range(num_experts):
        offset = (ei - num_experts / 2 + 0.5) * bar_width
        ax.bar(x + offset, task_expert_matrix[:, ei], bar_width, label=f'E{ei}',
               color=colors[ei], edgecolor='black', linewidth=0.3)

    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Expert Usage Ratio', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(task_names)
    ax.legend(loc='upper right', ncol=4, fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[TAMoE Analysis] Saved: {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze TAMoE routing distribution")
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--task_token_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='etth2')
    parser.add_argument('--num_batches', type=int, default=-1,
                        help='Batches per task (-1 = full epoch)')
    parser.add_argument('--context_points', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'valid'])
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else 'cpu'

    # Resolve task token path
    tt_gen_path = args.task_token_path
    if tt_gen_path is None:
        candidate = args.checkpoint_path.replace('.pth', '_task_token.pth')
        tt_gen_path = candidate if os.path.exists(candidate) else None

    model, config, task_id_ref, x_input_ref, n_vars = load_model_and_checkpoint(
        args.checkpoint_path, tt_gen_path=tt_gen_path, device=device)

    params = argparse.Namespace(
        dset=args.dataset, batch_size=args.batch_size,
        context_points=args.context_points, target_points=96,
        features='M', num_workers=0,
    )
    dls = get_dls(params)
    data_dl = dls.train if args.split == 'train' else dls.valid
    print(f"[TAMoE Analysis] Using {args.split} split")

    task_ids = [1, 2, 3, 4, 5, 6]
    patch_len = config.get('patch_len', 12)
    stride = config.get('stride', 12)

    task_stats = collect_routing_statistics(
        model, data_dl, task_ids, args.num_batches, device,
        patch_len=patch_len, stride=stride,
        current_task_id_ref=task_id_ref, current_x_input_ref=x_input_ref,
    )

    valid_ids = [tid for tid in task_ids if task_stats[tid]['expert_usage'] is not None]
    if not valid_ids:
        print("[TAMoE Analysis] No valid routing statistics collected!")
        return

    num_experts = len(task_stats[valid_ids[0]]['expert_usage'])
    matrix = np.array([task_stats[tid]['expert_usage'] for tid in valid_ids])
    matrix = matrix / (matrix.sum(axis=1, keepdims=True) + 1e-8)

    if _HAS_VIS:
        plot_expert_usage_stacked(
            matrix, valid_ids,
            os.path.join(args.output_dir, "routing_expert_usage_stacked.png"))
        plot_expert_usage_grouped(
            matrix, valid_ids,
            os.path.join(args.output_dir, "routing_expert_usage_grouped.png"))

    stats_path = os.path.join(args.output_dir, "routing_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump({
            'task_ids': valid_ids,
            'task_names': [TASK_ID_MAP.get(tid, f"Task{tid}") for tid in valid_ids],
            'num_experts': int(num_experts),
            'task_expert_matrix': matrix.tolist(),
        }, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

