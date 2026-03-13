# ══════════════════════════════════════════════════════════════════════════════
# Imports
# ══════════════════════════════════════════════════════════════════════════════

import os
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW

from src.models.tamoe_backbone import TAMoE
from src.learner import Learner
from src.utils import set_seed
from src.callback.core import *
from src.callback.tracking import *
from src.callback.transforms import *
from src.callback.multi_task_callback import MultiTaskReconCB
from src.callback.tensorboard_logger import TensorBoardCB, MultiTaskPretrainTensorBoardCB
from src.metrics import *
from datautils import get_dls

import argparse

# ══════════════════════════════════════════════════════════════════════════════
# CLI Arguments
# ══════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description='Multi-Task Reconstruction Pretraining')

# Dataset
parser.add_argument('--dset', type=str, default='etth1')
parser.add_argument('--context_points', type=int, default=512)
parser.add_argument('--target_points', type=int, default=96)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--features', type=str, default='M')

# Model
parser.add_argument('--patch_len', type=int, default=12)
parser.add_argument('--stride', type=int, default=12)
parser.add_argument('--revin', type=int, default=1)
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--d_model', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--head_dropout', type=float, default=0.2)

# Task Configuration
parser.add_argument('--task_mode', type=str, default='all',
                    choices=['all', 'time_only', 'freq_only', 'single'])
parser.add_argument('--single_task', type=str, default='PM',
                    choices=['PM', 'MPM', 'RFM', 'SFM', 'DM', 'HM'])

# Task Probabilities (for task_mode=all)
parser.add_argument('--pm_prob', type=float, default=0.18)
parser.add_argument('--mpm_prob', type=float, default=0.18)
parser.add_argument('--rfm_prob', type=float, default=0.16)
parser.add_argument('--sfm_prob', type=float, default=0.16)
parser.add_argument('--dm_prob', type=float, default=0.20)
parser.add_argument('--hm_prob', type=float, default=0.12)

# Masking Parameters
parser.add_argument('--pm_mask_ratio', type=float, default=0.4)
parser.add_argument('--mpm_mask_ratio', type=float, default=0.4)
parser.add_argument('--rfm_mask_ratio', type=float, default=0.3)
parser.add_argument('--sfm_tau', type=float, default=0.5)
parser.add_argument('--sfm_band', type=str, default='random', choices=['low', 'high', 'random'])
parser.add_argument('--dm_trend_mask_ratio', type=float, default=0.2)
parser.add_argument('--dm_residual_mask_ratio', type=float, default=0.5)
parser.add_argument('--hm_time_mask_ratio', type=float, default=0.4)

# Loss Weights
parser.add_argument('--time_loss_weight', type=float, default=1.0)
parser.add_argument('--decomp_loss_weight', type=float, default=0.5)

# MoE Parameters
# Configuration modes:
#   1. Standard FFN (use_routed_expert=0): No MoE, just standard FFN
#
#   2. MoE (only routed experts, use_routed_expert=1, use_shared_expert=0):
#      - Pretrain: router selects top-k experts
#      - Finetune: router selects top-k experts (aggregation_mode='router', auto-inferred)
#
#   3. Routed experts + Shared Expert (use_routed_expert=1, use_shared_expert=1, use_task_token=0):
#      - Pretrain: y = y_shared + y_routed (no alpha schedule)
#      - Finetune: y = y_shared + y_routed
#
#   4. Task-Adaptive MoE (use_routed_expert=1, use_shared_expert=1, use_task_token=1):
#      - Full task-adaptive architecture with task token for routing
#      - Pretrain: y = y_shared + alpha * y_routed (alpha schedule, task token guides routing)
#      - Finetune: y = y_shared only (aggregation_mode='shared_only', auto-inferred)
parser.add_argument('--use_routed_expert', type=int, default=0,
                    help='Enable MoE routed experts')
parser.add_argument('--use_shared_expert', type=int, default=0,
                    help='Enable MoE shared experts')
parser.add_argument('--num_experts', type=int, default=8,
                    help='Number of routed experts')
parser.add_argument('--moe_top_k', type=int, default=2,
                    help='Top-k routed experts for routing')
parser.add_argument('--moe_aux_loss_weight', type=float, default=0.0001,
                    help='Auxiliary loss weight for load balancing')
parser.add_argument('--moe_router_fusion_mode', type=str, default='concat',
                    choices=['concat', 'additive', 'multiplicative', 'none'],
                    help='How to fuse input and task token for routing')

# Task-Aware MoE specific parameters (requires use_routed_expert=1, use_shared_expert=1, use_task_token=1)
parser.add_argument('--moe_alpha_schedule', type=str, default='linear',
                    choices=['linear', 'fixed', 'plateau'],
                    help='Schedule for alpha (weight of routed experts)')
parser.add_argument('--moe_alpha_start', type=float, default=0.1,
                    help='Initial alpha value')
parser.add_argument('--moe_alpha_end', type=float, default=0.15,
                    help='Final alpha value')
parser.add_argument('--moe_alpha_plateau_start', type=int, default=50,
                    help='Epoch to start plateau (for plateau schedule)')
parser.add_argument('--moe_routed_l2_weight', type=float, default=1e-3,
                    help='L2 regularization weight for routed expert output')

# Task Token Generator (provides z_task to router for task-aware routing)
# Task-Aware MoE requires: use_routed_expert=1, use_shared_expert=1, use_task_token=1
parser.add_argument('--use_task_token', type=int, default=0,
                    help='Enable Task Token Generator (provides z_task to router)')
parser.add_argument('--d_task', type=int, default=16)
parser.add_argument('--vwr_beta', type=float, default=0.5,
                    help='VWR output scale')
parser.add_argument('--use_cr', type=int, default=1,
                    help='Enable Contextual Representation in Task Token')
parser.add_argument('--use_vwr', type=int, default=1,
                    help='Enable Variable-Wise Representation in Task Token. VWR: z_vise = beta * (periodic + spectral + stat)')
parser.add_argument('--use_periodic', type=int, default=1,
                    help='VWR periodic slot (FFT features)')
parser.add_argument('--use_spectral', type=int, default=1,
                    help='VWR spectral slot (band energies)')
parser.add_argument('--use_stat', type=int, default=1,
                    help='VWR stat slot (mean/std/slope)')
parser.add_argument('--use_global_desc', type=int, default=1,
                    help='Use global descriptor in StateGate')
parser.add_argument('--use_fine_grained_task_id', type=int, default=0,
                    help='Fine-grained task IDs (PM=1, MPM=2, ...) instead of shared recon=0')
parser.add_argument('--use_learnable_vwr', type=int, default=0,
                    help='Use learnable VWR (Conv encoder + cross-attention) instead of deterministic features')

# Training
parser.add_argument('--n_epochs_pretrain', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)

# Saving
parser.add_argument('--pretrained_model_id', type=int, default=1)
parser.add_argument('--model_type', type=str, default='multitask_recon')
parser.add_argument('--seed', type=int, default=2021)

# GPU
parser.add_argument('--gpu_ids', type=str, default='0')

args = parser.parse_args()

# ══════════════════════════════════════════════════════════════════════════════
# Configuration (derived paths, GPU, seed)
# ══════════════════════════════════════════════════════════════════════════════

if torch.cuda.is_available():
    gpu_id = int(args.gpu_ids.split(',')[0].strip()) if args.gpu_ids.strip() else 0
    torch.cuda.set_device(gpu_id)

args.save_path = f'saved_models/{args.dset}/{args.model_type}/'
os.makedirs(args.save_path, exist_ok=True)

parts = [
    'tamoe_multitask',
    f'cw{args.context_points}',
    f'patch{args.patch_len}',
    f'stride{args.stride}',
    f'd{args.d_model}',
    f'epochs{args.n_epochs_pretrain}',
    args.task_mode,
]
if args.task_mode == 'single':
    parts.append(args.single_task)
if args.use_routed_expert:
    parts.append(f'moe{args.num_experts}')
if args.use_task_token:
    parts.append('tasktoken')
parts.append(f'model{args.pretrained_model_id}')
args.save_pretrained_model = '_'.join(parts)

set_seed(args.seed, deterministic=False)
print(f'\n[Pretrain] Dataset: {args.dset}, Task: {args.task_mode}')
print(f'[Pretrain] Model: {args.save_pretrained_model}\n')


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

def get_model(c_in, verbose: bool = True):
    num_patch = (args.context_points - args.patch_len) // args.stride + 1

    model = TAMoE(
        c_in=c_in,
        target_dim=args.target_points,
        patch_len=args.patch_len,
        stride=args.stride,
        num_patch=num_patch,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        shared_embedding=True,
        d_ff=args.d_ff,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        act='relu',
        head_type='pretrain',
        res_attention=False,
        use_routed_expert=bool(args.use_routed_expert),
        num_experts=args.num_experts,
        moe_top_k=args.moe_top_k,
        moe_router_fusion_mode=args.moe_router_fusion_mode,
        use_shared_expert=bool(args.use_shared_expert),
        d_task=args.d_task,
    )

    model.config = {
        'c_in': c_in, 'num_patch': num_patch, 'context_points': args.context_points,
        'target_points': args.target_points, 'patch_len': args.patch_len, 'stride': args.stride,
        'n_layers': args.n_layers, 'n_heads': args.n_heads, 'd_model': args.d_model,
        'd_ff': args.d_ff, 'dropout': args.dropout, 'head_dropout': args.head_dropout,
        'd_task': args.d_task, 'task_mode': args.task_mode,
        'use_routed_expert': args.use_routed_expert, 'num_experts': args.num_experts,
        'moe_top_k': args.moe_top_k,
        'use_shared_expert': args.use_shared_expert,
        'moe_router_fusion_mode': args.moe_router_fusion_mode,
        'use_task_token': args.use_task_token,
        'use_cr': args.use_cr,
        'use_vwr': args.use_vwr,
        'use_periodic': args.use_periodic,
        'use_spectral': args.use_spectral,
        'use_stat': args.use_stat,
        'use_global_desc': args.use_global_desc,
    }

    if verbose:
        print(f'Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Callback Helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_task_probs():
    if args.task_mode == 'all':
        return {'PM': args.pm_prob, 'MPM': args.mpm_prob, 'RFM': args.rfm_prob,
                'SFM': args.sfm_prob, 'DM': args.dm_prob, 'HM': args.hm_prob}
    elif args.task_mode == 'time_only':
        return {'PM': 0.5, 'MPM': 0.5, 'RFM': 0, 'SFM': 0, 'DM': 0, 'HM': 0}
    elif args.task_mode == 'freq_only':
        return {'PM': 0, 'MPM': 0, 'RFM': 0.5, 'SFM': 0.5, 'DM': 0, 'HM': 0}
    else:  # single
        probs = {t: 0 for t in ['PM', 'MPM', 'RFM', 'SFM', 'DM', 'HM']}
        probs[args.single_task] = 1.0
        return probs


def create_multitask_cb():
    return MultiTaskReconCB(
        patch_len=args.patch_len,
        stride=args.stride,
        task_probs=get_task_probs(),
        pm_mask_ratio=args.pm_mask_ratio,
        mpm_mask_ratio=args.mpm_mask_ratio,
        rfm_mask_ratio=args.rfm_mask_ratio,
        sfm_tau=args.sfm_tau,
        sfm_band=args.sfm_band,
        dm_trend_mask_ratio=args.dm_trend_mask_ratio,
        dm_residual_mask_ratio=args.dm_residual_mask_ratio,
        hm_time_mask_ratio=args.hm_time_mask_ratio,
        loss_weights={
            'time': args.time_loss_weight,
            'decomp': args.decomp_loss_weight,
        },
        use_task_token=bool(args.use_task_token),
        use_fine_grained_task_id=bool(args.use_fine_grained_task_id),
        seed=args.seed,
    )


def _create_task_token_manager(verbose: bool = True):
    """Create TaskTokenManager from args (if enabled). Returns None otherwise."""
    if not args.use_task_token:
        return None
    from src.callback.task_token_manager import TaskTokenManager
    return TaskTokenManager(
        use_task_token=True,
        d_task=args.d_task,
        d_model=args.d_model,
        vwr_beta=args.vwr_beta,
        use_fine_grained_task_id=bool(args.use_fine_grained_task_id),
        use_cr=bool(args.use_cr),
        use_vwr=bool(args.use_vwr),
        use_periodic=bool(args.use_periodic),
        use_spectral=bool(args.use_spectral),
        use_stat=bool(args.use_stat),
        use_global_desc=bool(args.use_global_desc),
        use_learnable_vwr=bool(args.use_learnable_vwr),
        verbose=verbose,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def find_lr():
    dls = get_dls(args)
    model = get_model(dls.vars, verbose=False)

    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [create_multitask_cb()]

    tt_manager = _create_task_token_manager(verbose=False)
    if tt_manager:
        cbs += [tt_manager]

    opt_func = lambda params, lr, **kw: AdamW(params, lr=lr, weight_decay=0.0)
    _found_lr = Learner(dls, model, nn.MSELoss(), lr=args.lr, cbs=cbs, opt_func=opt_func).lr_finder(
        suggestion='valley'
    )

    return _found_lr


def pretrain_func(lr=None):
    task_probs = get_task_probs()
    active_tasks = [t for t, p in task_probs.items() if p > 0]
    print(f'Active Tasks: {active_tasks}')

    set_seed(args.seed, deterministic=False)
    dls = get_dls(args)
    model = get_model(dls.vars)

    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [create_multitask_cb()]

    tt_manager = _create_task_token_manager()
    extra_save_fn = tt_manager.save_task_token_weights if tt_manager else None

    cbs += [
        SaveModelCB(monitor='valid_loss', fname=args.save_pretrained_model,
                    path=args.save_path, extra_save_fn=extra_save_fn),
        TensorBoardCB(),
        MultiTaskPretrainTensorBoardCB(),
    ]

    if tt_manager:
        cbs += [tt_manager]

    if args.use_routed_expert:
        from src.callback.moe_callbacks import MoEAuxLossCB
        cbs += [MoEAuxLossCB(aux_loss_weight=args.moe_aux_loss_weight)]

        if args.use_shared_expert:
            from src.callback.moe_callbacks import MoEAlphaScheduleCB, MoERoutedL2CB
            cbs += [MoEAlphaScheduleCB(
                schedule=args.moe_alpha_schedule,
                alpha_start=args.moe_alpha_start,
                alpha_end=args.moe_alpha_end,
                plateau_start=args.moe_alpha_plateau_start,
            )]
            if args.moe_routed_l2_weight > 0:
                cbs += [MoERoutedL2CB(weight=args.moe_routed_l2_weight)]

    opt_func = lambda params, lr_arg, **kw: AdamW(params, lr=lr_arg, weight_decay=0.0)
    learn = Learner(dls, model, nn.MSELoss(), lr=lr or args.lr, cbs=cbs, opt_func=opt_func)
    learn.fit_one_cycle(n_epochs=args.n_epochs_pretrain, lr_max=lr or args.lr)

    df = pd.DataFrame({
        'train_loss': learn.recorder.get('train_loss', []),
        'valid_loss': learn.recorder.get('valid_loss', []),
    })
    df.to_csv(f'{args.save_path}{args.save_pretrained_model}_losses.csv', index=False)

    print(f'\nModel saved to: {args.save_path}{args.save_pretrained_model}')


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    train_lr = find_lr() or args.lr
    print(f'[LR] {train_lr:.2e}\n')
    pretrain_func(train_lr)
