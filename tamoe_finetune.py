# ══════════════════════════════════════════════════════════════════════════════
# Imports
# ══════════════════════════════════════════════════════════════════════════════

import os
import pandas as pd
import torch
from torch import nn

from src.models.tamoe_backbone import TAMoE
from src.learner import Learner
from src.utils import build_weight_path, load_model_config, transfer_weights, set_seed
from src.callback.core import *
from src.callback.tracking import *
from src.callback.transforms import *
from src.callback.tensorboard_logger import TensorBoardCB
from src.metrics import *
from datautils import get_dls

import argparse

# ══════════════════════════════════════════════════════════════════════════════
# CLI Arguments
# ══════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser()

# Mode
parser.add_argument('--is_finetune', type=int, default=0)
parser.add_argument('--is_linear_probe', type=int, default=0)

# Dataset
parser.add_argument('--dset', type=str, default='etth1')
parser.add_argument('--context_points', type=int, default=512)
parser.add_argument('--target_points', type=int, default=96)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--features', type=str, default='M')

# Model
parser.add_argument('--pretrained_model', type=str, default=None, help='Path to pretrained .pth')
parser.add_argument('--patch_len', type=int, default=12)
parser.add_argument('--stride', type=int, default=12)
parser.add_argument('--revin', type=int, default=1)
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--d_model', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--head_dropout', type=float, default=0.2)

# MoE
parser.add_argument('--use_routed_expert', type=int, default=0)
parser.add_argument('--use_shared_expert', type=int, default=0)
parser.add_argument('--num_experts', type=int, default=8)
parser.add_argument('--moe_top_k', type=int, default=2)
parser.add_argument('--aggregation_mode', type=str, default='auto',
                    choices=['router', 'shared_only', 'auto'],
                    help='MoE aggregation: auto infers from checkpoint config. '
                         'Config 3 (no task token): router. '
                         'Config 4 (task token): shared_only.')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='Alpha for routed experts in router mode with shared expert: '
                         'y = y_shared + alpha * y_routed. Default 1.0.')

# Training
parser.add_argument('--n_epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--use_manual_lr', type=int, default=0)
parser.add_argument('--seed', type=int, default=2021)

# Saving
parser.add_argument('--model_type', type=str, default='based_model')
parser.add_argument('--finetuned_model_id', type=int, default=1)

args = parser.parse_args()

# ══════════════════════════════════════════════════════════════════════════════
# Configuration (derived paths, GPU, seed)
# ══════════════════════════════════════════════════════════════════════════════

args.save_path = f'saved_models/{args.dset}/{args.model_type}/'
os.makedirs(args.save_path, exist_ok=True)

suffix = (f'_cw{args.context_points}_tw{args.target_points}'
          f'_patch{args.patch_len}_stride{args.stride}'
          f'_epochs{args.n_epochs}_model{args.finetuned_model_id}')
if args.is_finetune:
    mode_str = "finetuned"
elif args.is_linear_probe:
    mode_str = "linear-probe"
else:
    mode_str = "test"
args.save_finetuned_model = f'{args.dset}_tamoe_{mode_str}{suffix}'

if torch.cuda.is_available():
    torch.cuda.set_device(0)

if args.seed is not None:
    set_seed(args.seed, deterministic=True)
    print(f'Random seed set to {args.seed}, deterministic mode: True')


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

def _set_model_aggregation_mode(model, mode: str, verbose: bool = True):
    """Set aggregation mode for all MoE layers."""
    from src.models.task_adaptive_moe import TaskAdaptiveMoE
    count = 0
    for module in model.modules():
        if isinstance(module, TaskAdaptiveMoE):
            module.set_aggregation_mode(mode)
            count += 1
    if count > 0 and verbose:
        print(f'[MoE] Set aggregation_mode={mode} for {count} layer(s)')


def get_model(c_in, args, weight_path=None, load_weights=True, verbose=True):
    """Create model and optionally load pretrained weights.

    Architecture config is read from the checkpoint JSON sidecar when available,
    falling back to CLI args. Errors if pretrained .pth is specified but missing.
    """
    cfg = load_model_config(weight_path, verbose=verbose) or {} if weight_path else {}

    patch_len = cfg.get('patch_len', args.patch_len)
    stride = cfg.get('stride', args.stride)
    num_patch = cfg.get('num_patch') or (args.context_points - patch_len) // stride + 1

    n_layers = cfg.get('n_layers', args.n_layers)
    n_heads = cfg.get('n_heads', args.n_heads)
    d_model = cfg.get('d_model', args.d_model)
    d_ff = cfg.get('d_ff', args.d_ff)
    dropout = cfg.get('dropout', args.dropout)
    head_dropout = cfg.get('head_dropout', args.head_dropout)

    use_routed_expert = cfg.get('use_routed_expert', args.use_routed_expert)
    use_shared_expert = cfg.get('use_shared_expert', args.use_shared_expert)
    num_experts = cfg.get('num_experts', args.num_experts)
    moe_top_k = cfg.get('moe_top_k', args.moe_top_k)
    moe_router_fusion_mode = cfg.get('moe_router_fusion_mode', 'concat')
    d_task = cfg.get('d_task', 16)

    model = TAMoE(
        c_in=c_in,
        target_dim=args.target_points,
        patch_len=patch_len,
        stride=stride,
        num_patch=num_patch,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        shared_embedding=True,
        d_ff=d_ff,
        dropout=dropout,
        head_dropout=head_dropout,
        act='relu',
        head_type='prediction',
        res_attention=False,
        use_routed_expert=bool(use_routed_expert),
        num_experts=num_experts,
        moe_top_k=moe_top_k,
        use_shared_expert=bool(use_shared_expert),
        moe_router_fusion_mode=moe_router_fusion_mode,
        d_task=d_task,
    )

    model.config = {
        'c_in': c_in, 'num_patch': num_patch,
        'context_points': args.context_points, 'target_points': args.target_points,
        'patch_len': patch_len, 'stride': stride,
        'n_layers': n_layers, 'n_heads': n_heads, 'd_model': d_model,
        'd_ff': d_ff, 'dropout': dropout, 'head_dropout': head_dropout,
        'd_task': d_task,
        'use_routed_expert': int(use_routed_expert), 'num_experts': num_experts,
        'moe_top_k': moe_top_k,
        'use_shared_expert': int(use_shared_expert),
        'moe_router_fusion_mode': moe_router_fusion_mode,
    }

    if weight_path and load_weights:
        model = transfer_weights(weight_path, model, exclude_head=True, verbose=verbose)

    if use_routed_expert:
        if args.aggregation_mode != 'auto':
            agg_mode = args.aggregation_mode
        elif use_shared_expert:
            # Config 4 (Task-Aware MoE, use_task_token=True): shared expert only during finetune
            # Config 3 (MoE + Shared, no task token): keep routed experts active
            pretrain_use_task_token = cfg.get('use_task_token', False)
            agg_mode = 'shared_only' if pretrain_use_task_token else 'router'
        else:
            agg_mode = 'router'

        if agg_mode == 'shared_only' and not use_shared_expert:
            print(f"[Warning] shared_only requires use_shared_expert, switching to router")
            agg_mode = 'router'

        _set_model_aggregation_mode(model, agg_mode, verbose=verbose)

        # When router mode is active with shared expert, set alpha
        # (pretrain uses alpha schedule 0→target; finetune uses fixed value)
        if agg_mode == 'router' and use_shared_expert:
            from src.models.tamoe_backbone import TransformerLayer
            for module in model.modules():
                if isinstance(module, TransformerLayer) and module.use_shared_expert:
                    module.set_moe_alpha(args.alpha)
            if verbose:
                print(f'[MoE] Set moe_alpha={args.alpha} for router mode with shared expert')

    if verbose:
        print(f'Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    return model


def _is_pretrain_checkpoint(weight_path: str) -> bool:
    """Detect pretrain checkpoints, which contain reconstruction head weights."""
    checkpoint = torch.load(weight_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model') or checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
    else:
        state_dict = checkpoint
    return (
        isinstance(state_dict, dict)
        and 'head.linear.weight' in state_dict
        and not any(k.startswith('head.head') or k.startswith('head.heads') for k in state_dict)
    )


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def find_lr():
    dls = get_dls(args)
    weight_path = build_weight_path(args.pretrained_model, args.dset, args.model_type)
    model = get_model(dls.vars, args, weight_path=weight_path, verbose=False)

    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]

    return Learner(dls, model, nn.MSELoss(), lr=args.lr, cbs=cbs).lr_finder(suggestion='valley')


def _build_learner(lr=None):
    """Shared setup: create dataloader, model, callbacks, and Learner."""
    if args.seed is not None:
        set_seed(args.seed, deterministic=True)

    dls = get_dls(args)
    weight_path = build_weight_path(args.pretrained_model, args.dset, args.model_type)
    model = get_model(dls.vars, args, weight_path=weight_path)

    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []

    cbs += [
        PatchCB(patch_len=args.patch_len, stride=args.stride),
        SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path),
        TensorBoardCB(),
    ]

    learn = Learner(dls, model, nn.MSELoss(), lr=lr or args.lr, cbs=cbs, metrics=[mse])
    return learn


def _save_losses(learn):
    pd.DataFrame({
        'train_loss': learn.recorder['train_loss'],
        'valid_loss': learn.recorder['valid_loss']
    }).to_csv(f'{args.save_path}{args.save_finetuned_model}_losses.csv', index=False)


def finetune_func(lr=None):
    print('Finetuning...')
    learn = _build_learner(lr)
    learn.fine_tune(n_epochs=args.n_epochs, base_lr=lr or args.lr, freeze_epochs=10)
    _save_losses(learn)


def linear_probe_func(lr=None):
    print('Linear probing')
    learn = _build_learner(lr)
    learn.linear_probe(n_epochs=args.n_epochs, base_lr=lr or args.lr)
    _save_losses(learn)


def test_func(weight_path=None):
    if args.seed is not None:
        set_seed(args.seed, deterministic=True)

    weight_path = weight_path or build_weight_path(args.pretrained_model, args.dset, args.model_type)
    if not weight_path.endswith('.pth'):
        weight_path += '.pth'
    if _is_pretrain_checkpoint(weight_path):
        raise ValueError(
            "Forecast testing expects a finetuned prediction checkpoint. "
            "Pretrain checkpoints are reconstruction models and should be used via finetuning."
        )

    dls = get_dls(args)
    model = get_model(dls.vars, args, weight_path=weight_path, load_weights=False, verbose=False)

    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]

    learn = Learner(dls, model, cbs=cbs)
    out = learn.test(dls.test, weight_path=weight_path, scores=[mse, mae])

    metrics = out[2] if len(out) > 2 else [0, 0]
    print(f'Test -> MSE: {float(metrics[0]):.6f}, MAE: {float(metrics[1]):.6f}')

    pd.DataFrame([metrics], columns=['mse', 'mae']).to_csv(
        f'{args.save_path}{args.save_finetuned_model}_acc.csv', index=False)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if args.is_finetune or args.is_linear_probe:
        if args.use_manual_lr:
            lr = args.lr
            print(f'Using manual LR: {lr:.2e} (skipping LR finder)')
        else:
            lr = find_lr() or args.lr
            print(f'Using LR: {lr:.2e}')

        if args.is_finetune:
            finetune_func(lr)
        else:
            linear_probe_func(lr)
        test_func(args.save_path + args.save_finetuned_model)
    else:
        test_func(args.pretrained_model)

    print('Done!')
