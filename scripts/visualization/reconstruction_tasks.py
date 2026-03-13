#!/usr/bin/env python
"""
Visualize Mask Patterns for Multi-Task Pretraining

Visualize the 6 masking strategies used in multi-task reconstruction pretraining:
- PM (Patch Masking): Random patch-level masking
- MPM (Multi-patch Block Masking): Contiguous block masking
- RFM (Random Frequency Masking): Random frequency bin masking
- SFM (Structured Frequency Masking): Band-based frequency masking
- DM (Decomposition Masking): Trend/residual aware masking
- HM (Holistic Masking): Patch masking with full-sequence reconstruction
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add project root to path (3 levels up from scripts/visualization/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from src.masking import (
    PatchMasking, BlockMasking,
    RandomFreqMasking, StructuredFreqMasking,
    DecompositionMasking,
)
from datautils import get_dls

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'


def get_sample_data(dataset: str, ctx_points: int = 512):
    """Load a sample from the dataset."""
    params = argparse.Namespace(
        dset=dataset,
        context_points=ctx_points,
        target_points=96,
        batch_size=1,
        num_workers=0,
        features='M',
        patch_len=12,
        stride=12,
    )
    dls = get_dls(params)
    for batch in dls.train:
        xb, yb = batch
        return xb  # [1, T, C]


def reconstruct_from_patches(patches: np.ndarray, T: int, s_begin: int,
                             stride: int, patch_len: int) -> np.ndarray:
    """Reconstruct signal from patches by averaging overlapping regions.

    Args:
        patches: [num_patch, patch_len] patch data
        T: Total sequence length
        s_begin: Starting position of first patch
        stride: Stride between patches
        patch_len: Length of each patch

    Returns:
        Reconstructed signal [T]
    """
    num_patch = len(patches)
    signal = np.zeros(T)
    count = np.zeros(T)

    for i in range(num_patch):
        patch_start = s_begin + i * stride
        patch_end = patch_start + patch_len
        signal[patch_start:patch_end] += patches[i]
        count[patch_start:patch_end] += 1

    count[count == 0] = 1
    return signal / count


def create_mask_strategies(patch_len: int = 12, stride: int = 12):
    """Create all 6 masking strategies with actual pretraining ratios."""
    strategies = {
        'PM': PatchMasking(
            patch_len=patch_len, stride=stride, mask_ratio=0.4
        ),
        'MPM': BlockMasking(
            patch_len=patch_len, stride=stride, mask_ratio=0.4
        ),
        'RFM': RandomFreqMasking(
            patch_len=patch_len, stride=stride, mask_ratio=0.3
        ),
        'SFM': StructuredFreqMasking(
            patch_len=patch_len, stride=stride, tau=0.5,
            mask_band='low'
        ),
        'DM': DecompositionMasking(
            patch_len=patch_len, stride=stride,
            trend_mask_ratio=0.2, residual_mask_ratio=0.5
        ),
        'HM': PatchMasking(
            patch_len=patch_len, stride=stride, mask_ratio=0.4
        ),
    }
    return strategies


def visualize_combined_mask_and_input(
    x: torch.Tensor,
    strategies: dict,
    output_dir: str,
    var_idx: int = 0,
):
    """
    Combined visualization: 3x2 layout, 6 subplots total
    Each subplot shows BOTH mask regions AND original vs corrupted in the same plot
    For freq strategies (RFM, SFM), also show frequency mask in an inset
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    strategy_names = ['PM', 'MPM', 'RFM', 'SFM', 'DM', 'HM']
    strategy_titles = {
        'PM': 'Patch Masking',
        'MPM': 'Block Masking',
        'RFM': 'Random Freq Masking',
        'SFM': 'Structured Freq Masking',
        'DM': 'Decomposition Masking',
        'HM': 'Holistic Masking',
    }

    # Unified color scheme
    color_original = '#1f77b4'  # Blue for original signal
    color_corrupted = '#2ca02c'  # Green for corrupted signal
    color_mask = '#d3d3d3'  # Light gray for ALL mask regions (time & freq)
    color_freq_highlight = '#e74c3c'  # Red for frequency mask highlight in inset
    color_trend = '#f39c12'  # Yellow/Orange for trend (DM)
    color_residual = '#e67e22'  # Orange for residual (DM)

    T = x.shape[1]
    x_original = x[0, :, var_idx].cpu().numpy()
    time = np.arange(T)

    for idx, name in enumerate(strategy_names):
        ax = axes[idx]
        strategy = strategies[name]
        view = strategy.make_view(x)

        # Get patches info
        x_patch, num_patch = strategy.patchify(x)
        patch_len = strategy.patch_len
        stride = strategy.stride

        # Calculate patching start
        tgt_len = patch_len + stride * (num_patch - 1)
        s_begin = T - tgt_len

        # Get mask info
        mask = view['mask'][0].cpu().numpy()
        mask_ratio = mask.sum() / len(mask)

        # Reconstruct corrupted signal
        x_in = view['x_in'][0].cpu().numpy()
        n_vars = x_in.shape[-1] // patch_len
        x_in_patches = x_in.reshape(num_patch, n_vars, patch_len)[:, var_idx, :]

        corrupted_signal = reconstruct_from_patches(x_in_patches, T, s_begin, stride, patch_len)

        # Highlight masked regions FIRST (so they appear behind the lines)
        if name not in ['RFM', 'SFM']:
            for i in range(num_patch):
                if mask[i]:
                    patch_start = s_begin + i * stride
                    patch_end = patch_start + patch_len
                    ax.axvspan(patch_start, patch_end, alpha=0.4, color=color_mask, zorder=0)

        ax.plot(time, x_original, color=color_original, linewidth=1.8, alpha=0.9,
               label='Original', zorder=2)

        ax.plot(time[s_begin:], corrupted_signal[s_begin:], color=color_corrupted,
               linewidth=1.8, alpha=0.85, linestyle='--', label='Corrupted', zorder=3)

        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_xlim(0, T)

        if name in ['RFM', 'SFM'] and view['aux'] is not None:
            freq_mask = view['aux'].get('freq_mask', None)
            if freq_mask is not None:
                freq_mask_var = freq_mask[0, :, var_idx].cpu().numpy()
                freq_mask_ratio = freq_mask_var.mean()
                masked_freqs = np.where(freq_mask_var)[0]
            else:
                freq_mask_ratio = getattr(strategy, 'mask_ratio', 0.3)
                masked_freqs = []

            # Create inset axes for frequency mask (bottom-right corner)
            inset_ax = ax.inset_axes([0.77, 0.02, 0.22, 0.18])

            x_freq = np.fft.rfft(x_original)
            freq_bins = np.fft.rfftfreq(T)
            amplitude_log = np.log10(np.abs(x_freq) + 1e-8)

            inset_ax.fill_between(freq_bins, amplitude_log, alpha=0.4, color=color_freq_highlight)
            inset_ax.plot(freq_bins, amplitude_log, color=color_freq_highlight, linewidth=0.8)

            if len(masked_freqs) > 0:
                if name == 'SFM':
                    min_masked = freq_bins[masked_freqs[0]]
                    max_masked = freq_bins[masked_freqs[-1]]
                    inset_ax.axvspan(min_masked, max_masked, alpha=0.6, color=color_mask)
                else:
                    for mf in masked_freqs[::2]:
                        if mf < len(freq_bins):
                            inset_ax.axvline(freq_bins[mf], color=color_mask, alpha=0.7, linewidth=0.8)

            inset_ax.set_xlim(0, 0.5)
            inset_ax.set_xticks([0, 0.5])
            inset_ax.set_yticks([])
            inset_ax.tick_params(labelsize=4, pad=1)
            inset_ax.xaxis.set_tick_params(length=2)

            if name == 'SFM':
                band = view['aux'].get('band', 'low')
                ax.set_title(f'{strategy_titles[name]} ({band}-freq)\n(Mask Ratio: {freq_mask_ratio:.0%})',
                           fontsize=11, fontweight='bold')
            else:
                ax.set_title(f'{strategy_titles[name]}\n(Mask Ratio: {freq_mask_ratio:.0%})',
                           fontsize=11, fontweight='bold')

            legend_elements = [
                Line2D([0], [0], color=color_original, linewidth=1.8, label='Original'),
                Line2D([0], [0], color=color_corrupted, linewidth=1.8, linestyle='--', label='Corrupted'),
                Line2D([0], [0], color=color_freq_highlight, linewidth=1.5, label='Freq Spectrum'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        elif name == 'DM' and view['aux'] is not None:
            decomp = view['aux'].get('decomp', {})

            if 'trend' in decomp and 'residual' in decomp:
                trend_patches = decomp['trend'][0].cpu().numpy()
                residual_patches = decomp['residual'][0].cpu().numpy()

                trend_patches = trend_patches.reshape(num_patch, -1, patch_len)[:, var_idx, :]
                residual_patches = residual_patches.reshape(num_patch, -1, patch_len)[:, var_idx, :]

                trend_recon = reconstruct_from_patches(trend_patches, T, s_begin, stride, patch_len)
                residual_recon = reconstruct_from_patches(residual_patches, T, s_begin, stride, patch_len)

                ax.plot(time[s_begin:], trend_recon[s_begin:], color=color_trend,
                       linewidth=1.5, alpha=0.9, label='Trend', zorder=4)
                ax.plot(time[s_begin:], residual_recon[s_begin:], color=color_residual,
                       linewidth=1.2, alpha=0.8, linestyle=':', label='Residual', zorder=4)

            trend_mask_ratio = getattr(strategy, 'trend_mask_ratio', 0.2)
            residual_mask_ratio = getattr(strategy, 'residual_mask_ratio', 0.5)

            ax.set_title(f'{strategy_titles[name]}\n(Trend: {trend_mask_ratio:.0%}, Residual: {residual_mask_ratio:.0%})',
                        fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=7)
        else:
            ax.set_title(f'{strategy_titles[name]}\n(Mask Ratio: {mask_ratio:.0%})',
                        fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('Multi-view Reconstruction Pretraining', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    output_path_png = os.path.join(output_dir, 'combined_mask_visualization.png')
    plt.savefig(output_path_png, bbox_inches='tight', facecolor='white', dpi=300)
    print(f'  Saved: {output_path_png}')

    output_path_pdf = os.path.join(output_dir, 'combined_mask_visualization.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white', format='pdf')
    print(f'  Saved: {output_path_pdf}')

    output_path_svg = os.path.join(output_dir, 'combined_mask_visualization.svg')
    plt.savefig(output_path_svg, bbox_inches='tight', facecolor='white', format='svg')
    print(f'  Saved: {output_path_svg}')

    plt.close()


def visualize_pretraining_config(output_dir: str):
    # Multi-task pretraining configuration
    task_probs = {
        'PM': 0.16,    # Patch Masking
        'MPM': 0.16,   # Block Masking
        'RFM': 0.18,   # Random Freq Masking
        'SFM': 0.18,   # Structured Freq Masking
        'DM': 0.18,    # Decomposition Masking
        'HM': 0.14,    # Holistic Masking
    }

    mask_ratios = {
        'PM': 0.4,     # 40% patches masked
        'MPM': 0.4,    # 40% blocks masked
        'RFM': 0.3,    # 30% freq bins masked
        'SFM': 0.5,    # tau=0.5 (variable band masking)
        'DM': 0.35,    # avg(trend=0.2, residual=0.5)
        'HM': 0.4,     # 40% holistic masking
    }

    task_descriptions = {
        'PM': 'Patch\nMasking',
        'MPM': 'Block\nMasking',
        'RFM': 'Random\nFreq Mask',
        'SFM': 'Structured\nFreq Mask',
        'DM': 'Decomp\nMasking',
        'HM': 'Holistic\nMasking',
    }

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    colors = plt.cm.Set2(np.linspace(0, 1, 6))
    task_names = list(task_probs.keys())

    # 1. Task Sampling Probability (Pie chart)
    ax1 = fig.add_subplot(gs[0, 0])
    probs = [task_probs[t] for t in task_names]
    wedges, texts, autotexts = ax1.pie(
        probs, labels=task_names, autopct='%1.0f%%',
        colors=colors, startangle=90, explode=[0.02]*6
    )
    ax1.set_title('Task Sampling Probability\n(Random sample one task per batch)', fontsize=12, fontweight='bold')

    # 2. Mask Ratio (Bar chart)
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(task_names))
    ratios = [mask_ratios[t] for t in task_names]
    bars = ax2.bar(x, ratios, color=colors, edgecolor='black', linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels([task_descriptions[t] for t in task_names], fontsize=9)
    ax2.set_ylabel('Mask Ratio', fontsize=11)
    ax2.set_ylim(0, 0.6)
    ax2.set_title('Mask Ratio per Task\n(Masking intensity for each task)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, ratio in zip(bars, ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{ratio:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 3. Configuration Table
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')

    table_data = [
        ['Task', 'Domain', 'Mask Type', 'Mask Ratio', 'Sample Prob', 'Loss Type'],
        ['PM', 'Time', 'Random Patch', '40%', '16%', 'MSE on masked'],
        ['MPM', 'Time', 'Contiguous Block', '40%', '16%', 'MSE on masked'],
        ['RFM', 'Frequency', 'Random Freq Bins', '30%', '18%', 'MSE full seq'],
        ['SFM', 'Frequency', 'Band (low/high/random)', 'tau=0.5', '18%', 'MSE full seq'],
        ['DM', 'Time (Decomp)', 'Trend 20% + Residual 50%', '35% avg', '18%', 'MSE + Decomp'],
        ['HM', 'Time', 'Random Patch', '40%', '14%', 'MSE full seq'],
    ]

    table = ax3.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', weight='bold')

    # Alternate row colors
    for row in range(1, 7):
        for col in range(6):
            if row % 2 == 0:
                table[(row, col)].set_facecolor('#E8F0FE')

    plt.suptitle('Multi-Task Pretraining Configuration',
                fontsize=14, fontweight='bold', y=0.98)

    output_path = os.path.join(output_dir, 'pretraining_config.png')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', dpi=150)
    plt.close()
    print(f'  Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Visualize Mask Patterns')
    parser.add_argument('--dataset', type=str, default='etth2',
                       help='Dataset name (default: etth2)')
    parser.add_argument('--output_dir', type=str, default='analysis/mask_patterns',
                       help='Output directory')
    parser.add_argument('--context_points', type=int, default=512,
                       help='Context window size')
    parser.add_argument('--patch_len', type=int, default=12,
                       help='Patch length')
    parser.add_argument('--stride', type=int, default=12,
                       help='Stride between patches')
    parser.add_argument('--var_idx', type=int, default=0,
                       help='Variable index to visualize')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print('Mask Pattern Visualization')
    print(f'{"="*60}')
    print(f'Dataset: {args.dataset}')
    print(f'Context Points: {args.context_points}')
    print(f'Patch Length: {args.patch_len}')
    print(f'Stride: {args.stride}')
    print(f'Output Directory: {args.output_dir}')
    print(f'{"="*60}\n')

    print('Loading sample data...')
    x = get_sample_data(args.dataset, args.context_points)
    print(f'  Sample shape: {x.shape}')

    print('\nCreating masking strategies...')
    strategies = create_mask_strategies(args.patch_len, args.stride)
    print(f'  Created {len(strategies)} strategies: {list(strategies.keys())}')

    print('\nGenerating combined mask & input visualization...')
    visualize_combined_mask_and_input(x, strategies, args.output_dir, args.var_idx)

    print('\nGenerating pretraining configuration summary...')
    visualize_pretraining_config(args.output_dir)

    print(f'\n{"="*60}')
    print('Mask Pattern Visualization Complete!')
    print(f'Output directory: {args.output_dir}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()

