"""
KV Cache Benchmark — Plotting Script
Generates 5 publication-ready comparison plots.

Usage:
  python plot_results.py --results results/results.jsonl
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── STYLE ─────────────────────────────────────────────────────────────────────

# Colorblind-friendly palette (IBM accessible palette)
METHOD_COLORS = {
    'baseline': '#000000',
    'kivi':     '#0072B2',
    'xkv':      '#D55E00',
    'snapkv':   '#009E73',
    'topk':     '#CC79A7',
}
METHOD_MARKERS = {
    'baseline': '*',
    'kivi':     'o',
    'xkv':      's',
    'snapkv':   '^',
    'topk':     'D',
}

ACADEMIC_STYLE = {
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'lines.linewidth': 1.5,
    'lines.markersize': 7,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
}


def load_results(path: Path) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    # Flatten config dict into columns
                    cfg = rec.pop('config', {}) or {}
                    rec.update({f'cfg_{k}': v for k, v in cfg.items()})
                    records.append(rec)
                except json.JSONDecodeError:
                    pass
    if not records:
        raise ValueError(f"No valid records found in {path}")
    return pd.DataFrame(records)


def best_config_per_method(df, value_col, higher_is_better=True):
    """For each method, select the config with the best average value_col."""
    result = {}
    for method in df['method'].unique():
        method_df = df[df['method'] == method].dropna(subset=[value_col])
        if method_df.empty:
            continue
        # Group by config columns and take mean
        cfg_cols = [c for c in method_df.columns if c.startswith('cfg_')]
        # Drop config columns that are entirely NaN (baseline has no cfg_ values)
        cfg_cols = [c for c in cfg_cols if method_df[c].notna().any()]
        if cfg_cols:
            try:
                grouped = method_df.groupby(cfg_cols)[value_col].mean()
                if grouped.empty:
                    result[method] = method_df
                    continue
                best_idx = grouped.idxmax() if higher_is_better else grouped.idxmin()
                if not isinstance(best_idx, tuple):
                    best_idx = (best_idx,)
                mask = pd.Series([True] * len(method_df), index=method_df.index)
                for col, val in zip(cfg_cols, best_idx):
                    mask &= (method_df[col] == val)
                result[method] = method_df[mask]
            except (ValueError, KeyError):
                result[method] = method_df
        else:
            result[method] = method_df
    return result


def save_fig(fig, plots_dir: Path, name: str):
    plots_dir.mkdir(parents=True, exist_ok=True)
    png_path = plots_dir / f"{name}.png"
    pdf_path = plots_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  Saved {png_path} and {pdf_path}")


# ── PLOT 1: Memory vs Quality ──────────────────────────────────────────────────

def plot_memory_vs_quality(df, plots_dir):
    df = df[~df['method'].str.contains('FALLBACK', na=False)]
    ppl_df = df[df['prompt_type'] == 'wikitext'].dropna(subset=['perplexity', 'compression_ratio'])
    if ppl_df.empty:
        print("  [SKIP] memory_vs_quality: no wikitext/perplexity data")
        return

    with plt.rc_context(ACADEMIC_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))

        for method in ppl_df['method'].unique():
            method_df = ppl_df[ppl_df['method'] == method]
            base_name = method.replace('_FALLBACK', '')
            color = METHOD_COLORS.get(base_name, '#888888')
            marker = METHOD_MARKERS.get(base_name, 'x')
            marker_size = 14 if method == 'baseline' else 7

            ax.scatter(
                method_df['compression_ratio'],
                method_df['perplexity'],
                label=method,
                color=color,
                marker=marker,
                s=marker_size ** 2,
                zorder=5,
                alpha=0.8,
            )

        # Baseline gold star annotation
        base_df = ppl_df[ppl_df['method'] == 'baseline']
        if not base_df.empty:
            bx = base_df['compression_ratio'].mean()
            by = base_df['perplexity'].mean()
            ax.annotate(
                'Baseline\n(reference)',
                xy=(bx, by),
                xytext=(bx + 0.1, by + 1),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=8,
            )

        ax.set_xlabel('Compression Ratio (higher = more compressed)')
        ax.set_ylabel('Perplexity (lower = better)')
        ax.set_title('Memory–Quality Tradeoff\n(lower-left is better)')
        ax.legend(loc='upper right')

        # Add direction annotation
        ax.annotate('', xy=(0.15, 0.15), xytext=(0.25, 0.25),
                    xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
        ax.text(0.12, 0.12, 'Better', transform=ax.transAxes,
                color='green', fontsize=9)

        fig.tight_layout()
        save_fig(fig, plots_dir, 'memory_vs_quality')
        plt.close(fig)


# ── PLOT 2: Throughput vs Compression ─────────────────────────────────────────

def plot_throughput_vs_compression(df, plots_dir):
    df = df[~df['method'].str.contains('FALLBACK', na=False)]
    syn_df = df[df['prompt_type'] == 'synthetic'].dropna(
        subset=['throughput_tps', 'compression_ratio']
    )
    syn_df = syn_df[syn_df['throughput_tps'] > 0]
    if syn_df.empty:
        print("  [SKIP] throughput_vs_compression: no synthetic data")
        return

    with plt.rc_context(ACADEMIC_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))

        for method in syn_df['method'].unique():
            method_df = syn_df[syn_df['method'] == method]
            base_name = method.replace('_FALLBACK', '')
            color = METHOD_COLORS.get(base_name, '#888888')
            marker = METHOD_MARKERS.get(base_name, 'x')

            ax.scatter(
                method_df['compression_ratio'],
                method_df['throughput_tps'],
                label=method,
                color=color,
                marker=marker,
                alpha=0.7,
                s=50,
            )

        ax.set_xlabel('Compression Ratio (higher = more compressed)')
        ax.set_ylabel('Throughput (tokens/sec)')
        ax.set_title('Speed–Memory Tradeoff\n(TopK ≈ 1.0x compression; xKV slow due to SVD)')
        ax.legend(loc='upper left')
        fig.tight_layout()
        save_fig(fig, plots_dir, 'throughput_vs_compression')
        plt.close(fig)


# ── PLOT 3: Memory Scaling ─────────────────────────────────────────────────────

def plot_memory_scaling(df, plots_dir):
    df = df[~df['method'].str.contains('FALLBACK', na=False)]
    syn_df = df[df['prompt_type'] == 'synthetic'].dropna(subset=['peak_memory_gb', 'seq_len'])
    if syn_df.empty:
        print("  [SKIP] memory_scaling: no synthetic data")
        return

    # Best config per method (highest throughput as proxy)
    best = best_config_per_method(syn_df, 'throughput_tps', higher_is_better=True)

    with plt.rc_context(ACADEMIC_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))

        for method, method_df in best.items():
            base_name = method.replace('_FALLBACK', '')
            color = METHOD_COLORS.get(base_name, '#888888')
            marker = METHOD_MARKERS.get(base_name, 'x')

            grouped = method_df.groupby('seq_len')['peak_memory_gb'].mean().reset_index()
            grouped = grouped.sort_values('seq_len')

            ax.plot(
                grouped['seq_len'],
                grouped['peak_memory_gb'],
                label=method,
                color=color,
                marker=marker,
            )

        ax.set_xlabel('Input Sequence Length (tokens)')
        ax.set_ylabel('Peak GPU Memory (GB)')
        ax.set_title('Memory Scaling vs Sequence Length\n(best config per method)')
        ax.legend()
        fig.tight_layout()
        save_fig(fig, plots_dir, 'memory_scaling')
        plt.close(fig)


# ── PLOT 4: Latency Breakdown ──────────────────────────────────────────────────

def plot_latency_breakdown(df, plots_dir):
    df = df[~df['method'].str.contains('FALLBACK', na=False)]
    syn_df = df[df['prompt_type'] == 'synthetic'].dropna(
        subset=['ttft_ms', 'per_token_latency_ms']
    )
    if syn_df.empty:
        print("  [SKIP] latency_breakdown: no synthetic data")
        return

    seq_lens = sorted(syn_df['seq_len'].unique())
    methods = [m for m in syn_df['method'].unique() if 'FALLBACK' not in m]
    n_methods = len(methods)
    n_seq = len(seq_lens)

    with plt.rc_context(ACADEMIC_STYLE):
        fig, axes = plt.subplots(1, n_seq, figsize=(5 * n_seq, 5), sharey=False)
        if n_seq == 1:
            axes = [axes]

        bar_width = 0.35
        x = np.arange(n_methods)

        for ax, sl in zip(axes, seq_lens):
            sl_df = syn_df[syn_df['seq_len'] == sl]
            ttft_vals = []
            ptl_vals = []
            for m in methods:
                mdf = sl_df[sl_df['method'] == m]
                ttft_vals.append(mdf['ttft_ms'].mean() if not mdf.empty else 0)
                ptl_vals.append(mdf['per_token_latency_ms'].mean() if not mdf.empty else 0)

            bars1 = ax.bar(x - bar_width / 2, ttft_vals, bar_width,
                           label='TTFT (ms)', color='#0072B2', alpha=0.8)
            bars2 = ax.bar(x + bar_width / 2, ptl_vals, bar_width,
                           label='Per-token latency (ms)', color='#D55E00', alpha=0.8)

            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=30, ha='right')
            ax.set_title(f'seq_len = {sl}')
            ax.set_ylabel('Latency (ms)')
            ax.legend()

        fig.suptitle('Latency Breakdown: TTFT vs Per-Token Latency', fontsize=13)
        fig.tight_layout()
        save_fig(fig, plots_dir, 'latency_breakdown')
        plt.close(fig)


# ── PLOT 5: LongBench Radar ────────────────────────────────────────────────────

def plot_longbench_radar(df, plots_dir):
    df = df[~df['method'].str.contains('FALLBACK', na=False)]
    lb_df = df[df['prompt_type'] == 'longbench'].dropna(subset=['longbench_score'])
    if lb_df.empty:
        print("  [SKIP] longbench_radar: no LongBench data")
        return

    tasks = sorted(lb_df['task'].unique())
    if len(tasks) < 3:
        print(f"  [SKIP] longbench_radar: only {len(tasks)} tasks, need ≥3 for radar")
        return

    methods = [m for m in lb_df['method'].unique() if 'FALLBACK' not in m]
    n_tasks = len(tasks)
    angles = np.linspace(0, 2 * np.pi, n_tasks, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    with plt.rc_context(ACADEMIC_STYLE):
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

        for method in methods:
            base_name = method.replace('_FALLBACK', '')
            color = METHOD_COLORS.get(base_name, '#888888')
            # Best config: highest mean score across tasks
            method_lb = lb_df[lb_df['method'] == method]
            scores = []
            for task in tasks:
                task_df = method_lb[method_lb['task'] == task]
                scores.append(task_df['longbench_score'].mean() if not task_df.empty else 0.0)
            scores += scores[:1]

            ax.plot(angles, scores, color=color, label=method, linewidth=2)
            ax.fill(angles, scores, color=color, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tasks, size=9)
        ax.set_title('LongBench Task Quality\n(best config per method)', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        fig.tight_layout()
        save_fig(fig, plots_dir, 'longbench_radar')
        plt.close(fig)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KV Benchmark Plotting")
    parser.add_argument('--results', default='results/results.jsonl',
                        help='Path to results.jsonl')
    parser.add_argument('--plots_dir', default='results/plots',
                        help='Output directory for plots')
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run the benchmark first.")
        return

    plots_dir = Path(args.plots_dir)
    print(f"Loading results from {results_path}...")
    df = load_results(results_path)
    print(f"Loaded {len(df)} records across methods: {df['method'].unique().tolist()}")

    print("\nGenerating plots...")
    plot_memory_vs_quality(df, plots_dir)
    plot_throughput_vs_compression(df, plots_dir)
    plot_memory_scaling(df, plots_dir)
    plot_latency_breakdown(df, plots_dir)
    plot_longbench_radar(df, plots_dir)

    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
