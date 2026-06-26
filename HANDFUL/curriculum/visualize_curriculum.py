"""
Curriculum training visualizer.

Reads curriculum_results.json and produces a matplotlib figure showing
how each grasp performed across stages, which were pruned, and the
metric distribution at each stage.

Usage:
    # Called automatically at end of curriculum_train.py, or standalone:
    python visualize_curriculum.py
    python visualize_curriculum.py --results path/to/curriculum_results.json
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


STAGE_COLORS  = ['#4ECDC4', '#F5A623', '#E8563A']   # one per stage
PRUNED_ALPHA  = 0.25
SURVIVOR_ALPHA = 0.9


def short_name(g):
    """Shorten grasp dir name for axis labels."""
    return (g.replace("fingers_", "")
             .replace("_seed_8", "")
             .replace("active_", "a")
             .replace("palm_True", "pT")
             .replace("palm_False", "pF"))


def get_metric_val(stage_metrics, grasp, metric_name):
    g_dict = stage_metrics.get(grasp)
    if g_dict is None:
        return np.nan
    if isinstance(g_dict, dict):
        for k in [metric_name, f"eval/{metric_name}", f"train/{metric_name}"]:
            if k in g_dict:
                return g_dict[k]
        for k in g_dict:
            if metric_name in k:
                return g_dict[k]
        return list(g_dict.values())[0] if g_dict else np.nan
    return g_dict


def load_results(results_path: str) -> dict:
    with open(results_path) as f:
        return json.load(f)


def visualize(results_path: str, output_path: str = None, show: bool = True):
    results   = load_results(results_path)
    env_id    = results.get("env_id", "unknown")
    no_cull   = results.get("no_cull", False)
    stages    = results.get("stages", [])
    survivors = set(results.get("final_survivors", []))
    metric    = results.get("args", {}).get("metric", "success_once")

    # Collect all grasp names in first-seen order
    all_grasps = []
    for stage in stages:
        for g in stage["metrics"]:
            if g not in all_grasps:
                all_grasps.append(g)

    n_grasps = len(all_grasps)
    n_stages = len(stages)
    x        = np.arange(n_grasps)
    width    = 0.8 / n_stages  # bar width so groups sit side by side

    fig, axes = plt.subplots(3, 1, figsize=(max(12, n_grasps * 1.4), 14))
    fig.suptitle(
        f"Curriculum Training — {env_id}{'  [no_cull]' if no_cull else ''}",
        fontsize=14, fontweight='bold', y=0.98
    )

    # -----------------------------------------------------------------------
    # Plot 1: Grouped bar chart — metric per grasp per stage
    # -----------------------------------------------------------------------
    ax1 = axes[0]
    for s_idx, stage in enumerate(stages):
        vals    = [get_metric_val(stage["metrics"], g, metric) for g in all_grasps]
        pruned  = set(stage.get("pruned", []))
        alphas  = [PRUNED_ALPHA if g in pruned else SURVIVOR_ALPHA for g in all_grasps]
        offsets = x + (s_idx - (n_stages - 1) / 2) * width

        for i, (xpos, val, alpha) in enumerate(zip(offsets, vals, alphas)):
            if not np.isnan(val):
                ax1.bar(xpos, val, width=width * 0.9,
                        color=STAGE_COLORS[s_idx % len(STAGE_COLORS)],
                        alpha=alpha, linewidth=0)

    ax1.set_ylabel(metric, fontsize=11)
    ax1.set_title("Metric per grasp per stage  (faded = pruned after this stage)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([short_name(g) for g in all_grasps], rotation=40, ha='right', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1.05)

    legend_patches = [
        mpatches.Patch(color=STAGE_COLORS[i], label=f"Stage {s['stage']} (diff={s['difficulty']})")
        for i, s in enumerate(stages)
    ]
    ax1.legend(handles=legend_patches, fontsize=9)

    # -----------------------------------------------------------------------
    # Plot 2: Line chart — trajectory per grasp across stages
    # Grasps that get pruned have their line terminate at the pruning stage.
    # -----------------------------------------------------------------------
    ax2 = axes[1]
    stage_labels = [f"Stage {s['stage']}\ndiff={s['difficulty']}" for s in stages]
    pruned_at    = {}  # grasp -> stage_idx it was pruned
    for s_idx, stage in enumerate(stages):
        for g in stage.get("pruned", []):
            if g not in pruned_at:
                pruned_at[g] = s_idx

    for g in all_grasps:
        is_survivor = g in survivors
        vals = []
        xs   = []
        for s_idx, stage in enumerate(stages):
            if g in stage["metrics"]:
                vals.append(get_metric_val(stage["metrics"], g, metric))
                xs.append(s_idx)

        color     = '#F5A623' if is_survivor else '#94A3B8'
        lw        = 2.5 if is_survivor else 1.2
        alpha     = 0.9 if is_survivor else 0.4
        linestyle = '-' if is_survivor else '--'
        marker    = 'o' if is_survivor else 'x'

        ax2.plot(xs, vals, color=color, linewidth=lw, alpha=alpha,
                 linestyle=linestyle, marker=marker, markersize=6,
                 label=short_name(g) if is_survivor else None)

        # Mark pruning point with a vertical drop to zero
        if g in pruned_at and not no_cull:
            prune_s = pruned_at[g]
            if prune_s < len(vals):
                ax2.plot([prune_s, prune_s], [vals[prune_s], 0],
                         color='#E8563A', linewidth=1, alpha=0.5, linestyle=':')
                ax2.scatter([prune_s], [vals[prune_s]], color='#E8563A',
                            s=60, zorder=5, marker='X')

    ax2.set_ylabel(metric, fontsize=11)
    ax2.set_title("Metric trajectory per grasp  (gold = final survivors, ✕ = pruning point)", fontsize=11)
    ax2.set_xticks(range(n_stages))
    ax2.set_xticklabels(stage_labels, fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)
    if survivors:
        ax2.legend(title="Survivors", fontsize=8, title_fontsize=8,
                   loc='upper left', framealpha=0.6)

    # -----------------------------------------------------------------------
    # Plot 3: Active grasp count waterfall + final ranking table
    # -----------------------------------------------------------------------
    ax3 = axes[2]
    alive_counts = [len(s["metrics"]) for s in stages]
    bars = ax3.bar(range(n_stages), alive_counts,
                   color=[STAGE_COLORS[i] for i in range(n_stages)],
                   alpha=0.85, width=0.5)

    for bar, count in zip(bars, alive_counts):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax3.set_ylabel("Active grasps", fontsize=11)
    ax3.set_title("Active grasps per stage (pruning waterfall)", fontsize=11)
    ax3.set_xticks(range(n_stages))
    ax3.set_xticklabels(stage_labels, fontsize=9)
    ax3.set_ylim(0, n_grasps + 1.5)
    ax3.yaxis.get_major_locator().set_params(integer=True)
    ax3.grid(axis='y', alpha=0.3)

    # Print final ranking as text below the waterfall
    if stages:
        last_metrics = stages[-1]["metrics"]
        def sort_key(item):
            g_name, g_dict = item
            val = get_metric_val(stages[-1]["metrics"], g_name, metric)
            return val if not np.isnan(val) else -1e9
        ranked = sorted(last_metrics.items(), key=sort_key, reverse=True)
        summary_lines = ["Final stage ranking:"]
        for rank, (g, g_dict) in enumerate(ranked, 1):
            val = get_metric_val(stages[-1]["metrics"], g, metric)
            tag = " ✓" if g in survivors else ""
            summary_lines.append(f"  {rank}. {short_name(g)}: {val:.4f}{tag}")
        ax3.text(n_stages - 0.5 + 0.1, n_grasps + 1.3, "\n".join(summary_lines),
                 fontsize=7.5, va='top', ha='left',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path is None:
        output_path = str(Path(results_path).parent / "curriculum_report.png")

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Visualization saved to: {output_path}")

    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="curriculum_results.json")
    parser.add_argument("--output",  type=str, default=None)
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()
    visualize(args.results, args.output, show=not args.no_show)