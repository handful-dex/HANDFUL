import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ─── STYLE ────────────────────────────────────────────────────────────────────
sns.set_theme(
    style="whitegrid",
    palette="pastel",
    rc={"font.family": "serif", "font.serif": ["Palatino"]},
)
BASE_FONT_SIZE = 50

# ─── CONFIG ────────────────────────────────────────────────────────────────────
# Just drop your run folders here — one folder per stage, one for baseline.
# Each folder should contain one subfolder per seed.
STAGE_FOLDERS = [
    "runs/two_pick_curric_seeds_10-25/stage0",
    "runs/two_pick_curric_seeds_10-25/stage1",
    "runs/two_pick_curric_seeds_10-25/stage2",
]
BASELINE_FOLDER = "runs/two_pick_seeds_10-25"

LEARNING_STARTS_PER_STAGE = [0, 409_600, 409_600]
SMOOTH_WINDOW = 4
OUTPUT_FILE = "curriculum_vs_baseline"
METRIC_TAG = "train/success_at_end"


# ─── HELPERS ───────────────────────────────────────────────────────────────────


def find_run_dirs(folder: str) -> list[Path]:
    """Return sorted list of dirs containing tfevents files, searched recursively."""
    parent = Path(folder)
    runs = sorted(set(
        p.parent for p in parent.rglob("events.out.tfevents.*")
    ))
    return runs

def load_scalar_series(run_dir: Path, tag: str):
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return None, None
    ea = event_accumulator.EventAccumulator(str(event_files[0]))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None, None
    events = ea.Scalars(tag)
    return np.array([e.step for e in events]), np.array([e.value for e in events])


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    smoothed = np.zeros_like(values)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed[i] = values[start : i + 1].mean()
    return smoothed


def process_curriculum_seed(stage_run_dirs: list[Path], tag, learning_starts, window):
    """Stitch stages for a single seed; smooth each stage before stitching."""
    all_steps, all_values = [], []
    boundaries = []
    step_offset = 0.0

    for i, run_dir in enumerate(stage_run_dirs):
        steps, values = load_scalar_series(run_dir, tag)
        if steps is None:
            continue

        mask = steps >= learning_starts[i]
        s, v = steps[mask], values[mask]
        v = smooth(v, window)

        if i > 0:
            shift = step_offset - s[0]
            s = s + shift
            boundaries.append(s[0])
            all_steps.append(np.array([s[0]]))
            all_values.append(np.array([np.nan]))

        all_steps.append(s)
        all_values.append(v)
        all_steps.append(np.array([s[-1]]))
        all_values.append(np.array([np.nan]))
        step_offset = s[-1]

    if not all_steps:
        return None, None, boundaries

    return np.concatenate(all_steps), np.concatenate(all_values), boundaries


# ─── MAIN PLOT ─────────────────────────────────────────────────────────────────

def main():
    # Discover runs: each stage folder has N seed subdirs (matched by sort order)
    stage_run_lists = [find_run_dirs(f) for f in STAGE_FOLDERS]
    baseline_runs = find_run_dirs(BASELINE_FOLDER)

    # Validate seed counts match across stages
    n_seeds = len(stage_run_lists[0])
    for i, runs in enumerate(stage_run_lists):
        assert len(runs) == n_seeds, (
            f"Stage {i} has {len(runs)} seeds but Stage 0 has {n_seeds}. "
            "Each stage folder must have the same number of seed subdirs."
        )

    print(f"Found {n_seeds} seeds across {len(STAGE_FOLDERS)} stages "
          f"and {len(baseline_runs)} baseline runs.")

    fig, ax = plt.subplots(figsize=(22, 11))
    plot_data = []
    all_boundaries = []

    # Process Curriculum (group stage dirs by seed index)
    for seed_id in range(n_seeds):
        stages_for_seed = [stage_run_lists[stage][seed_id] for stage in range(len(STAGE_FOLDERS))]
        steps, values, bounds = process_curriculum_seed(
            stages_for_seed, METRIC_TAG, LEARNING_STARTS_PER_STAGE, SMOOTH_WINDOW
        )
        if steps is None:
            continue
        for s, v in zip(steps, values):
            plot_data.append({"Step": s, "Value": v, "Method": "Curri.", "Seed": seed_id})
        all_boundaries.append(bounds)

    # Process Baseline
    for seed_id, run_dir in enumerate(baseline_runs):
        steps, values = load_scalar_series(run_dir, METRIC_TAG)
        if steps is not None:
            values = smooth(values, SMOOTH_WINDOW)
            for s, v in zip(steps, values):
                plot_data.append({"Step": s, "Value": v, "Method": "w/o Curri.", "Seed": seed_id})

    df = pd.DataFrame(plot_data)

    # Stage boundary lines
    if all_boundaries:
        for i, b_step in enumerate(all_boundaries[0]):
            ax.axvline(x=b_step, color="gray", linestyle="--", linewidth=3, alpha=0.7)
            ax.text(b_step, 1.2, f"Stage {i+1}", rotation=90, color="gray",
                    fontsize=BASE_FONT_SIZE - 6, ha="right", va="top")

    sns.lineplot(
        data=df, x="Step", y="Value", hue="Method",
        palette=["#4A90E2", "#E35E52"],
        linewidth=8, errorbar="sd", ax=ax,
    )

    ax.set_title("Success Rate vs. Training Steps for Pick Second Task",
                 fontsize=BASE_FONT_SIZE + 4, pad=40)
    ax.set_xlabel("Total Training Steps", fontsize=BASE_FONT_SIZE)
    ax.set_ylabel("Success Rate", fontsize=BASE_FONT_SIZE)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.set_ylim(-0.02, 1.2)
    ax.tick_params(axis="both", labelsize=BASE_FONT_SIZE - 2)

    sns.despine(trim=True)
    ax.legend(loc="upper left", frameon=True, fontsize=BASE_FONT_SIZE - 12,
              facecolor="white", framealpha=1)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FILE}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"{OUTPUT_FILE}.png", format="png", dpi=600, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
