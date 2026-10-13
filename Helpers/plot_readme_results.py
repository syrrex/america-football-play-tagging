"""
Renders assets/results_tags.png for the README: accuracy of the three tags
under ground-truth input vs. the fully self-detected (end-to-end) pipeline,
read directly from Play_Analysis/*_eval*.json, with each tag's
majority-class baseline computed from the same files.

Usage:
  python plot_readme_results.py
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR    = PROJECT_DIR / "Play_Analysis"
OUT         = PROJECT_DIR / "assets" / "results_tags.png"

SURFACE   = "#fcfcfb"
TEXT      = "#0b0b0b"
TEXT_2    = "#52514e"
GRID      = "#e4e3df"
SERIES    = {"Ground-truth input": "#2a78d6", "End-to-end (self-detected)": "#eb6834"}


def load(name):
    return json.load(open(EVAL_DIR / f"{name}.json"))


def majority(preds, key):
    counts = Counter(p[key] for p in preds)
    return counts.most_common(1)[0][1] / sum(counts.values())


def main():
    f_gt, f_sd = load("formation_eval"), load("formation_eval_selfdetected")
    y_gt, y_sd = load("yard_hash_eval"), load("yard_hash_eval_selfdetected")
    t_gt, t_sd = load("play_type_eval"), load("play_type_eval_selfdetected")

    tags = ["Formation\n(5 classes)", "Hash mark\n(L / M / R)", "Play type\n(run / pass)"]
    gt = [f_gt["accuracy"], y_gt["hash"]["accuracy"], t_gt["accuracy"]]
    sd = [f_sd["accuracy"], y_sd["hash"]["accuracy"], t_sd["accuracy"]]
    base = [majority(f_gt["predictions"], "true_form"),
            majority(y_gt["predictions"], "true_hash"),
            majority(t_gt["predictions"], "true_play_type")]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = np.arange(len(tags))
    w = 0.34
    for i, (label, vals) in enumerate(zip(SERIES, (gt, sd))):
        xs = x + (i - 0.5) * (w + 0.02)
        bars = ax.bar(xs, [v * 100 for v in vals], w, color=SERIES[label], label=label,
                      edgecolor=SURFACE, linewidth=2, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.5, f"{v * 100:.1f}%",
                    ha="center", va="bottom", fontsize=10, color=TEXT)
    for xi, b in zip(x, base):
        ax.plot([xi - 0.4, xi + 0.4], [b * 100] * 2, color=TEXT_2, linestyle=(0, (4, 3)),
                linewidth=1.2, zorder=4)
    ax.plot([], [], color=TEXT_2, linestyle=(0, (4, 3)), linewidth=1.2, label="Majority-class baseline")

    ax.set_xticks(x, tags, color=TEXT)
    ax.set_ylim(0, 110)
    ax.set_yticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)], color=TEXT_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.set_title("Tag accuracy on 107 plays of one game (quarter-based 4-fold CV)",
                 loc="left", color=TEXT, fontsize=12, pad=26)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(l) for l in [*SERIES, "Majority-class baseline"]]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False,
              fontsize=9.5, labelcolor=TEXT_2, handlelength=1.6, borderaxespad=0.2)

    OUT.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, facecolor=SURFACE)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
