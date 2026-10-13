"""
Annotation analysis for Player_Detection_Ravelin dataset.

Expected counts per frame (players on the field):
  defense : 11
  oline   : 5
  qb      : 1
  skill   : 5
  ref     : variable (not counted against expected)

Outputs:
  - Console summary table
  - annotation_analysis.png  (per-class count distributions + missing player heatmap)
  - annotation_analysis.csv  (per-frame counts)
"""

from pathlib import Path
import collections
import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Config ────────────────────────────────────────────────────────────────────
LABEL_DIR   = Path(__file__).parent / "dataset" / "labels"
CLASS_NAMES = ["defense", "oline", "qb", "ref", "skill"]
EXPECTED    = {"defense": 11, "oline": 5, "qb": 1, "skill": 5}   # ref excluded
OUT_PNG     = Path(__file__).parent / "annotation_analysis.png"
OUT_CSV     = Path(__file__).parent / "annotation_analysis.csv"
# ─────────────────────────────────────────────────────────────────────────────


def load_counts() -> list[dict]:
    """Return per-frame annotation counts."""
    rows = []
    for txt in sorted(LABEL_DIR.glob("*.txt")):
        counts = collections.Counter()
        for line in txt.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                counts[CLASS_NAMES[int(parts[0])]] += 1
        rows.append({
            "frame": txt.stem,
            **{cls: counts.get(cls, 0) for cls in CLASS_NAMES},
        })
    return rows


def print_summary(rows: list[dict]):
    n = len(rows)
    print(f"\n{'='*65}")
    print(f"ANNOTATION ANALYSIS  —  {n} frames")
    print(f"{'='*65}")

    col_w = 10
    header = f"{'Class':<10} {'Expected':>{col_w}} {'Mean':>{col_w}} {'Median':>{col_w}} {'Min':>{col_w}} {'Max':>{col_w}} {'Full':>{col_w}} {'Missing':>{col_w}}"
    print(header)
    print("-" * 65)

    for cls in CLASS_NAMES:
        vals = [r[cls] for r in rows]
        exp  = EXPECTED.get(cls, "—")

        mean   = np.mean(vals)
        median = np.median(vals)
        mn     = np.min(vals)
        mx     = np.max(vals)

        if cls in EXPECTED:
            full    = sum(1 for v in vals if v == exp)
            missing = sum(exp - v for v in vals if v < exp)
            print(f"{cls:<10} {exp:>{col_w}} {mean:>{col_w}.2f} {median:>{col_w}.1f} "
                  f"{mn:>{col_w}} {mx:>{col_w}} "
                  f"{full:>{col_w}} {missing:>{col_w}}")
        else:
            print(f"{cls:<10} {'—':>{col_w}} {mean:>{col_w}.2f} {median:>{col_w}.1f} "
                  f"{mn:>{col_w}} {mx:>{col_w}} "
                  f"{'—':>{col_w}} {'—':>{col_w}}")

    print("-" * 65)
    print("Full = frames where count matches expected exactly")
    print("Missing = total missing annotations across all frames")

    # Frames with all expected counts correct
    full_frames = sum(
        1 for r in rows
        if all(r[cls] == EXPECTED[cls] for cls in EXPECTED)
    )
    print(f"\nFrames with ALL player classes fully annotated: {full_frames}/{n} "
          f"({100*full_frames/n:.1f}%)")
    print(f"{'='*65}\n")


def save_csv(rows: list[dict]):
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame"] + CLASS_NAMES +
                                [f"{cls}_missing" for cls in EXPECTED])
        writer.writeheader()
        for r in rows:
            row = dict(r)
            for cls, exp in EXPECTED.items():
                row[f"{cls}_missing"] = max(0, exp - r[cls])
            writer.writerow(row)
    print(f"Per-frame CSV saved to: {OUT_CSV}")


def plot(rows: list[dict]):
    frames     = [r["frame"] for r in rows]
    n          = len(frames)
    x          = np.arange(n)
    plot_classes = ["defense", "oline", "qb", "skill"]   # ref separate

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Annotation Analysis — Player_Detection_Ravelin", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    colors = {"defense": "#e74c3c", "oline": "#3498db", "qb": "#2ecc71",
              "skill": "#f39c12", "ref": "#9b59b6"}

    # ── Row 0: per-class count over frames ───────────────────────────────────
    for i, cls in enumerate(plot_classes):
        ax = fig.add_subplot(gs[0, i % 2]) if i < 2 else fig.add_subplot(gs[1, i % 2])
        vals = [r[cls] for r in rows]
        exp  = EXPECTED[cls]

        ax.bar(x, vals, color=colors[cls], alpha=0.7, label="annotated")
        ax.axhline(exp, color="black", linewidth=1.5, linestyle="--", label=f"expected ({exp})")
        ax.set_title(f"{cls.capitalize()} (expected: {exp})")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Count")
        ax.set_xlim(-1, n)
        ax.set_ylim(0, max(max(vals) + 1, exp + 2))
        ax.legend(fontsize=8)

    # ── Row 1 right: ref count over frames ───────────────────────────────────
    ax_ref = fig.add_subplot(gs[1, 1])
    ref_vals = [r["ref"] for r in rows]
    ax_ref.bar(x, ref_vals, color=colors["ref"], alpha=0.7)
    ax_ref.set_title("Ref (no fixed expected count)")
    ax_ref.set_xlabel("Frame index")
    ax_ref.set_ylabel("Count")
    ax_ref.set_xlim(-1, n)

    # ── Row 2 left: distribution (histogram) of counts per class ─────────────
    ax_hist = fig.add_subplot(gs[2, 0])
    for cls in plot_classes:
        vals = [r[cls] for r in rows]
        exp  = EXPECTED[cls]
        # Normalize by expected so all classes share the same x axis
        pct  = [v / exp * 100 for v in vals]
        ax_hist.hist(pct, bins=20, alpha=0.5, label=cls, color=colors[cls])
    ax_hist.axvline(100, color="black", linewidth=1.5, linestyle="--", label="100% = expected")
    ax_hist.set_title("Distribution of annotation completeness (%)")
    ax_hist.set_xlabel("% of expected count annotated")
    ax_hist.set_ylabel("Number of frames")
    ax_hist.legend(fontsize=8)

    # ── Row 2 right: heatmap of missing players per frame ────────────────────
    ax_heat = fig.add_subplot(gs[2, 1])
    heat_data = np.array([
        [max(0, EXPECTED[cls] - r[cls]) for cls in plot_classes]
        for r in rows
    ]).T  # shape: (4 classes, n_frames)

    im = ax_heat.imshow(heat_data, aspect="auto", cmap="Reds",
                        vmin=0, vmax=max(EXPECTED.values()))
    ax_heat.set_yticks(range(len(plot_classes)))
    ax_heat.set_yticklabels([c.capitalize() for c in plot_classes])
    ax_heat.set_xlabel("Frame index")
    ax_heat.set_title("Missing annotations per frame (darker = more missing)")
    plt.colorbar(im, ax=ax_heat, label="Missing count")

    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {OUT_PNG}")
    plt.show()


def main():
    rows = load_counts()
    print_summary(rows)
    save_csv(rows)
    plot(rows)


if __name__ == "__main__":
    main()
