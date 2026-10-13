"""
Scientific analysis — run AFTER experiment_grid.py completes.

Sections:
  1. Results heatmap (model × imgsz, baseline vs finetuned)
  2. Speed vs accuracy Pareto plot
  3. Per-class comparison bar charts
  4. PR curves, F1 curves, confusion matrices (collected from run folders)
  5. Statistical significance tests (Wilcoxon signed-rank, per-fold paired)
  6. Bounding box size analysis (COCO small / medium / large)
  7. Qualitative error analysis (worst frames visualised)

All figures saved to:  scientific_analysis/
"""

import json
import shutil
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import cv2
from pathlib import Path
from collections import Counter, defaultdict
from scipy.stats import wilcoxon, ttest_rel
from sklearn.model_selection import KFold
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent
DATASET      = BASE / "dataset"
GRID_DIR     = BASE / "grid_results"
OUT_DIR      = BASE / "scientific_analysis"
CLASS_NAMES  = ["defense", "oline", "qb", "ref", "skill"]
MODEL_SIZES  = ["n", "s", "m", "l", "x"]
IMAGE_SIZES  = [640, 1280]
K            = 5
DEVICE       = 0

COLORS = {
    "defense": "#e74c3c", "oline": "#3498db", "qb": "#2ecc71",
    "ref": "#9b59b6",    "skill": "#f39c12",
}
SIZE_COLORS  = {"baseline": "#95a5a6", "finetuned": "#2980b9"}
# ─────────────────────────────────────────────────────────────────────────────


def load_grid() -> dict:
    path = GRID_DIR / "grid_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"grid_summary.json not found. Run experiment_grid.py first.")
    return json.loads(path.read_text())


def weights_path(size: str) -> str:
    local = {
        "n": "/mnt/c/Users/Lukas/Documents/Semester_5/Seminar_Project/object_detection/yolo11n.pt",
        "m": "/mnt/c/Users/Lukas/Documents/Semester_5/Seminar_Project/object_detection/yolo11m.pt",
    }
    return local.get(size, f"yolo11{size}.pt")


# ── 1. Results heatmaps ───────────────────────────────────────────────────────
def plot_heatmaps(grid: dict):
    starts = ["coco", "roboflow"]
    modes  = ["baseline", "finetuned_avg"]
    titles = {
        ("coco",     "baseline"):      "COCO — Baseline (zero-shot)",
        ("coco",     "finetuned_avg"): "COCO — Finetuned (k-fold avg)",
        ("roboflow", "baseline"):      "Roboflow — Baseline (zero-shot)",
        ("roboflow", "finetuned_avg"): "Roboflow — Finetuned (k-fold avg)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("mAP50 — Model × Image Size", fontsize=13, fontweight="bold")

    for row, start in enumerate(starts):
        for col, mode in enumerate(modes):
            ax = axes[row][col]
            data = np.zeros((len(MODEL_SIZES), len(IMAGE_SIZES)))
            for i, s in enumerate(MODEL_SIZES):
                for j, isz in enumerate(IMAGE_SIZES):
                    key = f"{start}_yolo11{s}_{isz}"
                    if key in grid:
                        data[i, j] = grid[key][mode]["overall"]["mAP50"]

            vmin = 0.0 if mode == "baseline" else 0.4
            im = ax.imshow(data, cmap="RdYlGn", vmin=vmin, vmax=1.0, aspect="auto")
            ax.set_xticks(range(len(IMAGE_SIZES)))
            ax.set_xticklabels(IMAGE_SIZES)
            ax.set_yticks(range(len(MODEL_SIZES)))
            ax.set_yticklabels([f"yolo11{s}" for s in MODEL_SIZES])
            ax.set_xlabel("Image size")
            ax.set_title(titles[(start, mode)])
            for i in range(len(MODEL_SIZES)):
                for j in range(len(IMAGE_SIZES)):
                    ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center",
                            fontsize=9, color="black")
            plt.colorbar(im, ax=ax, label="mAP50")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "1_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓  heatmaps")


# ── 2. Accuracy vs model complexity ──────────────────────────────────────────
PARAM_COUNT = {"n": 2.6, "s": 9.4, "m": 20.1, "l": 25.3, "x": 56.8}

def plot_pareto(grid: dict):
    fig, ax = plt.subplots(figsize=(11, 6))
    # color = starting point, marker = resolution
    start_colors = {"coco": "#e67e22", "roboflow": "#2980b9"}
    isz_markers  = {640: "o", 1280: "s"}
    labels_done  = set()

    for start, color in start_colors.items():
        for isz, marker in isz_markers.items():
            for s in MODEL_SIZES:
                key = f"{start}_yolo11{s}_{isz}"
                if key not in grid:
                    continue
                o     = grid[key]["finetuned_avg"]["overall"]
                label = f"{start} / {isz}\,px"
                ax.scatter(PARAM_COUNT[s], o["mAP50"],
                           c=color, marker=marker, s=130,
                           label=label if label not in labels_done else "_",
                           zorder=3, alpha=0.85)
                labels_done.add(label)
                ax.annotate(f"11{s}", (PARAM_COUNT[s], o["mAP50"]),
                            textcoords="offset points", xytext=(5, 4), fontsize=8)

    ax.set_xlabel("Model parameters (M)", fontsize=11)
    ax.set_ylabel("mAP50", fontsize=11)
    ax.set_title("Accuracy vs.\ Model Complexity — finetuned models, all configurations",
                 fontsize=12)
    ax.legend(fontsize=9, title="Start / Resolution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "2_pareto.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓  accuracy vs complexity plot")


# ── 3. Per-class bar charts ───────────────────────────────────────────────────
def plot_per_class(grid: dict):
    # Use roboflow start, imgsz=640 — shows baseline vs finetuned per model size
    start   = "roboflow"
    metrics = ["mAP50", "mAP50-95", "precision", "recall"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Per-class metrics — {start} start, imgsz=640 (light=baseline, dark=finetuned)",
                 fontsize=12, fontweight="bold")

    x     = np.arange(len(CLASS_NAMES))
    width = 0.15

    for ax, metric in zip(axes.flat, metrics):
        for i, s in enumerate(MODEL_SIZES):
            key = f"{start}_yolo11{s}_640"
            if key not in grid:
                continue
            vals_ft = [grid[key]["finetuned_avg"]["per_class"][c][metric] for c in CLASS_NAMES]
            vals_bl = [grid[key]["baseline"]["per_class"][c][metric] for c in CLASS_NAMES]
            offset  = (i - 2) * width
            ax.bar(x + offset - width/2, vals_bl, width * 0.9,
                   alpha=0.35, color=f"C{i}", label=f"11{s} base" if metric == "mAP50" else "_")
            ax.bar(x + offset + width/2, vals_ft, width * 0.9,
                   alpha=0.9,  color=f"C{i}", label=f"11{s} ft"   if metric == "mAP50" else "_")

        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        if metric == "mAP50":
            ax.legend(fontsize=7, ncol=5)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "3_per_class.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓  per-class bar charts")


# ── 4. Collect PR / F1 / confusion matrix plots ───────────────────────────────
def collect_plots(grid: dict):
    """Arrange auto-generated Ultralytics plots into comparison figures."""
    plot_types = [
        ("BoxPR_curve.png",              "PR Curve"),
        ("BoxF1_curve.png",              "F1 Curve"),
        ("confusion_matrix_normalized.png", "Confusion Matrix (norm.)"),
    ]

    for fname, title in plot_types:
        # Collect one plot per model size (imgsz=640, finetuned fold_1 as representative)
        paths = {}
        for s in MODEL_SIZES:
            # prefer roboflow start; fall back to coco
            for start in ["roboflow", "coco"]:
                p = GRID_DIR / f"{start}_yolo11{s}_640" / "fold_1" / fname
                if p.exists():
                    paths[s] = p
                    break

        if not paths:
            print(f"  [skip] {fname} — no files found yet")
            continue

        n     = len(paths)
        nrows = 2
        ncols = (n + nrows - 1) // nrows
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
        fig.suptitle(f"{title} — yolo11 models (imgsz=640, fold 1)", fontsize=12)

        for ax, (s, p) in zip(axes_flat, paths.items()):
            img = plt.imread(str(p))
            ax.imshow(img)
            ax.set_title(f"yolo11{s}")
            ax.axis("off")

        # hide unused cells
        for ax in list(axes_flat)[n:]:
            ax.axis("off")

        plt.tight_layout()
        safe_name = fname.replace(".png", "").replace(" ", "_")
        fig.savefig(OUT_DIR / f"4_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓  {title}")


# ── 5. Statistical significance ───────────────────────────────────────────────
def statistical_tests(grid: dict, all_images: list):
    """
    For each (model, imgsz): evaluate BASELINE model on each fold's val set
    to get K paired measurements, then run Wilcoxon signed-rank test.
    """
    print("\n  Running per-fold baseline evaluations for significance tests...")
    kf     = KFold(n_splits=K, shuffle=True, random_state=42)
    splits = [(val_idx) for _, val_idx in kf.split(all_images)]

    results_table = []

    for start in ["coco", "roboflow"]:
      for s in MODEL_SIZES:
        for isz in IMAGE_SIZES:
            key = f"{start}_yolo11{s}_{isz}"
            if key not in grid:
                continue

            baseline_scores  = []
            finetuned_scores = []
            batch = 8

            for fold_idx, val_idx in enumerate(splits):
                fold_num  = fold_idx + 1
                val_imgs  = [all_images[i] for i in val_idx]

                # ── Baseline on this fold's val set ──
                cache_dir = GRID_DIR / key / f"sig_baseline_fold_{fold_num}"
                cache     = cache_dir / "metrics.json"
                if cache.exists():
                    b_map = json.loads(cache.read_text())["mAP50"]
                else:
                    tmp_dir  = cache_dir / "data"
                    img_out  = tmp_dir / "images" / "val"
                    lbl_out  = tmp_dir / "labels" / "val"
                    img_out.mkdir(parents=True, exist_ok=True)
                    lbl_out.mkdir(parents=True, exist_ok=True)
                    for img in val_imgs:
                        shutil.copy(img, img_out / img.name)
                        lbl = DATASET / "labels" / (img.stem + ".txt")
                        if lbl.exists():
                            shutil.copy(lbl, lbl_out / lbl.name)
                    yaml = tmp_dir / "data.yaml"
                    yaml.write_text(
                        f"path: {tmp_dir}\ntrain: images/val\nval: images/val\n"
                        f"\nnc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n"
                    )
                    model   = YOLO(weights_path(s))
                    res     = model.val(data=str(yaml), imgsz=isz, batch=batch,
                                        device=DEVICE, split="val", verbose=False,
                                        plots=False)
                    b_map   = round(float(res.box.map50), 4)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps({"mAP50": b_map}))

                baseline_scores.append(b_map)

                # ── Finetuned score for this fold ──
                fold_data = next(
                    (m for m in grid[key]["folds"] if m["fold"] == fold_num), None
                )
                if fold_data:
                    finetuned_scores.append(fold_data["overall"]["mAP50"])

            if len(baseline_scores) < 2 or len(finetuned_scores) < 2:
                continue

            b = np.array(baseline_scores)
            f = np.array(finetuned_scores)
            diff = f - b

            try:
                stat, p_wilcoxon = wilcoxon(f, b, alternative="greater")
            except Exception:
                p_wilcoxon = float("nan")

            _, p_ttest = ttest_rel(f, b)

            results_table.append({
                "model":        f"yolo11{s}",
                "start":        start,
                "imgsz":        isz,
                "baseline_mean": round(float(b.mean()), 4),
                "finetuned_mean": round(float(f.mean()), 4),
                "delta_mean":   round(float(diff.mean()), 4),
                "delta_std":    round(float(diff.std()), 4),
                "p_wilcoxon":   round(float(p_wilcoxon), 4),
                "p_ttest":      round(float(p_ttest), 4),
                "significant":  bool(p_wilcoxon < 0.05) if not np.isnan(p_wilcoxon) else False,
            })

    # Print table
    print(f"\n  {'Model':<10} {'imgsz':>6}  {'Base':>7}  {'FT':>7}  {'Δ':>7}  {'p(Wilcox)':>10}  {'p(t-test)':>10}  {'Sig?':>5}")
    print("  " + "-" * 70)
    for r in results_table:
        sig = "✓" if r["significant"] else "✗"
        print(f"  {r['model']:<10} {r['imgsz']:>6}  {r['baseline_mean']:>7.4f}  "
              f"{r['finetuned_mean']:>7.4f}  {r['delta_mean']:>+7.4f}  "
              f"{r['p_wilcoxon']:>10.4f}  {r['p_ttest']:>10.4f}  {sig:>5}")

    # Save + plot
    with open(OUT_DIR / "5_significance.json", "w") as f:
        json.dump(results_table, f, indent=2)

    # Bar chart: delta mAP50 ± std with significance markers
    fig, ax = plt.subplots(figsize=(12, 5))
    xs      = range(len(results_table))
    deltas  = [r["delta_mean"] for r in results_table]
    stds    = [r["delta_std"]  for r in results_table]
    colors  = ["#27ae60" if r["significant"] else "#e74c3c" for r in results_table]
    labels  = [f"11{r['model'][-1]}\n{r['imgsz']}" for r in results_table]

    bars = ax.bar(xs, deltas, color=colors, alpha=0.8, yerr=stds, capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Δ mAP50 (finetuned − baseline)")
    ax.set_title("Improvement from finetuning (green = p<0.05 significant)")
    ax.grid(True, axis="y", alpha=0.3)

    for x, r in zip(xs, results_table):
        marker = "★" if r["significant"] else ""
        ax.text(x, max(0, r["delta_mean"]) + r["delta_std"] + 0.005,
                marker, ha="center", fontsize=12, color="#27ae60")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "5_significance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓  significance tests")
    return results_table


# ── 6. Bounding box size analysis ─────────────────────────────────────────────
def bbox_size_analysis():
    label_dir = DATASET / "labels"
    image_dir = DATASET / "images"

    # Get image dimensions from first image
    sample_img = next(image_dir.glob("*.jpg"))
    h_img, w_img = cv2.imread(str(sample_img)).shape[:2]

    size_bins   = {"small": 0, "medium": 0, "large": 0}
    class_sizes = {cls: {"small": 0, "medium": 0, "large": 0} for cls in CLASS_NAMES}

    for txt in label_dir.glob("*.txt"):
        for line in txt.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            cls     = CLASS_NAMES[int(parts[0])]
            bw_px   = float(parts[3]) * w_img
            bh_px   = float(parts[4]) * h_img
            area_px = bw_px * bh_px

            if area_px < 32 ** 2:
                cat = "small"
            elif area_px < 96 ** 2:
                cat = "medium"
            else:
                cat = "large"

            size_bins[cat]       += 1
            class_sizes[cls][cat] += 1

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Bounding box size distribution  "
                 f"(image: {w_img}×{h_img} px)", fontsize=12)

    # Overall pie
    ax = axes[0]
    ax.pie(size_bins.values(), labels=size_bins.keys(),
           autopct="%1.1f%%", colors=["#e74c3c", "#f39c12", "#2ecc71"])
    ax.set_title("Overall (COCO thresholds: <32², 32–96², >96²)")

    # Per-class stacked bar
    ax = axes[1]
    cats  = ["small", "medium", "large"]
    cat_colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    bottoms = np.zeros(len(CLASS_NAMES))
    for cat, color in zip(cats, cat_colors):
        vals = [class_sizes[cls][cat] for cls in CLASS_NAMES]
        ax.bar(CLASS_NAMES, vals, bottom=bottoms, color=color, label=cat, alpha=0.85)
        bottoms += np.array(vals)
    ax.set_title("Per class")
    ax.set_ylabel("Box count")
    ax.legend()

    plt.tight_layout()
    fig.savefig(OUT_DIR / "6_bbox_sizes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓  bbox size analysis")
    print(f"     Overall: {size_bins}")
    return size_bins, class_sizes


# ── 7. Qualitative error analysis ─────────────────────────────────────────────
def qualitative_analysis(grid: dict, all_images: list, n_worst: int = 6):
    """
    Find the best (model, imgsz) combo, run inference on all images,
    identify worst frames (lowest recall), and visualise GT vs predictions.
    """
    # Find best finetuned combo
    best_key = max(
        [k for k in grid if "folds" in grid[k]],
        key=lambda k: grid[k]["finetuned_avg"]["overall"]["mAP50"]
    )
    _, model_str, isz_str = best_key.split("_")
    size = model_str.replace("yolo11", "")
    isz  = int(isz_str)
    print(f"  Best model for qualitative analysis: {best_key}")

    # Find best fold weights
    best_fold = max(grid[best_key]["folds"],
                    key=lambda m: m["overall"]["mAP50"])["fold"]
    model_path = GRID_DIR / best_key / f"fold_{best_fold}" / "weights" / "best.pt"
    if not model_path.exists():
        print(f"  [skip] weights not found: {model_path}")
        return

    model = YOLO(str(model_path))

    # Run inference on all images and compute per-image recall
    per_image = []
    for img_path in all_images:
        lbl_path = DATASET / "labels" / (img_path.stem + ".txt")
        gt_lines = lbl_path.read_text().splitlines() if lbl_path.exists() else []
        n_gt     = len([l for l in gt_lines if l.strip()])

        preds    = model.predict(str(img_path), imgsz=isz,
                                 conf=0.25, verbose=False)[0]
        n_pred   = len(preds.boxes) if preds.boxes is not None else 0

        # Simple recall proxy: how many GT boxes were roughly matched
        # (proper IoU matching would need more code; this gives an ordering)
        recall_proxy = min(n_pred / max(n_gt, 1), 1.0)
        per_image.append({
            "path":     img_path,
            "n_gt":     n_gt,
            "n_pred":   n_pred,
            "recall":   recall_proxy,
            "preds":    preds,
        })

    # Sort by recall (worst first)
    per_image.sort(key=lambda x: x["recall"])
    worst = per_image[:n_worst]

    # Visualise
    cols = 3
    rows = (n_worst + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    fig.suptitle(f"Worst-performing frames — {best_key} (fold {best_fold})\n"
                 f"Green=GT  Red=Prediction", fontsize=12, fontweight="bold")

    cls_colors = list(COLORS.values())

    for ax, item in zip(axes.flat, worst):
        img_bgr = cv2.imread(str(item["path"]))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w    = img_rgb.shape[:2]

        # Draw GT boxes (green)
        lbl_path = DATASET / "labels" / (item["path"].stem + ".txt")
        for line in lbl_path.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = (float(parts[1]) * w, float(parts[2]) * h,
                               float(parts[3]) * w, float(parts[4]) * h)
            x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
            x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 200, 0), 2)

        # Draw predicted boxes (red)
        if item["preds"].boxes is not None:
            for box in item["preds"].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id  = int(box.cls[0])
                conf    = float(box.conf[0])
                color   = (220, 50, 50)
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_rgb,
                            f"{CLASS_NAMES[cls_id]} {conf:.2f}",
                            (x1, max(y1 - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        ax.imshow(img_rgb)
        ax.set_title(f"{item['path'].stem}  GT={item['n_gt']}  Pred={item['n_pred']}",
                     fontsize=9)
        ax.axis("off")

    # Hide unused axes
    for ax in axes.flat[len(worst):]:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "7_qualitative_errors.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓  qualitative error analysis ({n_worst} worst frames)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid       = load_grid()
    all_images = sorted((DATASET / "images").glob("*.jpg"))
    print(f"Loaded grid results: {len(grid)} experiments")
    print(f"Dataset: {len(all_images)} images\n")

    print("1. Heatmaps...")
    plot_heatmaps(grid)

    print("2. Speed vs accuracy...")
    plot_pareto(grid)

    print("3. Per-class bar charts...")
    plot_per_class(grid)

    print("4. Collecting PR / F1 / confusion matrix plots...")
    collect_plots(grid)

    print("5. Statistical significance tests...")
    statistical_tests(grid, all_images)

    print("6. Bounding box size analysis...")
    bbox_size_analysis()

    print("7. Qualitative error analysis...")
    qualitative_analysis(grid, all_images)

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
