"""
Learning curve analysis — how does mAP50 scale with training data size?

Trains yolo11n @ imgsz=640 on increasing fractions of the dataset,
using K-fold cross-validation at each fraction to get error bars.

Output:
  learning_curve/learning_curve.png
  learning_curve/learning_curve.json
"""

import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
DATASET     = BASE / "dataset"
OUT_DIR     = BASE / "learning_curve"
MODEL       = "yolo11n.pt"          # fixed model for curve
IMGSZ       = 640
BATCH       = 16
DEVICE      = 0
K           = 5
EPOCHS      = 40                    # fewer epochs to keep runtime manageable
FRACTIONS   = [0.2, 0.4, 0.6, 0.8, 1.0]
CLASS_NAMES = ["defense", "oline", "qb", "ref", "skill"]
# ─────────────────────────────────────────────────────────────────────────────


def make_split_yaml(out_dir: Path, train_imgs: list, val_imgs: list) -> Path:
    for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
        img_out = out_dir / "images" / split
        lbl_out = out_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy(img, img_out / img.name)
            lbl = DATASET / "labels" / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy(lbl, lbl_out / lbl.name)
    yaml = out_dir / "data.yaml"
    yaml.write_text(
        f"path: {out_dir}\ntrain: images/train\nval: images/val\n"
        f"\nnc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n"
    )
    return yaml


def run_fraction(all_images: list, fraction: float) -> dict:
    """K-fold CV at a given data fraction. Returns mean/std mAP50."""
    cache_path = OUT_DIR / f"fraction_{int(fraction*100)}.json"
    if cache_path.exists():
        print(f"  [cache] fraction={fraction:.0%}")
        return json.loads(cache_path.read_text())

    kf          = KFold(n_splits=K, shuffle=True, random_state=42)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_images)):
        fold_num = fold_idx + 1

        # Sub-sample the training set by fraction
        n_train   = max(1, int(len(train_idx) * fraction))
        rng       = np.random.default_rng(seed=42 + fold_idx)
        sub_idx   = rng.choice(train_idx, size=n_train, replace=False)

        train_imgs = [all_images[i] for i in sub_idx]
        val_imgs   = [all_images[i] for i in val_idx]

        print(f"  Fraction {fraction:.0%}  Fold {fold_num}/{K}: "
              f"train={len(train_imgs)}  val={len(val_imgs)}")

        run_dir  = OUT_DIR / f"frac{int(fraction*100)}_fold{fold_num}"
        data_yaml = make_split_yaml(run_dir, train_imgs, val_imgs)

        model = YOLO(MODEL)
        model.train(
            data=str(data_yaml), epochs=EPOCHS, batch=BATCH,
            imgsz=IMGSZ, device=DEVICE,
            project=str(OUT_DIR), name=f"frac{int(fraction*100)}_fold{fold_num}",
            exist_ok=True,
            lr0=0.001, lrf=0.01, cos_lr=True,
            warmup_epochs=2, patience=10,
            mosaic=1.0, close_mosaic=5,
            fliplr=0.5, flipud=0.0,
            save=True, save_period=-1, verbose=False,
            plots=False,
        )

        best = OUT_DIR / f"frac{int(fraction*100)}_fold{fold_num}" / "weights" / "best.pt"
        finetuned = YOLO(str(best))
        val_results = finetuned.val(
            data=str(data_yaml), imgsz=IMGSZ, batch=BATCH,
            device=DEVICE, split="val", verbose=False, plots=False,
        )
        fold_scores.append(float(val_results.box.map50))
        print(f"    mAP50 = {fold_scores[-1]:.4f}")

    result = {
        "fraction":   fraction,
        "n_images":   int(len(all_images) * fraction),
        "map50_mean": round(float(np.mean(fold_scores)), 4),
        "map50_std":  round(float(np.std(fold_scores)),  4),
        "map50_min":  round(float(np.min(fold_scores)),  4),
        "map50_max":  round(float(np.max(fold_scores)),  4),
        "folds":      [round(s, 4) for s in fold_scores],
    }
    cache_path.write_text(json.dumps(result, indent=2))
    return result


def plot_curve(results: list[dict]):
    fig, ax = plt.subplots(figsize=(9, 5))

    n_images = [r["n_images"] for r in results]
    means    = [r["map50_mean"] for r in results]
    stds     = [r["map50_std"]  for r in results]

    ax.plot(n_images, means, "o-", color="#2980b9", linewidth=2,
            markersize=7, label="mAP50 (mean across folds)")
    ax.fill_between(n_images,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.25, color="#2980b9", label="±1 std")

    # Annotate each point
    for r in results:
        ax.annotate(f"{r['map50_mean']:.3f}",
                    (r["n_images"], r["map50_mean"]),
                    textcoords="offset points", xytext=(5, 6), fontsize=9)

    ax.set_xlabel("Number of training images", fontsize=11)
    ax.set_ylabel("mAP50", fontsize=11)
    ax.set_title("Learning curve — yolo11n @ 640\n"
                 f"(K={K} folds, {EPOCHS} epochs each)", fontsize=12)
    ax.set_xticks(n_images)
    ax.set_xticklabels([f"{r['fraction']:.0%}\n({r['n_images']} img)" for r in results])
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved to: {OUT_DIR / 'learning_curve.png'}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_images = sorted((DATASET / "images").glob("*.jpg"))
    print(f"Dataset: {len(all_images)} images")
    print(f"Fractions: {[f'{f:.0%}' for f in FRACTIONS]}")
    print(f"Total training runs: {len(FRACTIONS) * K}\n")

    results = []
    for fraction in FRACTIONS:
        print(f"\n{'='*50}")
        print(f"  Fraction: {fraction:.0%} ({int(len(all_images)*fraction)} images)")
        print(f"{'='*50}")
        r = run_fraction(all_images, fraction)
        results.append(r)
        print(f"  → mAP50: {r['map50_mean']:.4f} ± {r['map50_std']:.4f}")

    with open(OUT_DIR / "learning_curve.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_curve(results)

    print("\nLearning curve summary:")
    print(f"  {'Fraction':>10}  {'N images':>9}  {'mAP50':>7}  {'±std':>6}")
    for r in results:
        print(f"  {r['fraction']:>10.0%}  {r['n_images']:>9}  "
              f"{r['map50_mean']:>7.4f}  {r['map50_std']:>6.4f}")


if __name__ == "__main__":
    main()
