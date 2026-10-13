"""
Lists every misclassified clip from the Play-Type Classification eval
(GT-init and self-detected, both configs), with the QB/oline slope
values that drove the prediction -- so a clip can be pulled up and watched
to see what actually happens against the numbers.

Reads the already-computed play_type_eval*.json files; does not re-run
tracking or detection.

Usage:
  python list_misclassified_playtype.py
"""

import json
from pathlib import Path

PLAY_ANALYSIS = Path(__file__).resolve().parent.parent

CONFIGS = {
    "GT-init":        PLAY_ANALYSIS / "play_type_eval.json",
    "Self-detected":  PLAY_ANALYSIS / "play_type_eval_selfdetected.json",
}


def fmt_slope(v):
    return "None" if v is None else f"{v:+.3f}"


def main():
    for name, path in CONFIGS.items():
        if not path.exists():
            print(f"[{name}] {path.name} not found, skipping")
            continue
        d = json.loads(path.read_text())
        preds = d["predictions"]
        mis = [r for r in preds if r["true_play_type"] != r["pred_play_type"]]

        print(f"=== {name}: {len(mis)}/{len(preds)} misclassified ===")
        print(f"{'clip':16s} {'true':6s} {'pred':6s} {'qb_slope':14s} {'oline_slope':s}")
        for r in sorted(mis, key=lambda r: (r["true_play_type"], r["clip"])):
            pred = r["pred_play_type"]
            qb_str = fmt_slope(r.get("qb_slope_yd_s"))
            ol_str = fmt_slope(r.get("oline_slope_yd_s"))
            print(f"{r['clip']:16s} {r['true_play_type']:6s} {pred:6s} {qb_str:14s} {ol_str}")
        print()


if __name__ == "__main__":
    main()
