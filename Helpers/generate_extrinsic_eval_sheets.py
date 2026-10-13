"""
Generates the four Hudl-style .xlsx tagging sheets for the extrinsic
evaluation (thesis-template/sections/7_3_additional_experiments.tex,
sec:extrinsic_evaluation): a small timed comparison of manual tagging vs.
reviewing the pipeline's self-detected tags, run with two coaches.

Design (counterbalanced 2x2):
  - 50 offense clips (ODK=O) + 50 defense clips (ODK=D) sampled from the 107
    tb2026 clips (56 O / 51 D available).
  - Coach 1: tags the offense set manually, reviews the defense set's
    self-detected tags.
  - Coach 2: tags the defense set manually, reviews the offense set's
    self-detected tags.
  - Counterbalanced so neither coach only ever does one condition, and the
    two conditions are never compared across different clip sets for the
    same coach (avoids the same-team/order confounds discussed in chat).

Only the three tags the pipeline actually produces, OFF FORM, YARD LN
(+HASH), and PLAY TYPE, are pre-filled in the "review" sheets, pulled
directly from the already-computed self-detected evaluation results
(formation_eval_selfdetected.json, yard_hash_eval_selfdetected.json,
play_type_eval_selfdetected.json -- real self-detected pipeline output,
errors included, not idealized). Every other Hudl column stays blank in
both conditions, since the pipeline does not produce them and they are
outside this comparison's scope (see the Introduction's scope-boundary
discussion: down/distance, gain/loss, and play result are not recovered by
this pipeline).

All 19 standard Hudl columns are kept, in the same order as the reference
export (Data/metadata/PlaylistData_2026-06-03.xlsx), so the sheets stay
upload-compatible. PLAY # and ODK are pre-filled the same way the real
data file already has them; everything else follows the manual/review
split above. A extra CLIP column is appended at the end (not part of the
standard Hudl schema) so a coach can find the actual video file for each
row.

Usage:
  python generate_extrinsic_eval_sheets.py
"""

import json
import random
from pathlib import Path

import openpyxl
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
PLAY_ANALYSIS = PROJECT_DIR / "Play_Analysis"
META_CSV = PROJECT_DIR / "Data" / "metadata" / "tb2026_dataset_metadata.csv"
OUT_DIR = PLAY_ANALYSIS / "extrinsic_evaluation"

SEED = 0
N_PER_SIDE = 50

HUDL_COLUMNS = [
    "PLAY #", "ODK", "DN", "DIST", "HASH", "YARD LN", "PLAY TYPE", "RESULT",
    "GN/LS", "OFF FORM", "OFF PLAY", "OFF STR", "PLAY DIR", "GAP",
    "PASS ZONE", "DEF FRONT", "COVERAGE", "BLITZ", "QTR",
]
EXTRA_COLUMNS = ["CLIP"]  # not part of the Hudl schema, appended for the coach's own use

# The four tagged columns the pipeline actually produces, pre-filled only
# in the "review" condition.
TAGGED_COLUMNS = {"HASH", "YARD LN", "PLAY TYPE", "OFF FORM"}


def load_predictions() -> dict[str, dict]:
    """Returns {clip_filename: {"OFF FORM":..., "YARD LN":..., "HASH":..., "PLAY TYPE":...}}."""
    formation = json.load(open(PLAY_ANALYSIS / "formation_eval_selfdetected.json"))
    yard_hash = json.load(open(PLAY_ANALYSIS / "yard_hash_eval_selfdetected.json"))
    play_type = json.load(open(PLAY_ANALYSIS / "play_type_eval_selfdetected.json"))

    preds: dict[str, dict] = {}
    for row in formation["predictions"]:
        preds.setdefault(row["clip"], {})["OFF FORM"] = row["pred_form"]
    for row in yard_hash["predictions"]:
        preds.setdefault(row["clip"], {})["YARD LN"] = round(row["pred_yardline"])
        preds.setdefault(row["clip"], {})["HASH"] = row["pred_hash"]
    for row in play_type["predictions"]:
        preds.setdefault(row["clip"], {})["PLAY TYPE"] = row["pred_play_type"]
    return preds


def select_clips(meta: pd.DataFrame, odk: str, n: int, seed: int) -> pd.DataFrame:
    pool = meta[meta["ODK"] == odk]
    n = min(n, len(pool))
    idx = pool.index.tolist()
    random.Random(seed).shuffle(idx)
    return meta.loc[idx[:n]].sort_values("play_num")


def write_sheet(rows: pd.DataFrame, preds: dict[str, dict], condition: str, out_path: Path):
    """condition: 'manual' (nothing pre-filled) or 'review' (tagged columns pre-filled)."""
    columns = HUDL_COLUMNS + EXTRA_COLUMNS
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(columns)

    for _, row in rows.iterrows():
        clip = row["clip"]
        values = {c: None for c in HUDL_COLUMNS}
        values["PLAY #"] = int(row["play_num"])
        values["ODK"] = row["ODK"]
        if condition == "review":
            for col in TAGGED_COLUMNS:
                values[col] = preds.get(clip, {}).get(col)
        ws.append([values[c] for c in HUDL_COLUMNS] + [clip])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    print(f"saved -> {out_path}  ({len(rows)} rows, condition={condition})")


def main():
    meta = pd.read_csv(META_CSV)
    preds = load_predictions()

    offense = select_clips(meta, "O", N_PER_SIDE, SEED)
    defense = select_clips(meta, "D", N_PER_SIDE, SEED + 1)

    write_sheet(offense, preds, "manual", OUT_DIR / "Coach1_Manual_Offense.xlsx")
    write_sheet(defense, preds, "review", OUT_DIR / "Coach1_Review_Defense.xlsx")
    write_sheet(defense, preds, "manual", OUT_DIR / "Coach2_Manual_Defense.xlsx")
    write_sheet(offense, preds, "review", OUT_DIR / "Coach2_Review_Offense.xlsx")


if __name__ == "__main__":
    main()
