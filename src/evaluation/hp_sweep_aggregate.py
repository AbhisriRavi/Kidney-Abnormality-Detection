"""
Aggregate HP sweep results and pick the winner (highest test macro-F1).
Usage: python hp_sweep_aggregate.py [stage_a|stage_b]
"""
import os, sys, json
from pathlib import Path
import pandas as pd

SCRATCH = Path(os.environ["SCRATCH"])
STAGE = sys.argv[1] if len(sys.argv) > 1 else "stage_a"
ROOT = SCRATCH / f"kidney-results/kfold/hp_sweep/{STAGE}"

pattern = "lr*_wd*" if STAGE == "stage_a" else "bs*_aug*"

rows = []
for cfg_dir in sorted(ROOT.glob(pattern)):
    summary = cfg_dir / "summary.json"
    if not summary.exists():
        print(f"MISSING: {cfg_dir.name}")
        continue
    with open(summary) as f:
        s = json.load(f)
    if not s["fold_results"]:
        print(f"NO RESULTS: {cfg_dir.name}")
        continue
    fold0 = s["fold_results"][0]
    if STAGE == "stage_a":
        row_extra = {"lr": s["lr"], "wd": s["weight_decay"]}
    else:
        row_extra = {"batch_size": s["batch_size"], "aug_strength": s["aug_strength"]}
    rows.append({
        "config": cfg_dir.name,
        **row_extra,
        "test_acc": fold0["test_accuracy"],
        "test_f1": fold0["test_macro_f1"],
        "test_auc": fold0["test_macro_auc"],
        "f1_Normal": fold0["per_class"]["Normal"]["f1"],
        "f1_Cyst": fold0["per_class"]["Cyst"]["f1"],
        "f1_Tumor": fold0["per_class"]["Tumor"]["f1"],
        "f1_Stone": fold0["per_class"]["Stone"]["f1"],
    })

df = pd.DataFrame(rows).sort_values("test_f1", ascending=False)
print(f"\n=== {STAGE.upper()} leaderboard (sorted by test F1) ===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if len(df) > 0:
    winner = df.iloc[0]
    print(f"\n{STAGE.upper()} WINNER: {winner['config']}")
    print(f"  Test F1 = {winner['test_f1']:.4f}")

out_csv = ROOT / f"{STAGE}_leaderboard.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")