"""
Aggregate HP sweep results and pick the winner.

METHODOLOGICAL FIX (v2 of this script)
--------------------------------------
The original version ranked configurations by fold-0 *test* macro-F1. Because
fold 0 is subsequently included in the reported 5-fold cross-validation, this
leaks test information into the model-selection procedure and biases the
headline results upward.

This version ranks by `best_val_macro_f1`, which is the model-selection metric
already computed on a patient-disjoint validation split carved out of the
training folds. Test metrics are still printed, but only as a post-hoc
diagnostic, and are explicitly labelled as such.

The script also reports the rank correlation between validation and test
rankings. If the two orderings agree closely, you can state in the dissertation
that the original (leaky) selection would have chosen the same configuration,
which neutralises the criticism without retraining anything.

Usage: python -m src.evaluation.hp_sweep_aggregate [stage_a|stage_b]
"""
import os
import sys
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr, kendalltau

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
        # ---- SELECTION METRIC (validation only) ----
        "val_f1": fold0["best_val_macro_f1"],
        "best_epoch": fold0["best_epoch"],
        # ---- DIAGNOSTIC ONLY: never used for selection ----
        "test_acc__diag": fold0["test_accuracy"],
        "test_f1__diag": fold0["test_macro_f1"],
        "test_auc__diag": fold0["test_macro_auc"],
        "f1_Normal__diag": fold0["per_class"]["Normal"]["f1"],
        "f1_Cyst__diag": fold0["per_class"]["Cyst"]["f1"],
        "f1_Tumor__diag": fold0["per_class"]["Tumor"]["f1"],
        "f1_Stone__diag": fold0["per_class"]["Stone"]["f1"],
    })

if not rows:
    raise SystemExit(f"No sweep results found under {ROOT}")

df = pd.DataFrame(rows).sort_values("val_f1", ascending=False).reset_index(drop=True)
df.insert(0, "rank_val", range(1, len(df) + 1))

# Rank the same configs by the old (leaky) criterion, for comparison only.
df["rank_test__diag"] = df["test_f1__diag"].rank(ascending=False).astype(int)

print(f"\n=== {STAGE.upper()} leaderboard — ranked by VALIDATION macro-F1 ===")
print("(columns suffixed __diag are post-hoc diagnostics, not selection criteria)\n")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

winner = df.iloc[0]
print(f"\n{STAGE.upper()} WINNER (selected on validation): {winner['config']}")
print(f"  Validation macro-F1 = {winner['val_f1']:.4f}")
print(f"  [diagnostic] its fold-0 test macro-F1 = {winner['test_f1__diag']:.4f}")

# ---------------------------------------------------------------------------
# Agreement between the correct and the previously-used (leaky) ranking.
# ---------------------------------------------------------------------------
if len(df) > 2:
    rho, p_rho = spearmanr(df["val_f1"], df["test_f1__diag"])
    tau, p_tau = kendalltau(df["rank_val"], df["rank_test__diag"])
    old_winner = df.loc[df["rank_test__diag"] == 1, "config"].iloc[0]

    print("\n--- Selection-criterion agreement ---")
    print(f"Spearman rho (val_f1 vs test_f1) = {rho:.3f}  (p = {p_rho:.4f})")
    print(f"Kendall tau  (rank agreement)    = {tau:.3f}  (p = {p_tau:.4f})")
    print(f"Config that the old test-ranked criterion would have chosen: {old_winner}")
    if old_winner == winner["config"]:
        print("=> SAME configuration. State this in the dissertation: the corrected, "
              "leakage-free selection procedure reproduces the original choice, so "
              "no results need revising.")
    else:
        print("=> DIFFERENT configuration. Report the validation-selected config as "
              "the primary result. If you cannot retrain, disclose the discrepancy "
              "explicitly in the limitations section.")

out_csv = ROOT / f"{STAGE}_leaderboard_val_selected.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")
