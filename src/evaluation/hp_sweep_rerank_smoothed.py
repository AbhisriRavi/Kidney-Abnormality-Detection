"""
Re-rank a hyperparameter sweep using smoothed validation curves.

WHY
---
`best_val_macro_f1` is the maximum of ~25 noisy per-epoch estimates computed on
a small patient-grouped validation split. Maximising over many noisy draws
inflates the score of whichever configuration got lucky on a single epoch,
which is why several configurations in this project's stage-B sweep report
`best_epoch = 1`. The resulting ranking is close to arbitrary: validation and
test rankings in stage B correlate at Spearman rho = 0.17 (p = 0.67).

Two lower-variance selection statistics are computed here instead, both read
straight from the existing per-epoch log.csv files:

  final_k   mean validation macro-F1 over the last k epochs. Measures where
            training converges rather than where it spiked.
  best_ma   maximum of a centred moving average of width w. Still picks a peak,
            but a peak that had to persist across neighbouring epochs.

If the three criteria (best-epoch, final_k, best_ma) agree on a winner, the
selection is robust and can be reported without qualification. If they
disagree, the honest finding is that the sweep lacked the power to separate
those configurations, which is itself worth one sentence in the methodology.

Usage
-----
python -m src.evaluation.hp_sweep_rerank_smoothed stage_a
python -m src.evaluation.hp_sweep_rerank_smoothed stage_b --last-k 5 --ma-window 3
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def moving_average(x, w):
    if w <= 1 or len(x) < w:
        return np.asarray(x, dtype=float)
    return np.convolve(np.asarray(x, dtype=float), np.ones(w) / w, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="stage_a",
                    choices=["stage_a", "stage_b"])
    ap.add_argument("--last-k", type=int, default=5,
                    help="Average validation macro-F1 over the final k epochs")
    ap.add_argument("--ma-window", type=int, default=3,
                    help="Width of the moving average for best_ma")
    args = ap.parse_args()

    scratch = Path(os.environ["SCRATCH"])
    root = scratch / f"kidney-results/kfold/hp_sweep/{args.stage}"
    pattern = "lr*_wd*" if args.stage == "stage_a" else "bs*_aug*"

    rows = []
    for cfg_dir in sorted(root.glob(pattern)):
        summary_p = cfg_dir / "summary.json"
        log_p = cfg_dir / "fold_0" / "log.csv"
        if not summary_p.exists() or not log_p.exists():
            print(f"SKIP {cfg_dir.name}: missing summary.json or fold_0/log.csv")
            continue

        with open(summary_p) as f:
            s = json.load(f)
        fold0 = s["fold_results"][0]
        log = pd.read_csv(log_p)

        if "val_macro_f1" not in log.columns:
            print(f"SKIP {cfg_dir.name}: no val_macro_f1 column in log.csv")
            continue
        curve = log["val_macro_f1"].values

        extra = ({"lr": s["lr"], "wd": s["weight_decay"]} if args.stage == "stage_a"
                 else {"batch_size": s["batch_size"], "aug_strength": s["aug_strength"]})

        rows.append({
            "config": cfg_dir.name,
            **extra,
            "n_epochs": len(curve),
            "val_best": float(curve.max()),
            "best_epoch": int(curve.argmax()) + 1,
            f"val_final_{args.last_k}": float(curve[-args.last_k:].mean()),
            f"val_best_ma{args.ma_window}": float(
                moving_average(curve, args.ma_window).max()),
            "val_std": float(curve.std()),
            "test_f1__diag": fold0["test_macro_f1"],
        })

    if not rows:
        raise SystemExit(f"No usable sweep results under {root}")

    df = pd.DataFrame(rows)
    crit_best = "val_best"
    crit_final = f"val_final_{args.last_k}"
    crit_ma = f"val_best_ma{args.ma_window}"

    for c in (crit_best, crit_final, crit_ma, "test_f1__diag"):
        df[f"rank_{c}"] = df[c].rank(ascending=False).astype(int)

    df = df.sort_values(crit_final, ascending=False).reset_index(drop=True)

    print(f"\n=== {args.stage.upper()} re-ranked on smoothed validation "
          f"(primary criterion: {crit_final}) ===\n")
    show = ["config"] + [k for k in ("lr", "wd", "batch_size", "aug_strength")
                         if k in df.columns] + \
           [crit_final, crit_ma, crit_best, "best_epoch", "val_std",
            "test_f1__diag"]
    print(df[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    winners = {
        "best-epoch validation": df.loc[df[f"rank_{crit_best}"] == 1, "config"].iloc[0],
        f"mean of final {args.last_k} epochs": df.loc[df[f"rank_{crit_final}"] == 1, "config"].iloc[0],
        f"best {args.ma_window}-epoch moving average": df.loc[df[f"rank_{crit_ma}"] == 1, "config"].iloc[0],
        "test F1 (diagnostic only)": df.loc[df["rank_test_f1__diag"] == 1, "config"].iloc[0],
    }

    print("\n--- Winner under each criterion ---")
    for k, v in winners.items():
        print(f"  {k:<38} {v}")

    val_winners = {v for k, v in winners.items() if "diagnostic" not in k}
    print()
    if len(val_winners) == 1:
        w = val_winners.pop()
        print(f"=> All three validation criteria agree on {w}. "
              f"Report this as the selected configuration; the choice is robust "
              f"to how the validation curve is summarised.")
    else:
        print(f"=> Validation criteria DISAGREE ({', '.join(sorted(val_winners))}). "
              f"This sweep lacked the power to separate the top configurations. "
              f"Report the criterion you pre-specified, note the disagreement, "
              f"and treat the difference between these configurations as within "
              f"noise rather than as a meaningful ranking.")

    # How noisy is the validation signal relative to the spread between configs?
    between = df[crit_final].std()
    within = df["val_std"].mean()
    print(f"\nSpread across configurations (SD of {crit_final}): {between:.4f}")
    print(f"Mean epoch-to-epoch SD within a configuration:      {within:.4f}")
    if within > between:
        print("=> Within-run epoch noise EXCEEDS the between-configuration spread. "
              "The sweep cannot reliably rank these configurations; say so "
              "explicitly rather than presenting the ordering as meaningful.")

    rho, p = spearmanr(df[crit_final], df["test_f1__diag"])
    print(f"\nSpearman rho ({crit_final} vs test F1) = {rho:.3f} (p = {p:.4f})")

    out = root / f"{args.stage}_leaderboard_smoothed.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
