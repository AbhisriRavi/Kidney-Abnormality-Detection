"""
Statistical comparison of conditions using patient-clustered bootstrap
confidence intervals and McNemar's test.

WHY THIS REPLACES THE OLD APPROACH
----------------------------------
significance_tests.py runs paired t-tests on 5 fold-level means. With n=5 and
15 pairwise comparisons per metric, Bonferroni correction leaves essentially
no power -- which is exactly why notes_for_writing.md has to concede that the
Cyst ROI effect "did not survive Bonferroni correction".

Two better tests, both computed from the saved per-image predictions and
neither requiring any retraining:

1. CLUSTER BOOTSTRAP. Resample *patients* with replacement (not slices --
   slices within a patient are not independent), recompute the metric on each
   resample, and take percentile intervals. This uses all ~560-760 patients as
   the effective sample rather than 5 fold means, so intervals are honest and
   informative. Paired bootstrap on the difference gives a direct interval on
   Δmetric.

2. McNEMAR'S TEST. The standard test for comparing two classifiers on the
   *same* test set. It conditions on the discordant pairs (cases where exactly
   one model is right) and is far more powerful than comparing fold means. Run
   at patient level so the independence assumption holds.

Usage
-----
python -m src.evaluation.robust_stats \
    --conditions v3c_rn50_full v3c_rn50_roi \
    --level patient --n-boot 2000

python -m src.evaluation.robust_stats \
    --conditions RN50_full RN50_roi CBAM_full CBAM_roi ViT_full ViT_roi \
    --level patient --all-pairs
"""
import os
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar

from src.evaluation.patient_level_eval import aggregate_to_patient, CLASSES, PROB_COLS

RNG_SEED = 20260804


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metric_fn(name):
    if name == "accuracy":
        return lambda yt, yp: accuracy_score(yt, yp)
    if name == "macro_f1":
        return lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)
    if name.startswith("f1_"):
        cls = CLASSES.index(name[3:])
        return lambda yt, yp: f1_score(
            yt, yp, labels=[cls], average="macro", zero_division=0
        )
    raise ValueError(f"Unknown metric '{name}'")


# ---------------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------------
def cluster_bootstrap_ci(df, metric, n_boot=2000, alpha=0.05, seed=RNG_SEED):
    """
    Percentile CI for a single condition, resampling over patients.

    df must have columns: patient_id, y_true, y_pred (one row per unit).
    """
    fn = metric_fn(metric)
    patients = df["patient_id"].unique()
    by_pat = {p: g for p, g in df.groupby("patient_id", sort=False)}

    rng = np.random.default_rng(seed)
    point = fn(df["y_true"].values, df["y_pred"].values)

    boots = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(patients, size=len(patients), replace=True)
        sample = pd.concat([by_pat[p] for p in picked], ignore_index=True)
        boots[b] = fn(sample["y_true"].values, sample["y_pred"].values)

    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, lo, hi, boots


def paired_bootstrap_diff(df_a, df_b, metric, n_boot=2000, alpha=0.05, seed=RNG_SEED):
    """
    Paired cluster bootstrap on the difference A - B.

    Both frames must cover the same patients. The SAME resampled patient set is
    used for both conditions on each iteration, which is what makes the interval
    paired and much tighter than comparing two independent intervals.

    Returns (point_diff, lo, hi, p_two_sided) where the p-value is the
    bootstrap proportion of resamples crossing zero, doubled.
    """
    fn = metric_fn(metric)
    common = np.intersect1d(df_a["patient_id"].unique(), df_b["patient_id"].unique())
    if len(common) == 0:
        raise ValueError("No overlapping patients between conditions")

    a = {p: g for p, g in df_a[df_a["patient_id"].isin(common)].groupby("patient_id")}
    b = {p: g for p, g in df_b[df_b["patient_id"].isin(common)].groupby("patient_id")}

    rng = np.random.default_rng(seed)
    full_a = pd.concat(a.values(), ignore_index=True)
    full_b = pd.concat(b.values(), ignore_index=True)
    point = fn(full_a["y_true"].values, full_a["y_pred"].values) - \
        fn(full_b["y_true"].values, full_b["y_pred"].values)

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.choice(common, size=len(common), replace=True)
        sa = pd.concat([a[p] for p in picked], ignore_index=True)
        sb = pd.concat([b[p] for p in picked], ignore_index=True)
        diffs[i] = fn(sa["y_true"].values, sa["y_pred"].values) - \
            fn(sb["y_true"].values, sb["y_pred"].values)

    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return point, lo, hi, min(p, 1.0)


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------
def unit_key(df_a, df_b):
    """
    Pick a column that uniquely identifies an evaluation unit in both frames.

    At patient level, patient_id is unique. At slice level it is NOT -- merging
    on patient_id there would produce a many-to-many join and wildly inflated
    discordant counts. 'path' identifies a slice uniquely, so it is preferred
    whenever it is present and unique.
    """
    for col in ("path", "patient_id"):
        if col in df_a.columns and col in df_b.columns:
            if df_a[col].is_unique and df_b[col].is_unique:
                return col
    raise ValueError(
        "No column uniquely identifies an evaluation unit in both frames. "
        "Expected 'path' (slice level) or 'patient_id' (patient level)."
    )


def mcnemar_test(df_a, df_b):
    """
    McNemar on correct/incorrect for the two conditions over shared units.
    Uses the exact binomial test when discordant counts are small.
    """
    key = unit_key(df_a, df_b)
    merged = df_a[[key, "y_true", "y_pred"]].merge(
        df_b[[key, "y_true", "y_pred"]],
        on=key, suffixes=("_a", "_b"), validate="one_to_one"
    )
    if (merged["y_true_a"] != merged["y_true_b"]).any():
        raise ValueError("Ground-truth mismatch between conditions -- "
                         "are these the same manifest?")

    ok_a = (merged["y_pred_a"] == merged["y_true_a"]).values
    ok_b = (merged["y_pred_b"] == merged["y_true_b"]).values

    n00 = int((~ok_a & ~ok_b).sum())   # both wrong
    n01 = int((~ok_a & ok_b).sum())    # only B right
    n10 = int((ok_a & ~ok_b).sum())    # only A right
    n11 = int((ok_a & ok_b).sum())     # both right

    table = [[n11, n10], [n01, n00]]
    discordant = n01 + n10
    exact = discordant < 25
    res = mcnemar(table, exact=exact, correction=not exact)

    return {
        "n_units": len(merged),
        "both_correct": n11, "only_A_correct": n10,
        "only_B_correct": n01, "both_wrong": n00,
        "discordant": discordant,
        "test": "exact binomial" if exact else "chi2 w/ continuity correction",
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_units(stem, pred_dir, level, agg):
    df = pd.read_csv(pred_dir / f"{stem}.csv")
    if level == "slice":
        return df
    return aggregate_to_patient(df, how=agg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--level", default="patient", choices=["patient", "slice"])
    ap.add_argument("--agg", default="mean", choices=["mean", "max", "vote"])
    ap.add_argument("--metrics", nargs="+",
                    default=["accuracy", "macro_f1"] + [f"f1_{c}" for c in CLASSES])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--all-pairs", action="store_true",
                    help="Compare every pair, not just consecutive ones")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    scratch = Path(os.environ["SCRATCH"])
    pred_dir = scratch / "kidney-results/predictions"
    outdir = scratch / "kidney-results/comparison/robust_stats"
    outdir.mkdir(parents=True, exist_ok=True)

    units = {}
    for stem in args.conditions:
        path = pred_dir / f"{stem}.csv"
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        units[stem] = load_units(stem, pred_dir, args.level, args.agg)
        print(f"Loaded {stem}: {len(units[stem])} {args.level}-level units")

    if len(units) < 1:
        raise SystemExit("Nothing to analyse.")

    # ---- 1. Per-condition bootstrap CIs ----
    print(f"\n{'=' * 74}")
    print(f"CLUSTER BOOTSTRAP CIs ({args.level} level, {args.n_boot} resamples, "
          f"{100*(1-args.alpha):.0f}% percentile)")
    print("=" * 74)
    ci_rows = []
    for stem, df in units.items():
        for m in args.metrics:
            pt, lo, hi, _ = cluster_bootstrap_ci(
                df, m, n_boot=args.n_boot, alpha=args.alpha
            )
            ci_rows.append({"condition": stem, "metric": m,
                            "point": pt, "ci_low": lo, "ci_high": hi})
            print(f"  {stem:<24} {m:<12} {pt:.4f}  [{lo:.4f}, {hi:.4f}]")
    pd.DataFrame(ci_rows).to_csv(outdir / f"bootstrap_ci_{args.level}.csv", index=False)

    # ---- 2. Pairwise comparisons ----
    names = list(units)
    pairs = list(combinations(names, 2)) if args.all_pairs else \
        list(zip(names[:-1], names[1:]))

    n_tests = len(pairs) * len(args.metrics)
    bonf = args.alpha / max(n_tests, 1)

    print(f"\n{'=' * 74}")
    print(f"PAIRWISE COMPARISONS  ({len(pairs)} pairs x {len(args.metrics)} metrics "
          f"= {n_tests} tests; Bonferroni alpha = {bonf:.5f})")
    print("=" * 74)

    rows = []
    for a, b in pairs:
        mc = mcnemar_test(units[a], units[b])
        print(f"\n{a}  vs  {b}")
        print(f"  McNemar ({mc['test']}): stat={mc['statistic']:.3f}  "
              f"p={mc['p_value']:.5f}  "
              f"[only-{a} right: {mc['only_A_correct']}, "
              f"only-{b} right: {mc['only_B_correct']}]")

        for m in args.metrics:
            d, lo, hi, p = paired_bootstrap_diff(
                units[a], units[b], m, n_boot=args.n_boot, alpha=args.alpha
            )
            crosses_zero = lo <= 0 <= hi
            flag = "" if crosses_zero else "  *"
            print(f"    Δ{m:<12} {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
                  f"boot p={p:.4f}{flag}")
            rows.append({
                "condition_A": a, "condition_B": b, "metric": m,
                "delta": d, "ci_low": lo, "ci_high": hi,
                "boot_p": p,
                "ci_excludes_zero": not crosses_zero,
                "sig_bonferroni": p < bonf,
                "mcnemar_p": mc["p_value"],
                "mcnemar_sig_bonferroni": mc["p_value"] < bonf,
                "n_units": mc["n_units"],
                "only_A_correct": mc["only_A_correct"],
                "only_B_correct": mc["only_B_correct"],
            })

    out = pd.DataFrame(rows)
    out_path = outdir / f"pairwise_{args.level}.csv"
    out.to_csv(out_path, index=False)
    print(f"\n\nSaved: {out_path}")
    print("\nInterpretation note for the write-up: a CI on Δmetric that excludes "
          "zero is the strongest single statement you can make. Where the CI "
          "includes zero but is narrow, you can report a bounded null -- e.g. "
          "'any effect is smaller than 2 percentage points' -- which is a much "
          "stronger claim than 'not significant'.")


if __name__ == "__main__":
    main()
