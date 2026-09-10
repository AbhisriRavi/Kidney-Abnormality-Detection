"""
Chapter 4 numbers on the corrected v3c dataset.

Run on AIRE after dump_predictions has produced all six CSVs:

    python ch4_v3c.py | tee ch4_v3c.txt

Produces:
  [1] Table 4.1 rows  - slice and patient metrics, all six conditions
  [2] Section 4.1     - slice->patient aggregation gain
  [3] Section 4.2.2   - ROI effect per class, replicated across 3 architectures
  [4] Section 4.3     - pairwise architecture comparisons, patient-matched
"""
import os, itertools
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

S = os.environ["SCRATCH"]
PRED = f"{S}/kidney-results/predictions"
CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
PRIORITY = ["Tumor", "Cyst", "Stone", "Normal"]
RNG = np.random.default_rng(0)
NBOOT = 2000

COND = {
    "rn50_full": "v3c_rn50_baseline_seed42",
    "rn50_roi":  "v3c_roi_baseline_seed42",
    "cbam_full": "v3c_cbam_full_seed42",
    "cbam_roi":  "v3c_cbam_roi_seed42",
    "vit_full":  "v3c_vit_full_seed42",
    "vit_roi":   "v3c_vit_roi_seed42",
}


def load(tag):
    d = pd.read_csv(f"{PRED}/{tag}.csv")
    if d.y_true.dtype == object:
        d["y_true"] = d.y_true.map(CLASSES.index)
        d["y_pred"] = d.y_pred.map(CLASSES.index)
    return d


def to_patient(d):
    """Max aggregation over slices; severity-priority ground truth."""
    pcols = [c for c in d.columns if c.startswith("p_")]
    rows = []
    for pid, g in d.groupby("patient_id"):
        probs = g[pcols].to_numpy()
        pred = int(np.unravel_index(probs.argmax(), probs.shape)[1])
        present = set(g.y_true)
        true = next(CLASSES.index(c) for c in PRIORITY if CLASSES.index(c) in present)
        rows.append({"patient_id": pid, "source": g.source.iloc[0],
                     "y_true": true, "y_pred": pred,
                     **{c: probs.mean(0)[i] for i, c in enumerate(pcols)}})
    return pd.DataFrame(rows)


def metrics(df):
    acc = (df.y_true == df.y_pred).mean()
    mf1 = f1_score(df.y_true, df.y_pred, average="macro", zero_division=0)
    per = f1_score(df.y_true, df.y_pred, average=None,
                   labels=range(4), zero_division=0)
    pcols = [c for c in df.columns if c.startswith("p_")]
    try:
        auc = roc_auc_score(df.y_true, df[pcols].div(df[pcols].sum(1), axis=0),
                            multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    return acc, mf1, auc, per


def boot_diff(a, b, fn):
    """Paired cluster bootstrap over patients. a, b share patient_id order."""
    pids = a.patient_id.to_numpy()
    obs = fn(a) - fn(b)
    out = np.empty(NBOOT)
    idx_by_pid = {p: i for i, p in enumerate(pids)}
    for k in range(NBOOT):
        samp = RNG.choice(pids, size=len(pids), replace=True)
        ii = [idx_by_pid[p] for p in samp]
        out[k] = fn(a.iloc[ii]) - fn(b.iloc[ii])
    lo, hi = np.percentile(out, [2.5, 97.5])
    p = 2 * min((out <= 0).mean(), (out >= 0).mean())
    return obs, lo, hi, min(p, 1.0)


def mcnemar(a, b):
    ca = (a.y_true.to_numpy() == a.y_pred.to_numpy())
    cb = (b.y_true.to_numpy() == b.y_pred.to_numpy())
    n01 = int((~ca & cb).sum()); n10 = int((ca & ~cb).sum())
    if n01 + n10 == 0:
        return n10, n01, float("nan"), 1.0
    chi2 = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    from scipy.stats import chi2 as c2, binomtest
    if n01 + n10 < 25:
        return n10, n01, float("nan"), binomtest(n10, n10 + n01, 0.5).pvalue
    return n10, n01, chi2, 1 - c2.cdf(chi2, 1)


slice_d = {k: load(v) for k, v in COND.items()}
pat_d = {k: to_patient(v) for k, v in slice_d.items()}

print("=" * 78)
print("[1] TABLE 4.1  -- slice and patient metrics, v3c seed 42")
print("=" * 78)
print(f"{'condition':<12}{'level':<9}{'n':>6}{'acc':>8}{'mF1':>8}{'AUC':>8}"
      f"{'Normal':>8}{'Cyst':>8}{'Tumor':>8}{'Stone':>8}")
for k in COND:
    for lvl, df in [("slice", slice_d[k]), ("patient", pat_d[k])]:
        a, m, u, per = metrics(df)
        print(f"{k:<12}{lvl:<9}{len(df):>6}{a:>8.4f}{m:>8.4f}{u:>8.4f}"
              f"{per[0]:>8.3f}{per[1]:>8.3f}{per[2]:>8.3f}{per[3]:>8.3f}")

print()
print("=" * 78)
print("[2] SECTION 4.1  -- aggregation gain (patient acc - slice acc)")
print("=" * 78)
for k in COND:
    sa = (slice_d[k].y_true == slice_d[k].y_pred).mean()
    pa = (pat_d[k].y_true == pat_d[k].y_pred).mean()
    print(f"{k:<12} slice={sa:.4f}  patient={pa:.4f}  gain={100*(pa-sa):+.1f} pp")

print()
print("=" * 78)
print("[3] SECTION 4.2.2  -- ROI effect per class, by architecture (matched)")
print("=" * 78)
for arch in ["rn50", "cbam", "vit"]:
    f, r = pat_d[f"{arch}_full"], pat_d[f"{arch}_roi"]
    shared = sorted(set(f.patient_id) & set(r.patient_id))
    fa = f[f.patient_id.isin(shared)].sort_values("patient_id").reset_index(drop=True)
    ra = r[r.patient_id.isin(shared)].sort_values("patient_id").reset_index(drop=True)
    print(f"\n{arch.upper()}  (matched n={len(shared)})")
    for name, fn in [("accuracy", lambda d: (d.y_true == d.y_pred).mean()),
                     ("macro_f1", lambda d: f1_score(d.y_true, d.y_pred,
                                                     average="macro", zero_division=0))]:
        o, lo, hi, p = boot_diff(fa, ra, fn)
        print(f"  {name:<10} full-roi = {o:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p={p:.4f}")
    for i, c in enumerate(CLASSES):
        fn = lambda d, i=i: f1_score(d.y_true, d.y_pred, average=None,
                                     labels=range(4), zero_division=0)[i]
        o, lo, hi, p = boot_diff(fa, ra, fn)
        flag = "SIG" if (lo > 0 or hi < 0) else "   "
        print(f"  F1 {c:<8} full-roi = {o:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p={p:.4f} {flag}")

print()
print("=" * 78)
print("[4] SECTION 4.3  -- architecture comparisons, patient-matched")
print("=" * 78)
for cond in ["full", "roi"]:
    keys = [f"{a}_{cond}" for a in ["rn50", "cbam", "vit"]]
    shared = set(pat_d[keys[0]].patient_id)
    for k in keys[1:]:
        shared &= set(pat_d[k].patient_id)
    shared = sorted(shared)
    print(f"\n--- {cond.upper()} input, matched n={len(shared)}")
    sub = {k: pat_d[k][pat_d[k].patient_id.isin(shared)]
                 .sort_values("patient_id").reset_index(drop=True) for k in keys}
    for k in keys:
        a, m, u, _ = metrics(sub[k])
        print(f"  {k:<12} acc={a:.4f}  macroF1={m:.4f}")
    for x, y in itertools.combinations(keys, 2):
        for name, fn in [("acc", lambda d: (d.y_true == d.y_pred).mean()),
                         ("mF1", lambda d: f1_score(d.y_true, d.y_pred,
                                                    average="macro", zero_division=0))]:
            o, lo, hi, p = boot_diff(sub[x], sub[y], fn)
            print(f"  {x} vs {y}  d{name} = {o:+.4f}  [{lo:+.4f}, {hi:+.4f}]  p={p:.4f}")
        n10, n01, chi2, pm = mcnemar(sub[x], sub[y])
        print(f"      McNemar: {x} only correct {n10}, {y} only correct {n01}, "
              f"chi2={chi2:.1f}, p={pm:.2e}")
