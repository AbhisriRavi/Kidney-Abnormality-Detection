"""Section 4.5 intervention comparisons, v3c seed 42, patient-matched.

Run:  python -m src.evaluation.ch4_intervention
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

S = os.environ["SCRATCH"]
PRED = f"{S}/kidney-results/predictions"
C = ["Normal", "Cyst", "Tumor", "Stone"]
P = ["Tumor", "Cyst", "Stone", "Normal"]
RNG = np.random.default_rng(0)


def pat(tag):
    f = f"{PRED}/{tag}.csv"
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    if d.y_true.dtype == object:
        d["y_true"] = d.y_true.map(C.index)
        d["y_pred"] = d.y_pred.map(C.index)
    pc = [c for c in d.columns if c.startswith("p_")]
    rows = []
    for pid, g in d.groupby("patient_id"):
        pr = g[pc].to_numpy()
        rows.append({
            "patient_id": pid,
            "y_pred": int(np.unravel_index(pr.argmax(), pr.shape)[1]),
            "y_true": next(C.index(c) for c in P if C.index(c) in set(g.y_true)),
        })
    return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)


base = pat("v3c_rn50_baseline_seed42")
runs = {
    "Focal loss":         "v3c_focal_seed42",
    "Multi-head (both)":  "v3c_mh_both_seed42",
    "Multi-head (KiTS)":  "v3c_mh_kits_seed42",
    "Multi-head (KAUH)":  "v3c_mh_kauh_seed42",
    "CBAM":               "v3c_cbam_full_seed42",
    "DANN":               "v3c_dann_seed42",
}

print(f"baseline n={len(base)}  acc={(base.y_true == base.y_pred).mean():.4f}  "
      f"mF1={f1_score(base.y_true, base.y_pred, average='macro', zero_division=0):.4f}\n")

for name, tag in runs.items():
    o = pat(tag)
    if o is None:
        print(f"{name:<20} MISSING {tag}")
        continue
    sh = sorted(set(base.patient_id) & set(o.patient_id))
    a = base[base.patient_id.isin(sh)].reset_index(drop=True)
    b = o[o.patient_id.isin(sh)].reset_index(drop=True)
    print(f"{name}  (n={len(sh)}, intervention acc="
          f"{(b.y_true == b.y_pred).mean():.4f})")
    for label, fn in [
        ("acc", lambda d: (d.y_true == d.y_pred).mean()),
        ("mF1", lambda d: f1_score(d.y_true, d.y_pred, average="macro", zero_division=0)),
    ]:
        obs = fn(a) - fn(b)
        bs = np.empty(2000)
        for k in range(2000):
            i = RNG.integers(0, len(a), len(a))
            bs[k] = fn(a.iloc[i]) - fn(b.iloc[i])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
        print(f"    d{label} = {obs:+.4f}  [{lo:+.4f}, {hi:+.4f}]  p={min(p, 1):.4f}")
    ca = (a.y_true == a.y_pred).to_numpy()
    cb = (b.y_true == b.y_pred).to_numpy()
    print(f"    McNemar: baseline-only {int((ca & ~cb).sum())}, "
          f"intervention-only {int((~ca & cb).sum())}\n")
