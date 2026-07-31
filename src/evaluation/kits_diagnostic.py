"""
Diagnose the KiTS 38% accuracy.
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

SCRATCH = Path(os.environ["SCRATCH"])
PD_ROOT = SCRATCH / "kidney-results/per_dataset"
V3C_ROOT = SCRATCH / "kidney-results/kfold"

def check_fold_composition():
    print("=" * 70)
    print("1. FOLD COMPOSITION CHECK (kits/manifest_seed42.csv)")
    print("=" * 70)
    manifest = SCRATCH / "kidney-data/processed/per_dataset/kits/manifest_seed42.csv"
    df = pd.read_csv(manifest)
    for fold in range(5):
        sub = df[df["fold"] == fold]
        print(f"\nFold {fold}: {len(sub)} imgs, {sub['patient_id'].nunique()} patients")
        print(f"  Class distribution: {sub['label'].value_counts().to_dict()}")
        pt_class = sub.groupby("patient_id")["label"].first()
        print(f"  Patients per class in TEST fold: {pt_class.value_counts().to_dict()}")

def check_training_logs(tag):
    print("=" * 70)
    print(f"2. TRAINING LOG: {tag}")
    print("=" * 70)
    for fold in range(5):
        log = PD_ROOT / tag / f"fold_{fold}" / "log.csv"
        if not log.exists():
            print(f"  Fold {fold}: no log")
            continue
        df = pd.read_csv(log)
        first_epoch = df.iloc[0]
        peak_epoch = df.loc[df["val_macro_f1"].idxmax()]
        last_epoch = df.iloc[-1]
        print(f"\nFold {fold}:")
        print(f"  Ep 1:    tr_acc={first_epoch['train_acc']:.3f}  val_F1={first_epoch['val_macro_f1']:.3f}")
        print(f"  Peak:    ep {int(peak_epoch['epoch']):2d}  tr_acc={peak_epoch['train_acc']:.3f}  val_F1={peak_epoch['val_macro_f1']:.3f}")
        print(f"  Ep {int(last_epoch['epoch']):2d}: tr_acc={last_epoch['train_acc']:.3f}  val_F1={last_epoch['val_macro_f1']:.3f}")

def check_confusion_matrices(tag):
    print("=" * 70)
    print(f"3. TEST CONFUSION MATRICES: {tag}")
    print("=" * 70)
    for fold in range(5):
        m = PD_ROOT / tag / f"fold_{fold}" / "metrics.json"
        if not m.exists():
            continue
        d = json.load(open(m))
        cm = np.array(d["confusion_matrix"])
        classes = d["class_labels"]
        print(f"\nFold {fold}: acc={d['test_accuracy']:.3f}  F1={d['test_macro_f1']:.3f}")
        print(f"  Rows=true, Cols=pred, classes={classes}")
        print(f"  {cm}")
        for i, c in enumerate(classes):
            support = cm[i].sum()
            recall = cm[i, i] / max(support, 1)
            print(f"    {c}: recall={recall:.3f} ({cm[i,i]}/{support})")

def compare_to_v3c_kits_subset():
    print("=" * 70)
    print("4. COMPARE: KiTS test-accuracy in v3c multi-source vs per-dataset")
    print("=" * 70)
    v3c_tags = [f"v3c_rn50_baseline_seed{s}" for s in [42, 123, 456]]
    v3c_kits_accs = []
    for tag in v3c_tags:
        p = V3C_ROOT / tag / "summary.json"
        if not p.exists():
            continue
        s = json.load(open(p))
        for fr in s["fold_results"]:
            if "per_source" in fr and "kits" in fr["per_source"]:
                v3c_kits_accs.append(fr["per_source"]["kits"]["accuracy"])
    if v3c_kits_accs:
        print(f"\nv3c multi-source, KiTS test-images only:")
        print(f"  n={len(v3c_kits_accs)} folds")
        print(f"  Mean: {np.mean(v3c_kits_accs):.4f} +- {np.std(v3c_kits_accs):.4f}")
    else:
        print("\nNo per-source metrics in v3c baseline runs.")

    pd_kits_accs = []
    for seed in [42, 123, 456]:
        tag = f"pd_kits_resnet50_seed{seed}"
        p = PD_ROOT / tag / "summary.json"
        if not p.exists():
            continue
        s = json.load(open(p))
        for fr in s["fold_results"]:
            pd_kits_accs.append(fr["test_accuracy"])
    print(f"\nper-dataset KiTS-only, RN50:")
    print(f"  n={len(pd_kits_accs)} folds")
    print(f"  Mean: {np.mean(pd_kits_accs):.4f} +- {np.std(pd_kits_accs):.4f}")

if __name__ == "__main__":
    check_fold_composition()
    print()
    check_training_logs("pd_kits_resnet50_seed42")
    print()
    check_confusion_matrices("pd_kits_resnet50_seed42")
    print()
    compare_to_v3c_kits_subset()
