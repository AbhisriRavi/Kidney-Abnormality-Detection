"""
HP sweep Stage A: optimizer parameters (LR x WD) on fold 0 of v2.
Outputs: $SCRATCH/kidney-results/hp_sweep/stage_a/<config_id>/summary.json
Tracks: val_f1 (early-stop best), test_f1.
"""
import os, json, subprocess, itertools
from pathlib import Path

SCRATCH = Path(os.environ["SCRATCH"])
OUT_ROOT = SCRATCH / "kidney-results/hp_sweep/stage_a"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

GRID = {
    "lr": [5e-5, 1e-4, 2e-4],
    "wd": [1e-5, 1e-4, 1e-3],
}
configs = list(itertools.product(GRID["lr"], GRID["wd"]))

for i, (lr, wd) in enumerate(configs):
    cfg_id = f"a_lr{lr:.0e}_wd{wd:.0e}".replace("+0", "").replace("-0", "-")
    print(f"[{i+1}/{len(configs)}] {cfg_id}: lr={lr}, wd={wd}")
    cmd = [
        "python", "-u", "src/training/train_kfold_v2.py",
        "--tag", f"hp_sweep_stage_a/{cfg_id}",
        "--manifest", "v2",
        "--seed", "42",
        "--lr", str(lr),
        "--weight-decay", str(wd),
        "--folds-to-run", "0",  # only fold 0
    ]
    subprocess.run(cmd, check=False)