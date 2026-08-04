"""
Full TCGA-KIRC download — one good series per patient, post-contrast preferred.

Output:
  $SCRATCH/kidney-data/raw/tcga_kirc/full/
    dicom/<PatientID>/<SeriesInstanceUID>/*.dcm
    chosen_series.csv     <- manifest of what we kept
    download_log.csv      <- per-series status

This is the big download (~17 GB, ~3-6 hours via TCIA API). Run as Slurm job.
"""
import os
import json
from pathlib import Path
from tcia_utils import nbia
import pandas as pd

SCRATCH = Path(os.environ["SCRATCH"])
ROOT = SCRATCH / "kidney-data/raw/tcga_kirc"
SMOKE = ROOT / "smoke_test"
OUT = ROOT / "full"
DICOM_OUT = OUT / "dicom"
DICOM_OUT.mkdir(parents=True, exist_ok=True)

# Reuse the cached series list from smoke test, or re-fetch
SERIES_CSV = SMOKE / "all_ct_series.csv"
if not SERIES_CSV.exists():
    raise SystemExit(f"Series list not found: {SERIES_CSV}. Run smoke_test.py first.")

# ---------------------------------------------------------------------------
# Pick one series per patient
# ---------------------------------------------------------------------------
df = pd.read_csv(SERIES_CSV)
df["desc_upper"] = df["SeriesDescription"].fillna("").str.upper()
df["bpe_upper"] = df["BodyPartExamined"].fillna("").str.upper()

good_bp = df["bpe_upper"].isin(["KIDNEY", "ABDOMEN", "ABDOMENPELVIS", "ABDOMENPELVI", "ABD"])
good_size = df["ImageCount"].between(30, 400)
not_scout = ~df["desc_upper"].str.contains("SCOUT|LOC|TOPOGRAM|SURVIEW", na=False)
candidates = df[good_bp & good_size & not_scout].copy()

def categorise(d):
    d = d.upper()
    if any(s in d for s in ["PRE-CONTRAST", "PRE CONTRAST", "PRE LIVER", "NON CONTRAST", "NONCONTRAST"]):
        return "pre"
    if any(s in d for s in ["DELAY", "EXCRETORY"]):
        return "delayed"
    if any(s in d for s in ["POST CONTRAST", "POST-CONTRAST", "ARTERIAL", "PORTAL",
                             "NEPHROGRAPHIC", "CORTICOMEDULLARY", "RENAL", "C/A/P", "A/P", "CAP "]):
        return "post"
    return "unknown"

candidates["phase"] = candidates["desc_upper"].apply(categorise)
PHASE_PRIORITY = {"post": 0, "delayed": 1, "unknown": 2, "pre": 3}
candidates["phase_rank"] = candidates["phase"].map(PHASE_PRIORITY)

# One series per patient: prefer post > delayed > unknown > pre, then by ImageCount desc
chosen = (candidates.sort_values(["phase_rank", "ImageCount"], ascending=[True, False])
                     .drop_duplicates("PatientID")
                     .reset_index(drop=True))
chosen.to_csv(OUT / "chosen_series.csv", index=False)
print(f"Will download {len(chosen)} series for {chosen['PatientID'].nunique()} patients")
print(f"Total slices: {chosen['ImageCount'].sum():,}")
print(f"Estimated size: {chosen['FileSize'].sum() / 1e9:.1f} GB")
print(f"Phase distribution: {chosen['phase'].value_counts().to_dict()}\n")

# ---------------------------------------------------------------------------
# Download — one patient at a time so we can resume if interrupted
# ---------------------------------------------------------------------------
log = []
total = len(chosen)
for idx, row in chosen.iterrows():
    pid = row["PatientID"]
    suid = row["SeriesInstanceUID"]
    patient_dir = DICOM_OUT / pid
    series_dir = patient_dir / suid

    # Skip if already downloaded (resumability)
    if series_dir.exists() and any(series_dir.rglob("*.dcm")):
        existing = len(list(series_dir.rglob("*.dcm")))
        print(f"[{idx+1}/{total}] {pid}: already have {existing} DICOMs, skipping")
        log.append({"patient_id": pid, "status": "skipped", "n_dcm": existing})
        continue

    patient_dir.mkdir(parents=True, exist_ok=True)
    desc = str(row['SeriesDescription'])[:30] if pd.notna(row['SeriesDescription']) else "(no desc)"
    print(f"[{idx+1}/{total}] {pid} ({desc}, {row['ImageCount']} imgs)...", flush=True)
    try:
        nbia.downloadSeries(
            series_data=[{"SeriesInstanceUID": suid}],
            path=str(patient_dir),
            csv_filename=f"{pid}_meta.csv",
        )
        # Move the just-downloaded files into a stable per-series subfolder
        dcms = sorted(patient_dir.rglob("*.dcm"))
        # find the dir TCIA used (it'll be a UID-named folder)
        tcia_dirs = [d for d in patient_dir.iterdir() if d.is_dir() and d.name != suid]
        if tcia_dirs:
            tcia_dirs[0].rename(series_dir)
        dcms = list(series_dir.rglob("*.dcm"))
        print(f"   ✓ downloaded {len(dcms)} DICOMs")
        log.append({"patient_id": pid, "status": "ok", "n_dcm": len(dcms)})
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        log.append({"patient_id": pid, "status": "failed", "n_dcm": 0, "error": str(e)})

# Save the log
pd.DataFrame(log).to_csv(OUT / "download_log.csv", index=False)
n_ok = sum(1 for r in log if r["status"] in ("ok", "skipped"))
print(f"\nDone. {n_ok}/{total} series available locally.")