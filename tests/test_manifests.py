"""Software checks for the manifest invariants this study depends on.

Verifies the properties asserted in the dissertation: patient-disjoint
folds, the documented manifest schema, and the dataset sizes reported
in Table 4.1.

Requires the built manifests on a scratch filesystem. Skips cleanly
when the data is absent, so a clone without data still collects.

    pytest tests/ -v
"""

import os
from pathlib import Path

import pandas as pd
import pytest

EXPECTED_COLUMNS = {"path", "label", "patient_id", "source", "fold"}
EXPECTED_LABELS = {"Normal", "Cyst", "Tumor", "Stone"}
N_FOLDS = 5

# Figures reported in Table 4.1.
REPORTED = {
    "full": {"images": 8924, "patients": 561},
    "roi": {"images": 5769, "patients": 463},
}

# Paths relative to $SCRATCH. Note the ROI directory is unified_v3c_roi,
# not unified_v3_corrected_roi as printed in Appendix C.
DEFAULT_MANIFESTS = {
    "full": "kidney-data/processed/unified_v3_corrected/manifest_with_masks.csv",
    "roi": "kidney-data/processed/unified_v3c_roi/manifest_with_masks.csv",
}


def _load(key, env_var):
    override = os.environ.get(env_var)
    if override:
        path = Path(override)
    else:
        scratch = os.environ.get("SCRATCH")
        if not scratch:
            pytest.skip(f"neither {env_var} nor $SCRATCH is set")
        path = Path(scratch) / DEFAULT_MANIFESTS[key]
    if not path.exists():
        pytest.skip(f"{key} manifest not found at {path}; set {env_var}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def full():
    return _load("full", "KIDNEY_MANIFEST_V3C")


@pytest.fixture(scope="module")
def roi():
    return _load("roi", "KIDNEY_MANIFEST_V3C_ROI")


@pytest.fixture(params=["full", "roi"])
def manifest(request, full, roi):
    return request.param, (full if request.param == "full" else roi)


def test_schema(manifest):
    name, df = manifest
    missing = EXPECTED_COLUMNS - set(df.columns)
    assert not missing, f"{name} manifest missing columns: {sorted(missing)}"


def test_labels_are_the_four_classes(manifest):
    name, df = manifest
    unexpected = set(df["label"].unique()) - EXPECTED_LABELS
    assert not unexpected, f"{name} has unexpected labels: {sorted(unexpected)}"


def test_folds_are_contiguous(manifest):
    name, df = manifest
    assert set(df["fold"].unique()) == set(range(N_FOLDS)), (
        f"{name} folds are {sorted(df['fold'].unique())}, expected 0-{N_FOLDS - 1}"
    )


def test_no_patient_appears_in_two_folds(manifest):
    """The leakage this project exists to demonstrate must not be present here."""
    name, df = manifest
    for i in range(N_FOLDS):
        for j in range(i + 1, N_FOLDS):
            a = set(df.loc[df["fold"] == i, "patient_id"])
            b = set(df.loc[df["fold"] == j, "patient_id"])
            shared = a & b
            assert not shared, (
                f"{name}: {len(shared)} patient(s) shared between "
                f"folds {i} and {j}, e.g. {sorted(shared)[:3]}"
            )


def test_patient_ids_are_source_prefixed(manifest):
    """Prefixing prevents collisions between independently numbered sources."""
    name, df = manifest
    sources = set(df["source"].unique())
    bad = [p for p in df["patient_id"].unique()
           if not any(str(p).lower().startswith(str(s).lower()) for s in sources)]
    assert not bad, f"{name}: unprefixed patient ids, e.g. {bad[:3]}"


# Class composition per source, corrected dataset. This is the
# collinearity the project exists to characterise: kauh and tcga
# contribute a single class each, so recognising the source is
# sufficient to predict the label for those patients. The test pins
# the structure so a future manifest change cannot alter it silently.
EXPECTED_CLASSES_PER_SOURCE = {
    "abdalla": 2,
    "kauh": 1,
    "kits": 3,
    "tcga": 1,
}


def test_source_class_composition_is_unchanged(manifest):
    name, df = manifest
    counts = df.groupby("source")["label"].nunique().to_dict()
    assert counts == EXPECTED_CLASSES_PER_SOURCE, (
        f"{name}: source composition changed from {EXPECTED_CLASSES_PER_SOURCE} "
        f"to {counts}. The collinearity analysis in Chapter 4 assumes the former."
    )


@pytest.mark.parametrize("key", ["full", "roi"])
def test_sizes_match_reported(key, full, roi):
    df = full if key == "full" else roi
    assert len(df) == REPORTED[key]["images"]
    assert df["patient_id"].nunique() == REPORTED[key]["patients"]


def test_roi_patients_are_a_subset_of_full(full, roi):
    """ROI drops patients to segmentation failure; it must never add any."""
    extra = set(roi["patient_id"]) - set(full["patient_id"])
    assert not extra, f"ROI has patients absent from full: {sorted(extra)[:3]}"
