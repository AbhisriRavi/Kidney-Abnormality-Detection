#!/bin/bash
# ---------------------------------------------------------------------------
# One-off repository hygiene fix.
#
# 1. Force-adds src/data/, which was silently excluded by the old `data/`
#    gitignore pattern. This is the single most important fix in the repo:
#    the entire preprocessing pipeline (a named deliverable in the project
#    specification) was never committed.
# 2. Moves ~200 loose TCGA per-series metadata CSVs out of the repo root.
# 3. Renames the stray duplicate-download filename.
#
# Run from the repository root. Review `git status` before committing.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "==> 1. Checking which src/data files are currently ignored"
if [ -d src/data ]; then
    git check-ignore -v src/data/* 2>/dev/null || echo "    (none ignored -- gitignore fix already applied)"
else
    echo "    ERROR: src/data/ does not exist on disk."
    echo "    Recover it from your local working copy or the HPC checkout at"
    echo "    ~/projects/Kidney-Abnormality-Detection/src/data before continuing."
    exit 1
fi

echo "==> 2. Force-adding the preprocessing package"
git add -f src/data/
git add -f src/data/tcga/ 2>/dev/null || true

echo "==> 3. Relocating TCGA metadata CSVs"
mkdir -p metadata/tcga
for f in TCGA-*_meta.csv.csv; do
    [ -e "$f" ] || continue
    # strip the doubled .csv.csv extension while moving
    base="${f%.csv.csv}"
    git mv "$f" "metadata/tcga/${base}.csv" 2>/dev/null || mv "$f" "metadata/tcga/${base}.csv"
done
for f in downloaded.csv.csv downloaded_series.csv.csv; do
    [ -e "$f" ] || continue
    base="${f%.csv.csv}"
    git mv "$f" "metadata/tcga/${base}.csv" 2>/dev/null || mv "$f" "metadata/tcga/${base}.csv"
done

echo "==> 4. Renaming stray duplicate-download file"
if [ -e "src/evaluation/v3_results_section(1).md" ]; then
    git mv "src/evaluation/v3_results_section(1).md" \
           "dissertation/v3_results_section.md" 2>/dev/null \
    || mv "src/evaluation/v3_results_section(1).md" \
          "dissertation/v3_results_section.md"
fi

echo "==> 5. Adding package markers"
touch src/data/__init__.py
mkdir -p src/data/tcga && touch src/data/tcga/__init__.py
git add src/data/__init__.py src/data/tcga/__init__.py 2>/dev/null || true

echo
echo "==> Done. Verify with:"
echo "      git status"
echo "      git ls-files src/data/"
echo "    You should see all ten preprocessing modules listed."
