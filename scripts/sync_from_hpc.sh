#!/bin/bash
#
# Sync training results back FROM the HPC to the Mac (reverse of
# sync_to_hpc.sh). Use after SLURM jobs finish to pull the generated
# model outputs down for inspection / committing.
#
# Usage:
#   ./scripts/sync_from_hpc.sh                          # pull all model results
#   ./scripts/sync_from_hpc.sh --dry                    # dry-run
#   ./scripts/sync_from_hpc.sh --path models/01_Base_CNN_RandomSearch
#   ./scripts/sync_from_hpc.sh --path models/... --dry
#
# Transport:
#   rsync over SSH using the 'slurm' host alias in ~/.ssh/config. Requires
#   the NBI VPN to be connected.
#
# Source:
#   slurm:agroinfiltration_scoring/AutoCDAScorer_Models/<path>/
#
# Scope:
#   Pulls only HPC-GENERATED result artefacts, never source:
#     - array_task*/ directories (saved models, plots, per-run CSVs)
#     - results_*.csv and combined_sorted_results.csv
#   Source scripts (*.py, *.slurm.sh, *.qmd) and data/ are NOT pulled --
#   the Mac copy of those is authoritative.
#
# This pull is additive (no --delete): it will not remove local files.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HPC_USER_HOST="slurm"
HPC_BASE="agroinfiltration_scoring/AutoCDAScorer_Models"

DRY_RUN=""
REL="models"

usage() {
    cat <<EOF
Usage: $(basename "$0") [--path REL] [--dry]

  --path REL   Repo-relative directory to pull results from
               (default: models). Must be a relative path.
  --dry        rsync dry-run; show what would change without copying.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry|--dry-run) DRY_RUN="--dry-run"; shift ;;
        --path)          REL="${2:-}"; shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        *)               echo "ERROR: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

# Reject absolute paths / parent-dir escapes.
case "$REL" in
    /*|*..*) echo "ERROR: --path must be a relative in-repo path: $REL" >&2; exit 2 ;;
esac
REL="${REL%/}"

SRC="${HPC_USER_HOST}:${HPC_BASE}/${REL}/"
DST="${REPO_ROOT}/${REL}/"

if [ -n "$DRY_RUN" ]; then
    echo "=== DRY RUN -- no files will be changed ==="
fi

# Bail out with a friendly message if the remote path does not exist.
if ! ssh -q -o BatchMode=yes "$HPC_USER_HOST" "[ -d ${HPC_BASE}/${REL} ]" 2>/dev/null; then
    echo "ERROR: no remote directory at ${HPC_BASE}/${REL}" >&2
    exit 1
fi

mkdir -p "$DST"
echo "Pulling results: ${SRC} -> ${DST}"

# Include only result artefacts (recurse into every dir, keep the files the search
# harness actually writes), exclude everything else.
#
# These patterns must track src/random_search.py and src/results_store.py. The harness
# writes: results/<avg_vaf>_<model_id>/{model.keras,*.png}, parts/<run_id>.csv,
# random_search_results.csv, threshold.txt, next.txt, and (run_single_test.py)
# testrun/<config>/bench.json, memtest/<phase>/memory_samples.json. The array_task*/
# and results_*.csv patterns below are
# kept only for the older from_HPC-era layout; they match nothing in the current one.
rsync -av --prune-empty-dirs $DRY_RUN -e ssh \
    --include='*/' \
    --include='*.keras' \
    --include='*.png' \
    --include='random_search_results.csv' \
    --include='threshold.txt' \
    --include='next.txt' \
    --include='bench.json' \
    --include='memory_samples.json' \
    --include='features.npz' \
    --include='bench_result_*.csv' \
    --include='parts/*.csv' \
    --include='logs/**' \
    --include='*.out' \
    --include='*.err' \
    --include='array_task*/**' \
    --include='results_*.csv' \
    --include='combined_sorted_results.csv' \
    --exclude='.DS_Store' \
    --exclude='*' \
    "$SRC" "$DST"

echo
echo "Sync from HPC complete."
if [ -n "$DRY_RUN" ]; then
    echo "(dry-run -- nothing was actually pulled)"
else
    echo "Next: review with 'git status ${REL}/', then commit if desired."
fi
