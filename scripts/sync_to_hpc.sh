#!/bin/bash
#
# Sync the local repo TO the HPC via SSH (uses ~/.ssh/config alias 'slurm').
#
# Usage:
#   ./scripts/sync_to_hpc.sh           # real sync
#   ./scripts/sync_to_hpc.sh --dry     # dry run, no changes
#
# Transport:
#   rsync over SSH, using the 'slurm' host alias defined in ~/.ssh/config
#   (slurm.nbi.ac.uk, key id_ed25519_nbi). Requires the NBI VPN to be
#   connected and passwordless key auth.
#
# Destination:
#   slurm:agroinfiltration_scoring/AutoCDAScorer_Models/
#
# What is pushed:
#   Repo code and config (src/, models/ training scripts + SLURM jobs,
#   analyses/, pyproject.toml, etc.) and the full data/ tree (images, cropped
#   images and the built datasets) so a run is self-contained on the HPC. The
#   first sync of data/ is large (~10 GB); rsync only transfers the delta after.
#
# What is NOT pushed:
#   - Local-only artefacts: models/00_Miscellaneous/, analyses/tmp/,
#     caches, egg-info, htmlcov, venvs, macOS/Word metadata.
#   - HPC-generated run artefacts. All of these are EXCLUDED, which also
#     protects them from --delete so a push never wipes results produced on the
#     HPC (even mid-run): array_task*/, *.out, *.err, and the search outputs
#     results/, parts/, logs/, random_search_results.csv, threshold.txt,
#     next.txt, and the .*_lock/ dirs.
#
# The sync uses --delete, so files removed from the local repo are also
# removed on the HPC (excluded paths above are protected). Run with --dry
# first when in doubt.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HPC_USER_HOST="slurm"
HPC_DEST="agroinfiltration_scoring/AutoCDAScorer_Models/"

DRY_RUN=""
if [ "${1:-}" = "--dry" ] || [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN="--dry-run"
    echo "=== DRY RUN -- no files will be changed ==="
fi

cd "$REPO_ROOT"

# Ensure the destination directory tree exists on the HPC.
ssh -o BatchMode=yes "$HPC_USER_HOST" "mkdir -p ${HPC_DEST}"

rsync -av --delete $DRY_RUN -e ssh \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='.pytest_cache/' \
    --exclude='.ruff_cache/' \
    --exclude='.mypy_cache/' \
    --exclude='.DS_Store' \
    --exclude='*.pyc' \
    --exclude='*.egg-info/' \
    --exclude='htmlcov/' \
    --exclude='.coverage' \
    --exclude='.venv/' \
    --exclude='venv/' \
    --exclude='~$*' \
    --exclude='models/00_Miscellaneous/' \
    --exclude='analyses/tmp/' \
    --exclude='array_task*/' \
    --exclude='*.out' \
    --exclude='*.err' \
    --exclude='results/' \
    --exclude='parts/' \
    --exclude='logs/' \
    --exclude='random_search_results.csv' \
    --exclude='threshold.txt' \
    --exclude='next.txt' \
    --exclude='.topn_lock/' \
    --exclude='.csv_lock/' \
    --exclude='.counter_lock/' \
    --exclude='_freeze/' \
    --exclude='.quarto/' \
    --exclude='*_files/' \
    "$REPO_ROOT/" "${HPC_USER_HOST}:${HPC_DEST}"

echo
echo "Sync to HPC complete."
if [ -n "$DRY_RUN" ]; then
    echo "(dry-run -- nothing was actually pushed)"
fi
