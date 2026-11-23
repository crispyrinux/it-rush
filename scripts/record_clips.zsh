#!/usr/bin/env zsh
set -euo pipefail

REPO_DIR="/home/dika/projects/bisindo-main"
VENV_PYTHON="/home/dika/projects/bisindo/.venv/bin/python"
OUT_DIR="${REPO_DIR}/data/training_word_videos"

mkdir -p "${OUT_DIR}"/{alam,ancaman,bencana,dampak,degradasi,iklim,lingkungan,mitigasi,pemulihan,perubahan,sumber_daya}

echo "============================================================"
echo "  SIGNIFY — Perekam Dataset Aman (Folder Permanen)"
echo "  Output: ${OUT_DIR}"
echo "============================================================"

cd "${REPO_DIR}"
PYTHONPATH="${REPO_DIR}/src" "${VENV_PYTHON}" scripts/record_word_clips.py \
  --out-dir "${OUT_DIR}" \
  "$@"

echo ""
echo "=== Ringkasan Jumlah Klip Tersimpan (${OUT_DIR}) ==="
total=0
for d in "${OUT_DIR}"/*/; do
  count=$(find "${d}" -maxdepth 1 -type f -name "*.mp4" | wc -l | tr -d ' ')
  total=$((total + count))
  printf "  %-14s : %s klip\n" "$(basename "${d}")" "${count}"
done
echo "  ----------------------------"
printf "  %-14s : %s klip\n" "TOTAL" "${total}"
