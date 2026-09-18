#!/usr/bin/env bash
# Read-only diagnostic, not a training recipe or a benchmark repair.
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  cat <<'HELP'
Usage: bash scripts/reproduce-libero-reset.sh LEFT_PYTHON RIGHT_PYTHON OUTPUT_DIR

Compare all 50 official LIBERO Spatial task-5 states in two existing,
isolated simulator environments. Supply absolute Python executable paths.
Use matching dependencies/assets except for MuJoCo (e.g. 3.3.0 vs 3.8.1).
No installs, policy loading, rendering, state repair, or training are performed.
Both arms use seed 0, 10 wait steps and relative control. A completed comparison
is diagnostic evidence, NOT a passed training gate. See
docs/experimental/libero-reset-health.md for other recipes and reset settings.
HELP
  exit 0
fi
[[ $# == 3 ]] || { echo 'Expected LEFT_PYTHON RIGHT_PYTHON OUTPUT_DIR; use --help.' >&2; exit 2; }
left=$1 right=$2 output=$3
for python in "$left" "$right"; do
  [[ "$python" == /* && -x "$python" ]] || {
    echo "Expected an absolute, executable Python path: $python" >&2; exit 2;
  }
done
[[ ! -e "$output" ]] || { echo "Refusing existing output: $output" >&2; exit 2; }
mkdir -p "$output"
output=$(cd "$output" && pwd)
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
# Do not inherit a diagnostic package override from the caller's PYTHONPATH.
export PYTHONPATH="$repo/src:$repo" CUDA_VISIBLE_DEVICES=''
for arm in left right; do
  python=$left
  [[ "$arm" == left ]] || python=$right
  "$python" -m examples.embodied.libero.audit_reset \
    --suites libero_spatial --task-ids 5 --all-states \
    --seed 0 --wait-steps 10 --control-mode relative --report-only \
    --output "$output/$arm"
done
"$left" -m examples.embodied.libero.compare_reset_audits \
  --left "$output/left/report.json" --right "$output/right/report.json" \
  --output "$output/comparison.json"
echo "Diagnostic comparison: $output/comparison.json"
echo 'Review each report for reset failures; completion does not qualify training.'
