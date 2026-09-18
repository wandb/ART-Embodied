#!/usr/bin/env bash
#SBATCH --job-name=pi0fast-fp16-grpo-repair
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
set -a
source "${ART_EMBODIED_ENV_FILE:-../ART-Embodied/.env}"
set +a
unset WANDB_RUN_ID WANDB_RESUME
export ART_EMBODIED_STORAGE_ROOT="${ART_EMBODIED_STORAGE_ROOT:?ART_EMBODIED_STORAGE_ROOT is required}"
export HF_HOME="$ART_EMBODIED_STORAGE_ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_DIR="$ART_EMBODIED_STORAGE_ROOT/wandb"
export WANDB_CACHE_DIR="$ART_EMBODIED_STORAGE_ROOT/cache/wandb"
export TMPDIR="$ART_EMBODIED_STORAGE_ROOT/tmp"
export PYTHONPATH="$PWD/src:$PWD"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH="${ART_EMBODIED_EGL_LIBRARY_DIR:-$PWD/../ART-Embodied/.runtime/lib}:${LD_LIBRARY_PATH:-}"
[[ "${SLURM_JOB_NUM_NODES:?}" == 1 ]] || exit 1
IFS=',' read -r -a devices <<< "${CUDA_VISIBLE_DEVICES:?}"
[[ "${#devices[@]}" == 8 ]] || exit 1
python="$ART_EMBODIED_STORAGE_ROOT/environments/pi0-fast-libero-mujoco330/bin/python"
root="${ART_EMBODIED_FP16_RESTART_ROOT:?}"
[[ -f "$root/sft/FIRST_UPDATE_VERIFIED.json" && -f "$root/sft/complete.json" ]]
"$python" - "$root/finite-localization/eos-conformance.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
assert r['finite'] and r['seeds'] >= 5
assert r['abs_mean'] <= .02 and abs(r['ratio_mean'] - 1) <= .02
PY
[[ ! -f "$root/REPAIR_READY" ]]
for attempt in 1 2 3; do
  sha256sum --status -c "$root/source.sha256"
  if [[ -f "$root/STAGE_FAILED" ]]; then
    mv "$root/STAGE_FAILED" "$root/STAGE_FAILED-before-${SLURM_JOB_ID}-$attempt"
  fi
  set +e
  "$python" -u -m examples.embodied.pi0_fast_long_restart --root "$root" --phase grpo \
    > "$root/grpo-${SLURM_JOB_ID}-$attempt.log" 2>&1
  status=$?
  set -e
  if [[ "$status" == 0 ]]; then
    sha256sum --status -c "$root/source.sha256"
    date -u +%FT%TZ > "$root/COMPLETE"
    exit 0
  fi
  printf '%s stage=grpo attempt=%s exit=%s\n' "$(date -u +%FT%TZ)" "$attempt" "$status" > "$root/STAGE_FAILED"
  # No automatic blind retry. The agent must repair, validate and publish a
  # new source manifest plus this marker. Bound idle retention to 15 minutes.
  ready=false
  for _ in $(seq 1 90); do
    if [[ -f "$root/REPAIR_READY" ]]; then
      sha256sum --status -c "$root/source.sha256"
      mv "$root/REPAIR_READY" "$root/REPAIR_READY-${SLURM_JOB_ID}-$attempt"
      ready=true
      break
    fi
    sleep 10
  done
  [[ "$ready" == true ]] || exit "$status"
done
exit 1
