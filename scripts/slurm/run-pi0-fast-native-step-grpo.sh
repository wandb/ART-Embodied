#!/usr/bin/env bash
#SBATCH --job-name=pi0fast-native-step-grpo
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
root="${ART_EMBODIED_FP16_RESTART_ROOT:?}"
for _ in $(seq 1 90); do
  if [[ -f "$root/NATIVE_RECIPE_READY" ]]; then
    exec bash scripts/slurm/run-pi0-fast-fp16-grpo-repair.sh
  fi
  sleep 10
done
echo 'Native-step recipe did not pass its qualification within 15 minutes' >&2
exit 1
