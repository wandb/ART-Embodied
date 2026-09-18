#!/usr/bin/env bash
#SBATCH --job-name=pi0fast-native-sft
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
set -a
source "${ART_EMBODIED_ENV_FILE:-../ART-Embodied/.env}"
set +a
unset WANDB_RUN_ID WANDB_RESUME ART_EMBODIED_SFT_SOURCE_ROOT
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
root="${ART_EMBODIED_NATIVE_RESTART_ROOT:?}"
[[ -f "$root/plan.json" && ! -e "$root/sft" ]] || exit 1
sha256sum --status -c "$root/source.sha256"
"$python" -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m examples.embodied.pi0_fast_long_shared_sft --root "$root" --steps 400 > "$root/sft.log" 2>&1
sha256sum --status -c "$root/source.sha256"
# Do not interpret a throughput-only probe as a qualification to launch GRPO.
printf 'Fresh SFT completed. GRPO requires separate native-likelihood qualification.\n'
