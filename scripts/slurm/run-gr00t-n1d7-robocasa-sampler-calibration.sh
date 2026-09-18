#!/usr/bin/env bash
#SBATCH --job-name=gr00t-n17-robocasa-cal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

repository_root="${ART_EMBODIED_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${repository_root}" ]]; then
  repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
env_file="${ART_EMBODIED_ENV_FILE:-${repository_root}/.env}"
if [[ -f "${env_file}" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${env_file}"
  set +a
fi

python="${repository_root}/.venv-gr00t-n1d7/bin/python"
config="${ART_EMBODIED_ROBOCASA_CONFIG:-${repository_root}/examples/embodied/gr00t_n1d7_robocasa_gr1_tabletop_cuttingboard_pan_single_task_development.yaml}"
checkpoint="${ART_EMBODIED_CANDIDATE_CHECKPOINT:-${repository_root}/outputs/gr00t-n1d7-robocasa-single-task/cuttingboard-pan-isolation-r010-r075-r64-lr3e5-u6-v2/checkpoints/step-000005}"
preregistration="${ART_EMBODIED_CALIBRATION_PREREGISTRATION:-${repository_root}/examples/embodied/robocasa/development_experiments/gr1_tabletop_cuttingboard_pan_sampler_calibration_v1.json}"
output_dir="${ART_EMBODIED_CALIBRATION_OUTPUT_DIR:-${repository_root}/outputs/gr00t-n1d7-robocasa-sampler-calibration/cuttingboard-pan-k4-noise05-v1}"
egl_library_dir="${ART_EMBODIED_EGL_LIBRARY_DIR:-${repository_root}/.runtime-libs/egl}"
ffmpeg_library_dir="${repository_root}/.runtime-libs/ffmpeg7/lib"

[[ -x "${python}" ]] || { echo "Missing GR00T N1.7 runtime" >&2; exit 1; }
[[ -x "${repository_root}/.venv-robocasa-gr1/bin/python" ]] || {
  echo "Missing RoboCasa runtime" >&2
  exit 1
}
[[ -f "${config}" ]] || { echo "Missing base config: ${config}" >&2; exit 1; }
[[ -f "${preregistration}" ]] || {
  echo "Missing preregistration: ${preregistration}" >&2
  exit 1
}
[[ -f "${checkpoint}/art_embodied_checkpoint_complete.json" ]] || {
  echo "Candidate checkpoint is incomplete: ${checkpoint}" >&2
  exit 1
}
[[ -e "${ffmpeg_library_dir}/libavcodec.so.61" ]] || {
  echo "Missing pinned FFmpeg 7 runtime" >&2
  exit 1
}
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN is required" >&2; exit 1; }
[[ -n "${WANDB_API_KEY:-}" ]] || { echo "WANDB_API_KEY is required" >&2; exit 1; }

mkdir -p "${repository_root}/outputs/slurm"
cd "${repository_root}"
export ART_EMBODIED_REPO_ROOT="${repository_root}"
export LD_LIBRARY_PATH="${ffmpeg_library_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
if [[ -d "${egl_library_dir}" ]]; then
  [[ -e "${egl_library_dir}/libEGL.so.1" ]] || {
    echo "EGL directory lacks libEGL.so.1: ${egl_library_dir}" >&2
    exit 1
  }
  export LD_LIBRARY_PATH="${egl_library_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONSAFEPATH=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repository_root}/src:${repository_root}"
unset WANDB_RUN_ID WANDB_RESUME

args=(
  --config "${config}"
  --candidate-checkpoint "${checkpoint}"
  --output-dir "${output_dir}"
  --preregistration "${preregistration}"
  --seed-start 400
  --seed-count 16
  --policy-seed-repetitions 8
  --maximum-absolute-gap 0.10
)

"${python}" -P -m examples.embodied.robocasa.gr00t_n1d7_sampler_calibration \
  "${args[@]}" --preflight \
  > "${repository_root}/outputs/slurm/gr00t-n17-robocasa-cal-${SLURM_JOB_ID}-preflight.json"

for generated_config in "${output_dir}"/configs/*.yaml; do
  "${python}" -m art_embodied.cli validate "${generated_config}" --json >/dev/null
done

exec "${repository_root}/scripts/slurm/run-in-repo.sh" \
  "${python}" -P -m examples.embodied.robocasa.gr00t_n1d7_sampler_calibration \
  "${args[@]}"
