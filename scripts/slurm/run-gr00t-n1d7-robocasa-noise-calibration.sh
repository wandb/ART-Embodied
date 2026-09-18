#!/usr/bin/env bash
#SBATCH --job-name=gr00t-n17-noise-cal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=04:00:00
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
config="${repository_root}/examples/embodied/gr00t_n1d7_robocasa_gr1_tabletop_cuttingboard_pan_single_task_development.yaml"
reference_root="${repository_root}/outputs/gr00t-n1d7-robocasa-sampler-calibration/cuttingboard-pan-k4-noise05-v1/baseline-ode/evaluation"
reference_outcomes="${reference_root}/update_000000_episode_outcomes.json"
reference_evidence="${reference_root}/update_000000_evidence.json"
preregistration="${repository_root}/examples/embodied/robocasa/development_experiments/gr1_tabletop_cuttingboard_pan_noise_calibration_v1.json"
output_dir="${repository_root}/outputs/gr00t-n1d7-robocasa-sampler-calibration/cuttingboard-pan-k4-noise-screen-v1"
ffmpeg_library_dir="${repository_root}/.runtime-libs/ffmpeg7/lib"
egl_library_dir="${repository_root}/.runtime-libs/egl"

for required in "${python}" "${config}" "${reference_outcomes}" "${reference_evidence}" "${preregistration}"; do
  [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 1; }
done
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN is required" >&2; exit 1; }
[[ -n "${WANDB_API_KEY:-}" ]] || { echo "WANDB_API_KEY is required" >&2; exit 1; }

mkdir -p "${repository_root}/outputs/slurm"
cd "${repository_root}"
export ART_EMBODIED_REPO_ROOT="${repository_root}"
export LD_LIBRARY_PATH="${ffmpeg_library_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
if [[ -d "${egl_library_dir}" ]]; then
  export LD_LIBRARY_PATH="${egl_library_dir}:${LD_LIBRARY_PATH}"
fi
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONSAFEPATH=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false PYTHONPATH="${repository_root}/src:${repository_root}"
unset WANDB_RUN_ID WANDB_RESUME

args=(
  --config "${config}"
  --reference-ode-outcomes "${reference_outcomes}"
  --reference-ode-evidence "${reference_evidence}"
  --output-dir "${output_dir}"
  --preregistration "${preregistration}"
  --noise-levels 0.1 0.2 0.3
  --seed-start 400
  --seed-count 16
  --policy-seed-repetitions 8
  --maximum-absolute-gap 0.10
)

"${python}" -P -m examples.embodied.robocasa.gr00t_n1d7_noise_calibration \
  "${args[@]}" --preflight \
  > "${repository_root}/outputs/slurm/gr00t-n17-noise-cal-${SLURM_JOB_ID}-preflight.json"
for generated_config in "${output_dir}"/configs/*.yaml; do
  "${python}" -m art_embodied.cli validate "${generated_config}" --json >/dev/null
done

exec "${repository_root}/scripts/slurm/run-in-repo.sh" \
  "${python}" -P -m examples.embodied.robocasa.gr00t_n1d7_noise_calibration \
  "${args[@]}"
