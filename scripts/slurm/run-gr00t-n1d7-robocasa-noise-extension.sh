#!/usr/bin/env bash
#SBATCH --job-name=gr00t-n17-noise-ext
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
repository_root="${ART_EMBODIED_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${repository_root}" ]]; then
  repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [[ -f "${repository_root}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${repository_root}/.env"
  set +a
fi
python="${repository_root}/.venv-gr00t-n1d7/bin/python"
config="${ART_EMBODIED_EXTENSION_CONFIG:-${repository_root}/examples/embodied/gr00t_n1d7_robocasa_gr1_tabletop_cuttingboard_pan_single_task_development.yaml}"
prior_ode="${ART_EMBODIED_EXTENSION_PRIOR_ODE:-${repository_root}/outputs/gr00t-n1d7-robocasa-sampler-calibration/cuttingboard-pan-k4-noise05-v1/baseline-ode/evaluation/update_000000_episode_outcomes.json}"
prior_sde="${ART_EMBODIED_EXTENSION_PRIOR_SDE:-${repository_root}/outputs/gr00t-n1d7-robocasa-sampler-calibration/cuttingboard-pan-k4-noise-screen-v1/sde-noise-0p10/evaluation/update_000000_episode_outcomes.json}"
preregistration="${ART_EMBODIED_EXTENSION_PREREGISTRATION:-${repository_root}/examples/embodied/robocasa/development_experiments/gr1_tabletop_cuttingboard_pan_noise_extension_v1.json}"
output_dir="${ART_EMBODIED_EXTENSION_OUTPUT_DIR:-${repository_root}/outputs/gr00t-n1d7-robocasa-sampler-calibration/cuttingboard-pan-k4-noise01-extension-v1}"
noise_level="${ART_EMBODIED_EXTENSION_NOISE_LEVEL:-0.1}"
seed_start="${ART_EMBODIED_EXTENSION_SEED_START:-416}"
seed_count="${ART_EMBODIED_EXTENSION_SEED_COUNT:-16}"
policy_seed_repetitions="${ART_EMBODIED_EXTENSION_POLICY_SEED_REPETITIONS:-8}"
maximum_absolute_gap="${ART_EMBODIED_EXTENSION_MAXIMUM_ABSOLUTE_GAP:-0.10}"
wandb_group="${ART_EMBODIED_EXTENSION_WANDB_GROUP:-cuttingboard-pan-k4-noise01-extension-v1}"
ffmpeg_library_dir="${repository_root}/.runtime-libs/ffmpeg7/lib"
egl_library_dir="${repository_root}/.runtime-libs/egl"
for required in "${python}" "${config}" "${prior_ode}" "${prior_sde}" "${preregistration}"; do
  [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 1; }
done
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN is required" >&2; exit 1; }
[[ -n "${WANDB_API_KEY:-}" ]] || { echo "WANDB_API_KEY is required" >&2; exit 1; }
mkdir -p "${repository_root}/outputs/slurm"
cd "${repository_root}"
export ART_EMBODIED_REPO_ROOT="${repository_root}"
export LD_LIBRARY_PATH="${ffmpeg_library_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
if [[ -d "${egl_library_dir}" ]]; then export LD_LIBRARY_PATH="${egl_library_dir}:${LD_LIBRARY_PATH}"; fi
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONSAFEPATH=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false PYTHONPATH="${repository_root}/src:${repository_root}"
unset WANDB_RUN_ID WANDB_RESUME
args=(
  --config "${config}"
  --prior-ode-outcomes "${prior_ode}"
  --prior-sde-outcomes "${prior_sde}"
  --output-dir "${output_dir}"
  --preregistration "${preregistration}"
  --noise-level "${noise_level}"
  --seed-start "${seed_start}"
  --seed-count "${seed_count}"
  --policy-seed-repetitions "${policy_seed_repetitions}"
  --maximum-absolute-gap "${maximum_absolute_gap}"
  --wandb-group "${wandb_group}"
)
"${python}" -P -m examples.embodied.robocasa.gr00t_n1d7_noise_extension "${args[@]}" --preflight \
  > "${repository_root}/outputs/slurm/gr00t-n17-noise-ext-${SLURM_JOB_ID}-preflight.json"
for generated_config in "${output_dir}"/configs/*.yaml; do
  "${python}" -m art_embodied.cli validate "${generated_config}" --json >/dev/null
done
exec "${repository_root}/scripts/slurm/run-in-repo.sh" \
  "${python}" -P -m examples.embodied.robocasa.gr00t_n1d7_noise_extension "${args[@]}"
