#!/usr/bin/env bash
#SBATCH --job-name=register-gr00t-n17-model
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
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
config="${ART_EMBODIED_ROBOCASA_CONFIG:-${repository_root}/examples/embodied/gr00t_n1d7_robocasa_gr1_tabletop_sft_replay_c200_development.yaml}"
artifact_name="${ART_EMBODIED_INPUT_MODEL_ARTIFACT_NAME:-gr00t-n1d7-robocasa-gr1-tabletop-sft-u60000}"
report="${ART_EMBODIED_INPUT_MODEL_ARTIFACT_REPORT:-${repository_root}/outputs/conformance/gr00t-n17-robocasa-sft-input-model-${SLURM_JOB_ID}.json}"

[[ -x "${python}" ]] || { echo "Missing runtime: ${python}" >&2; exit 1; }
[[ -f "${config}" ]] || { echo "Missing config: ${config}" >&2; exit 1; }
[[ -n "${WANDB_API_KEY:-}" ]] || { echo "WANDB_API_KEY is required" >&2; exit 1; }

mkdir -p "${repository_root}/outputs/slurm" "$(dirname -- "${report}")"
cd "${repository_root}"
export PYTHONSAFEPATH=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${repository_root}/src:${repository_root}"
unset WANDB_RUN_ID WANDB_RESUME

exec "${python}" -P -m examples.embodied.register_wandb_input_model \
  --config "${config}" \
  --artifact-name "${artifact_name}" \
  --report "${report}"
