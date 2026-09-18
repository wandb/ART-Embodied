#!/usr/bin/env bash
#SBATCH --job-name=robocasa-gr1-conformance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
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

config="${ART_EMBODIED_ROBOCASA_CONFIG:-${repository_root}/examples/embodied/gr00t_n1d7_robocasa_gr1_tabletop_flow_sde_grpo_gate.yaml}"
output="${ART_EMBODIED_ROBOCASA_CONFORMANCE_OUTPUT:-${repository_root}/outputs/conformance/robocasa-gr1-${SLURM_JOB_ID}.json}"
python="${ART_EMBODIED_ROBOCASA_ENVIRONMENT:-${repository_root}/.venv-robocasa-gr1}/bin/python"
storage_root="${ART_EMBODIED_STORAGE_ROOT:-${repository_root}}"
egl_library_dir="${ART_EMBODIED_EGL_LIBRARY_DIR:-${repository_root}/.runtime-libs/egl}"
dataset_root="${ART_EMBODIED_ROBOCASA_DATASET_ROOT:-${storage_root}/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim/09c6de8af50168090e7e9cc01e1ec3bce788de24}"
manifest="${repository_root}/examples/embodied/robocasa/gr1_tabletop_tasks.yaml"
dataset_report="${output%.json}-dataset.json"

[[ -x "${python}" ]] || { echo "Missing RoboCasa runtime: ${python}" >&2; exit 1; }
mkdir -p "$(dirname -- "${output}")" "${repository_root}/outputs/slurm"
cd "${repository_root}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
if [[ -d "${egl_library_dir}" ]]; then
  [[ -e "${egl_library_dir}/libEGL.so.1" ]] || {
    echo "ART_EMBODIED_EGL_LIBRARY_DIR lacks libEGL.so.1: ${egl_library_dir}" >&2
    exit 1
  }
  export LD_LIBRARY_PATH="${egl_library_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export PYTHONUNBUFFERED=1
"${repository_root}/.venv-gr00t-n1d7/bin/python" -m \
  examples.embodied.robocasa.verify_dataset \
  --root "${dataset_root}" \
  --manifest "${manifest}" \
  --output "${dataset_report}"
"${repository_root}/.venv-gr00t-n1d7/bin/python" -m \
  examples.embodied.robocasa.conformance \
  --config "${config}" \
  --output "${output}"
