#!/usr/bin/env bash
#SBATCH --job-name=robocasa-reset-repeatability
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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

config="${ART_EMBODIED_ROBOCASA_CONFIG:-${repository_root}/examples/embodied/gr00t_n1d7_robocasa_gr1_tabletop_frontier8_stability_noise01_development.yaml}"
output="${ART_EMBODIED_ROBOCASA_RESET_OUTPUT:-${repository_root}/outputs/conformance/robocasa-reset-repeatability-${SLURM_JOB_ID}.json}"
egl_library_dir="${ART_EMBODIED_EGL_LIBRARY_DIR:-${repository_root}/.runtime-libs/egl}"

[[ -x "${repository_root}/.venv-gr00t-n1d7/bin/python" ]] || {
  echo "Missing GR00T runtime" >&2
  exit 1
}
[[ -x "${repository_root}/.venv-robocasa-gr1/bin/python" ]] || {
  echo "Missing RoboCasa runtime" >&2
  exit 1
}

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
export PYTHONPATH="${repository_root}/src:${repository_root}"
export PYTHONSAFEPATH=1
export PYTHONUNBUFFERED=1

exec "${repository_root}/.venv-gr00t-n1d7/bin/python" -P -m \
  examples.embodied.robocasa.reset_repeatability \
  --config "${config}" \
  --output "${output}" \
  --worker-repeats 2
