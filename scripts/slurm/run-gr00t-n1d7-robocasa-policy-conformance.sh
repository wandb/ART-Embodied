#!/usr/bin/env bash
#SBATCH --job-name=gr00t-n17-robocasa-policy-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=01:00:00
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
egl_library_dir="${ART_EMBODIED_EGL_LIBRARY_DIR:-${repository_root}/.runtime-libs/egl}"
config="${ART_EMBODIED_ROBOCASA_CONFIG:-${repository_root}/examples/embodied/gr00t_n1d7_robocasa_gr1_tabletop_flow_sde_grpo_gate.yaml}"
oracle_manifest="${ART_EMBODIED_ROBOCASA_ORACLE_MANIFEST:-${repository_root}/examples/embodied/robocasa/oracle_parity_manifest.json}"
job_label="${SLURM_JOB_ID}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  job_label="${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
fi
output="${ART_EMBODIED_ROBOCASA_POLICY_CONFORMANCE_OUTPUT:-${repository_root}/outputs/conformance/gr00t-n17-robocasa-policy-${job_label}.json}"
checkpoint="${ART_EMBODIED_ROBOCASA_SFT_CHECKPOINT:-${repository_root}/outputs/gr00t-n1d7-robocasa-sft/gr00t-n1d7-robocasa-gr1-tabletop-sft-u60000/checkpoint-60000}"
policy_checkpoint="${ART_EMBODIED_ROBOCASA_POLICY_CHECKPOINT:-}"
oracle_only="${ART_EMBODIED_ROBOCASA_ORACLE_ONLY:-0}"
oracle_full_episode="${ART_EMBODIED_ROBOCASA_ORACLE_FULL_EPISODE:-0}"
oracle_task_id="${ART_EMBODIED_ROBOCASA_ORACLE_TASK_ID:-}"
oracle_tasks=(
  PnPCupToDrawerClose
  PnPMilkToMicrowaveClose
  PnPPotatoToMicrowaveClose
  PosttrainPnPNovelFromCuttingboardToPanSplitA
  PosttrainPnPNovelFromPlacematToBasketSplitA
  PosttrainPnPNovelFromPlateToBowlSplitA
  PosttrainPnPNovelFromTrayToPotSplitA
  PosttrainPnPNovelFromTrayToTieredbasketSplitA
)
if [[ -z "${oracle_task_id}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  oracle_task_id="${oracle_tasks[SLURM_ARRAY_TASK_ID]}"
fi
ffmpeg_library_dir="${repository_root}/.runtime-libs/ffmpeg7/lib"

[[ -x "${python}" ]] || { echo "Missing GR00T N1.7 runtime" >&2; exit 1; }
[[ -x "${repository_root}/.venv-robocasa-gr1/bin/python" ]] || {
  echo "Missing RoboCasa runtime" >&2
  exit 1
}
[[ -f "${checkpoint}/art_embodied_sft_complete.json" ]] || {
  echo "SFT checkpoint is incomplete: ${checkpoint}" >&2
  exit 1
}
[[ -e "${ffmpeg_library_dir}/libavcodec.so.61" ]] || {
  echo "Missing pinned FFmpeg 7 runtime" >&2
  exit 1
}

mkdir -p "$(dirname -- "${output}")" "${repository_root}/outputs/slurm"
cd "${repository_root}"
export ART_EMBODIED_REPO_ROOT="${repository_root}"
export LD_LIBRARY_PATH="${ffmpeg_library_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
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
export TOKENIZERS_PARALLELISM=false

args=("${python}" -P -m examples.embodied.robocasa.policy_conformance \
  --config "${config}" \
  --output "${output}" \
  --official-gr00t-source "${repository_root}/.runtime-sources/isaac-gr00t-376ba890cff8c9de64d71d982772a9c36185fdd7" \
  --oracle-manifest "${oracle_manifest}")
if [[ -n "${policy_checkpoint}" ]]; then
  [[ -f "${policy_checkpoint}/art_embodied_gr00t_n1d7_snapshot.json" ]] || {
    echo "Policy checkpoint is incomplete: ${policy_checkpoint}" >&2
    exit 1
  }
  args+=(--policy-checkpoint "${policy_checkpoint}")
fi
if [[ "${oracle_only}" == "1" ]]; then
  args+=(--oracle-only)
fi
if [[ "${oracle_full_episode}" == "1" ]]; then
  args+=(--oracle-full-episode)
fi
if [[ -n "${oracle_task_id}" ]]; then
  args+=(--oracle-task-id "${oracle_task_id}")
fi
exec "${args[@]}"
