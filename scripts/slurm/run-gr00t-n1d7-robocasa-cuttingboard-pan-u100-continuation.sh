#!/usr/bin/env bash
#SBATCH --job-name=gr00t-n17-cutpan-u100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=36:00:00
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
ffmpeg_library_dir="${repository_root}/.runtime-libs/ffmpeg7/lib"
config="${ART_EMBODIED_ROBOCASA_CONFIG:-${repository_root}/examples/embodied/robocasa/development_experiments/gr1_tabletop_cuttingboard_pan_noise01_u100_continuation_v1.yaml}"
parent_config="${repository_root}/examples/embodied/gr00t_n1d7_robocasa_gr1_tabletop_cuttingboard_pan_noise01_u20_development.yaml"
preregistration="${ART_EMBODIED_ROBOCASA_PREREGISTRATION:-${repository_root}/examples/embodied/robocasa/development_experiments/gr1_tabletop_cuttingboard_pan_noise01_u100_continuation_v1.json}"
parent_preregistration="${repository_root}/examples/embodied/robocasa/development_experiments/gr1_tabletop_cuttingboard_pan_noise01_u20_v1.json"
output_dir="${repository_root}/outputs/gr00t-n1d7-robocasa-single-task/cuttingboard-pan-noise01-r010-r075-r64-lr3e5-u20-v1"
checkpoint="${output_dir}/checkpoints/step-000020"
parent_adjudication="${output_dir}/development_adjudication.json"

[[ -x "${python}" ]] || { echo "Missing GR00T N1.7 runtime: ${python}" >&2; exit 1; }
[[ -x "${repository_root}/.venv-robocasa-gr1/bin/python" ]] || {
  echo "Missing RoboCasa runtime" >&2
  exit 1
}
required_files=(
  "${config}"
  "${parent_config}"
  "${preregistration}"
  "${parent_preregistration}"
  "${parent_adjudication}"
  "${checkpoint}/art_embodied_checkpoint_complete.json"
  "${checkpoint}/art_embodied_training_state.pt"
)
for required in "${required_files[@]}"; do
  [[ -f "${required}" ]] || { echo "Missing continuation input: ${required}" >&2; exit 1; }
done
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
    echo "ART_EMBODIED_EGL_LIBRARY_DIR lacks libEGL.so.1: ${egl_library_dir}" >&2
    exit 1
  }
  export LD_LIBRARY_PATH="${egl_library_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONSAFEPATH=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${repository_root}/src:${repository_root}"

# The YAML owns recovery of the existing coordinator run.
unset WANDB_RUN_ID WANDB_RESUME

"${python}" -m art_embodied.cli validate "${config}" --json > "${repository_root}/outputs/slurm/gr00t-n17-cutpan-u100-${SLURM_JOB_ID}-config.json"
"${python}" -P -m examples.embodied.robocasa.train --config "${config}" --preflight > "${repository_root}/outputs/slurm/gr00t-n17-cutpan-u100-${SLURM_JOB_ID}-preflight.json"

"${python}" - "${config}" "${parent_config}" "${preregistration}" "${parent_preregistration}" "${parent_adjudication}" "${checkpoint}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import sys

from art_embodied import EmbodiedExperimentConfig

config_path = Path(sys.argv[1])
parent_config_path = Path(sys.argv[2])
preregistration_path = Path(sys.argv[3])
parent_preregistration_path = Path(sys.argv[4])
parent_adjudication_path = Path(sys.argv[5])
checkpoint = Path(sys.argv[6])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


config = EmbodiedExperimentConfig.from_yaml(config_path)
parent = EmbodiedExperimentConfig.from_yaml(parent_config_path)
preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
frozen = preregistration["frozen_inputs"]
if sha256(config_path) != frozen["config_sha256"]:
    raise SystemExit("The continuation config differs from its preregistration")
if sha256(parent_config_path) != frozen["parent_config_sha256"]:
    raise SystemExit("The parent config differs from the frozen continuation input")
if sha256(parent_preregistration_path) != frozen["parent_preregistration_sha256"]:
    raise SystemExit("The parent preregistration differs from the frozen input")
for relative, expected in frozen["implementation_sha256"].items():
    source = Path(relative)
    if not source.is_file() or sha256(source) != expected:
        raise SystemExit(f"Frozen implementation changed: {relative}")
if config.resume_contract_fingerprint != parent.resume_contract_fingerprint:
    raise SystemExit("Continuation training math differs from the parent run")
if config.resume_contract_fingerprint != frozen["resume_contract_fingerprint"]:
    raise SystemExit("Continuation resume fingerprint differs from preregistration")
if (
    config.training.updates != 100
    or config.experiment.seed != parent.experiment.seed
    or config.storage.resume_from_checkpoint.resolve() != checkpoint.resolve()
    or config.evaluation.evaluate_before_training
    or config.evaluation.every_updates != 5
    or config.observability.wandb.connection != "resume"
    or config.observability.wandb.run_id != frozen["parent_wandb_run_id"]
    or config.observability.wandb.resume != "must"
):
    raise SystemExit("Executable config violates the continuation contract")

marker = json.loads(
    (checkpoint / "art_embodied_checkpoint_complete.json").read_text(encoding="utf-8")
)
state = json.loads(
    (checkpoint / "art_embodied_training_state.json").read_text(encoding="utf-8")
)
if (
    marker.get("complete") is not True
    or marker.get("metadata", {}).get("step") != 20
    or state.get("step") != 20
    or marker.get("config_fingerprint") != parent.fingerprint
    or marker.get("resume_contract_fingerprint")
    != config.resume_contract_fingerprint
):
    raise SystemExit("Step-20 checkpoint is incomplete or belongs to another contract")

adjudication = json.loads(parent_adjudication_path.read_text(encoding="utf-8"))
checks = adjudication.get("gate_checks", {})
curve = adjudication.get("evaluation_curve", [])
update20 = next(
    (point for point in curve if int(point.get("update", -1)) == 20),
    None,
)
if (
    not checks.get("all_evaluations_complete")
    or not checks.get("all_training_rollouts_complete")
    or update20 is None
    or not math.isfinite(float(update20.get("success_rate", float("nan"))))
    or float(update20["success_rate"]) < 0.25
):
    raise SystemExit(
        "Update-20 operational/collapse gate failed; do not spend the update-100 budget"
    )
if (config.storage.output_dir / "checkpoints/step-000100").exists():
    raise SystemExit("Update-100 checkpoint already exists; refusing duplicate training")
print(
    "Continuation admitted: exact policy/Adam resume from update 20 to update 100 "
    f"under contract {config.resume_contract_fingerprint}"
)
PY

adjudicate() {
  "${python}" -P -m examples.embodied.robocasa.adjudicate_one_task_u100 --config "${config}" --preregistration "${preregistration}" --output "${output_dir}/development_adjudication_u100.json"
}

startup_deadline_seconds="${ART_EMBODIED_GPU_STARTUP_DEADLINE_SECONDS:-300}"
minimum_gpu_memory_mib="${ART_EMBODIED_GPU_STARTUP_MIN_MEMORY_MIB:-1024}"

"${repository_root}/scripts/slurm/run-in-repo.sh" "${python}" -P -m examples.embodied.robocasa.train --config "${config}" &
training_pid=$!

terminate_training() {
  if kill -0 "${training_pid}" 2>/dev/null; then
    kill -TERM "${training_pid}" 2>/dev/null || true
    sleep 10
    kill -KILL "${training_pid}" 2>/dev/null || true
  fi
}
trap terminate_training EXIT INT TERM

startup_started_at=${SECONDS}
while kill -0 "${training_pid}" 2>/dev/null; do
  max_gpu_memory_mib="$({
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits || true
  } | awk 'BEGIN { max = 0 } $1 > max { max = $1 } END { print max }')"
  if (( max_gpu_memory_mib >= minimum_gpu_memory_mib )); then
    echo "GPU startup confirmed: max memory ${max_gpu_memory_mib} MiB"
    trap - EXIT INT TERM
    wait "${training_pid}"
    adjudicate
    exit $?
  fi
  if (( SECONDS - startup_started_at >= startup_deadline_seconds )); then
    echo "GPU startup deadline exceeded: max memory ${max_gpu_memory_mib} MiB after ${startup_deadline_seconds}s" >&2
    terminate_training
    trap - EXIT INT TERM
    wait "${training_pid}" 2>/dev/null || true
    exit 70
  fi
  sleep 10
done

trap - EXIT INT TERM
wait "${training_pid}"
adjudicate
