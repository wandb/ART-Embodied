#!/usr/bin/env bash
#SBATCH --job-name=gr00t-n17-cutpan-u20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=12:00:00
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
config="${ART_EMBODIED_ROBOCASA_CONFIG:-${repository_root}/examples/embodied/gr00t_n1d7_robocasa_gr1_tabletop_cuttingboard_pan_noise01_u20_development.yaml}"
checkpoint="${ART_EMBODIED_ROBOCASA_SFT_CHECKPOINT:-${repository_root}/outputs/gr00t-n1d7-robocasa-sft/gr00t-n1d7-robocasa-gr1-tabletop-sft-u60000/checkpoint-60000}"
preregistration="${ART_EMBODIED_ROBOCASA_PREREGISTRATION:-${repository_root}/examples/embodied/robocasa/development_experiments/gr1_tabletop_cuttingboard_pan_noise01_u20_v1.json}"
calibration_report="${ART_EMBODIED_ROBOCASA_CALIBRATION_REPORT:-${repository_root}/outputs/gr00t-n1d7-robocasa-sampler-calibration/cuttingboard-pan-k4-noise01-post-oracle-v1/post-oracle-sampler-calibration-report.json}"
conformance_report="${ART_EMBODIED_ROBOCASA_CONFORMANCE_REPORT:-${repository_root}/outputs/conformance/robocasa-gr1-cuttingboard-pan-noise01-u20-v1.json}"
policy_conformance_report="${ART_EMBODIED_ROBOCASA_POLICY_CONFORMANCE_REPORT:-${repository_root}/outputs/conformance/gr00t-n17-robocasa-policy-cuttingboard-pan-noise01-u20-v2.json}"
wandb_report="${ART_EMBODIED_WANDB_REPORT:-${repository_root}/outputs/conformance/gr00t-n17-robocasa-single-task-wandb-preflight-20260819.json}"

[[ -x "${python}" ]] || { echo "Missing GR00T N1.7 runtime: ${python}" >&2; exit 1; }
[[ -x "${repository_root}/.venv-robocasa-gr1/bin/python" ]] || {
  echo "Missing RoboCasa runtime" >&2
  exit 1
}
[[ -f "${config}" ]] || { echo "Missing experiment YAML: ${config}" >&2; exit 1; }
for required in "${preregistration}" "${calibration_report}" \
  "${conformance_report}" "${policy_conformance_report}" "${wandb_report}"; do
  [[ -f "${required}" ]] || { echo "Missing admission evidence: ${required}" >&2; exit 1; }
done
[[ -e "${ffmpeg_library_dir}/libavcodec.so.61" ]] || {
  echo "Missing pinned FFmpeg 7 runtime" >&2
  exit 1
}
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN is required" >&2; exit 1; }
[[ -n "${WANDB_API_KEY:-}" ]] || { echo "WANDB_API_KEY is required" >&2; exit 1; }
[[ -f "${checkpoint}/art_embodied_sft_complete.json" ]] || {
  echo "SFT completion marker is missing: ${checkpoint}" >&2
  exit 1
}

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

# A primary coordinator owns one new W&B run. Inherited attachment variables
# would either fragment history or overwrite an earlier run's console record.
unset WANDB_RUN_ID WANDB_RESUME

"${python}" -m art_embodied.cli validate "${config}" --json \
  > "${repository_root}/outputs/slurm/gr00t-n17-cutpan-u20-${SLURM_JOB_ID}-config.json"
"${python}" -P -m examples.embodied.robocasa.train \
  --config "${config}" \
  --preflight \
  > "${repository_root}/outputs/slurm/gr00t-n17-cutpan-u20-${SLURM_JOB_ID}-preflight.json"

"${python}" - \
  "${config}" \
  "${checkpoint}/art_embodied_sft_complete.json" \
  "${preregistration}" \
  "${calibration_report}" \
  "${conformance_report}" \
  "${policy_conformance_report}" \
  "${wandb_report}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

from art_embodied import EmbodiedExperimentConfig

config_path = Path(sys.argv[1])
marker_path = Path(sys.argv[2])
preregistration_path = Path(sys.argv[3])
calibration_report_path = Path(sys.argv[4])
environment_report_path = Path(sys.argv[5])
policy_report_path = Path(sys.argv[6])
wandb_report_path = Path(sys.argv[7])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


config = EmbodiedExperimentConfig.from_yaml(config_path)
marker = json.loads(marker_path.read_text(encoding="utf-8"))
if marker.get("schema_version") != 1 or marker.get("status") != "complete":
    raise SystemExit(f"Invalid SFT completion marker: {marker_path}")
if marker.get("optimizer_steps") != 60_000:
    raise SystemExit("GRPO gate requires the official 60,000-step SFT warm start")
preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
frozen = preregistration["frozen_inputs"]
if frozen.get("config_sha256") != sha256(config_path):
    raise SystemExit("The executable config differs from the preregistration")
if frozen.get("sft_completion_marker_sha256") != sha256(marker_path):
    raise SystemExit("The SFT completion marker differs from the preregistration")
if frozen.get("post_oracle_sampler_report_sha256") != sha256(
    calibration_report_path
):
    raise SystemExit("The sampler calibration differs from the preregistration")
if frozen.get("environment_conformance_report_sha256") != sha256(
    environment_report_path
):
    raise SystemExit("Environment conformance differs from the preregistration")
if frozen.get("policy_conformance_report_sha256") != sha256(policy_report_path):
    raise SystemExit("Policy conformance differs from the preregistration")
for relative, expected in frozen.get("implementation_sha256", {}).items():
    source = Path(relative)
    if not source.is_file() or sha256(source) != expected:
        raise SystemExit(f"Frozen implementation changed: {relative}")

calibration = json.loads(calibration_report_path.read_text(encoding="utf-8"))
if calibration.get("decision") != "pass" or not calibration.get(
    "training_admission"
):
    raise SystemExit("Post-oracle sampler calibration did not admit training")
environment_report = json.loads(environment_report_path.read_text(encoding="utf-8"))
if (
    environment_report.get("status") != "passed"
    or environment_report.get("config_fingerprint") != config.fingerprint
):
    raise SystemExit("Environment conformance does not match this config")
policy_report = json.loads(policy_report_path.read_text(encoding="utf-8"))
if (
    policy_report.get("status") != "passed"
    or policy_report.get("config_fingerprint") != config.fingerprint
):
    raise SystemExit(
        "The model-backed RoboCasa conformance report did not pass for this YAML"
    )
wandb_report = json.loads(wandb_report_path.read_text(encoding="utf-8"))
if (
    wandb_report.get("status") != "passed"
    or wandb_report.get("entity") != config.observability.wandb.entity
    or wandb_report.get("project") != config.observability.wandb.project
    or frozen.get("wandb_roundtrip_report_sha256") != sha256(wandb_report_path)
):
    raise SystemExit("W&B round-trip report does not match this config")
if (
    config.training.updates != 20
    or config.algorithm.flow_sde.noise_level != 0.1
    or config.storage.resume_from_checkpoint is not None
    or config.evaluation.kwargs["seed_contract"]["environment_mode"]
    != "configured"
):
    raise SystemExit("Executable config violates the preregistered recipe")
if (config.storage.output_dir / "checkpoints").exists():
    raise SystemExit("Fresh-SFT run output already contains checkpoints")
print(f"Admission passed for config fingerprint {config.fingerprint}")
PY

adjudicate() {
  "${python}" -P -m examples.embodied.robocasa.adjudicate_one_task_u20 \
    --config "${config}" \
    --preregistration "${preregistration}" \
    --calibration-report "${calibration_report}" \
    --output "${repository_root}/outputs/gr00t-n1d7-robocasa-single-task/cuttingboard-pan-noise01-r010-r075-r64-lr3e5-u20-v1/development_adjudication.json"
}

startup_deadline_seconds="${ART_EMBODIED_GPU_STARTUP_DEADLINE_SECONDS:-300}"
minimum_gpu_memory_mib="${ART_EMBODIED_GPU_STARTUP_MIN_MEMORY_MIB:-1024}"

"${repository_root}/scripts/slurm/run-in-repo.sh" \
  "${python}" -P -m examples.embodied.robocasa.train \
  --config "${config}" &
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
    echo \
      "GPU startup deadline exceeded: max memory ${max_gpu_memory_mib} MiB " \
      "after ${startup_deadline_seconds}s" >&2
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
