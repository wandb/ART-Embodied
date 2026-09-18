#!/usr/bin/env bash
#SBATCH --job-name=gr00t-n17-cutpan-sealed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=06:00:00
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
ffmpeg_library_dir="${repository_root}/.runtime-libs/ffmpeg7/lib"
egl_library_dir="${ART_EMBODIED_EGL_LIBRARY_DIR:-${repository_root}/.runtime-libs/egl}"
manifest="${repository_root}/examples/embodied/robocasa/sealed_tests/gr1_tabletop_cuttingboard_pan_u100_v1.json"
expected_manifest_sha256="10ae4dcc3c194f9b422bfadb70a7cb01134eb88417b787caaf685af5550e8912"
baseline_config="${repository_root}/examples/embodied/gr00t_n1d7_robocasa_gr1_tabletop_cuttingboard_pan_u100_sealed_v1_sft_baseline.yaml"
candidate_config="${repository_root}/examples/embodied/gr00t_n1d7_robocasa_gr1_tabletop_cuttingboard_pan_u100_sealed_v1_candidate.yaml"
development_dir="${repository_root}/outputs/gr00t-n1d7-robocasa-single-task/cuttingboard-pan-noise01-r010-r075-r64-lr3e5-u20-v1"
development_adjudication="${development_dir}/development_adjudication_u100.json"
candidate_checkpoint="${development_dir}/checkpoints/step-000100"
candidate_policy="${candidate_checkpoint}/policy"
baseline_outcomes="${repository_root}/outputs/gr00t-n1d7-robocasa-sealed/cuttingboard-pan-u100-v1-sft-baseline/evaluation/update_000000_episode_outcomes.json"
candidate_outcomes="${repository_root}/outputs/gr00t-n1d7-robocasa-sealed/cuttingboard-pan-u100-v1-candidate/evaluation/update_000100_episode_outcomes.json"
baseline_evidence="${repository_root}/outputs/gr00t-n1d7-robocasa-sealed/cuttingboard-pan-u100-v1-sft-baseline/evaluation/update_000000_evidence.json"
candidate_evidence="${repository_root}/outputs/gr00t-n1d7-robocasa-sealed/cuttingboard-pan-u100-v1-candidate/evaluation/update_000100_evidence.json"
sealed_adjudication="${repository_root}/outputs/gr00t-n1d7-robocasa-sealed/cuttingboard-pan-u100-v1-adjudication.json"

[[ -x "${python}" ]] || { echo "Missing GR00T N1.7 runtime: ${python}" >&2; exit 1; }
[[ -x "${repository_root}/.venv-robocasa-gr1/bin/python" ]] || {
  echo "Missing RoboCasa runtime" >&2
  exit 1
}
required_files=(
  "${manifest}"
  "${baseline_config}"
  "${candidate_config}"
  "${development_adjudication}"
  "${candidate_checkpoint}/art_embodied_checkpoint_complete.json"
  "${candidate_policy}/art_embodied_gr00t_n1d7_snapshot.json"
)
for required in "${required_files[@]}"; do
  [[ -f "${required}" ]] || { echo "Missing sealed input: ${required}" >&2; exit 1; }
done
actual_manifest_sha256="$(sha256sum "${manifest}" | awk '{print $1}')"
if [[ "${actual_manifest_sha256}" != "${expected_manifest_sha256}" ]]; then
  echo "Sealed manifest changed after job submission" >&2
  exit 1
fi
[[ -e "${ffmpeg_library_dir}/libavcodec.so.61" ]] || {
  echo "Missing pinned FFmpeg 7 runtime" >&2
  exit 1
}
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN is required" >&2; exit 1; }
[[ -n "${WANDB_API_KEY:-}" ]] || { echo "WANDB_API_KEY is required" >&2; exit 1; }
if [[ -e "${baseline_outcomes}" || -e "${candidate_outcomes}" ]]; then
  echo "Sealed outcomes already exist; refusing to rerun or overwrite them" >&2
  exit 1
fi

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
unset WANDB_RUN_ID WANDB_RESUME

"${python}" -m art_embodied.cli validate "${baseline_config}" --json > "${repository_root}/outputs/slurm/gr00t-n17-cutpan-sealed-baseline-${SLURM_JOB_ID}-config.json"
"${python}" -m art_embodied.cli validate "${candidate_config}" --json > "${repository_root}/outputs/slurm/gr00t-n17-cutpan-sealed-candidate-${SLURM_JOB_ID}-config.json"

"${python}" - "${manifest}" "${development_adjudication}" "${baseline_config}" "${candidate_config}" "${candidate_checkpoint}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1])
adjudication_path = Path(sys.argv[2])
baseline_config = Path(sys.argv[3])
candidate_config = Path(sys.argv[4])
candidate_checkpoint = Path(sys.argv[5])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
if adjudication.get("status") != "passed" or not adjudication.get("sealed_eligible"):
    raise SystemExit("Update-100 development gate did not admit sealed evaluation")
if (
    manifest.get("sealed_outcomes_observed") is not False
    or manifest.get("checkpoint_selection_rule", "").split(",", 1)[0]
    != "update 100 only"
    or manifest.get("baseline_config_sha256") != sha256(baseline_config)
    or manifest.get("candidate_config_sha256") != sha256(candidate_config)
):
    raise SystemExit("Sealed manifest/configs drifted after being frozen")
checkpoint_marker = json.loads(
    (candidate_checkpoint / "art_embodied_checkpoint_complete.json").read_text(
        encoding="utf-8"
    )
)
if (
    checkpoint_marker.get("complete") is not True
    or int(checkpoint_marker.get("metadata", {}).get("step", -1)) != 100
    or checkpoint_marker.get("resume_contract_fingerprint") != "691554e6497ccb86"
):
    raise SystemExit("The sealed candidate is not the frozen update-100 policy")
print("Sealed admission passed; running the paired panel once without intervention")
PY

"${python}" -P -m examples.embodied.robocasa.train --config "${baseline_config}" --evaluate-only --evaluation-step 0 --preflight > "${repository_root}/outputs/slurm/gr00t-n17-cutpan-sealed-baseline-${SLURM_JOB_ID}-preflight.json"
"${python}" -P -m examples.embodied.robocasa.train --config "${candidate_config}" --evaluate-only --evaluation-step 100 --policy-checkpoint "${candidate_checkpoint}" --preflight > "${repository_root}/outputs/slurm/gr00t-n17-cutpan-sealed-candidate-${SLURM_JOB_ID}-preflight.json"

# Baseline and candidate run back-to-back in one allocation. No decision point
# exists between observing baseline outcomes and executing the frozen candidate.
"${repository_root}/scripts/slurm/run-in-repo.sh" "${python}" -P -m examples.embodied.robocasa.train --config "${baseline_config}" --evaluate-only --evaluation-step 0
"${repository_root}/scripts/slurm/run-in-repo.sh" "${python}" -P -m examples.embodied.robocasa.train --config "${candidate_config}" --evaluate-only --evaluation-step 100 --policy-checkpoint "${candidate_checkpoint}"

"${python}" -P -m examples.embodied.robocasa.adjudicate_one_task_u100_sealed \
  --manifest "${manifest}" \
  --baseline "${baseline_outcomes}" \
  --candidate "${candidate_outcomes}" \
  --baseline-evidence "${baseline_evidence}" \
  --candidate-evidence "${candidate_evidence}" \
  --candidate-checkpoint "${candidate_checkpoint}" \
  --output "${sealed_adjudication}"
