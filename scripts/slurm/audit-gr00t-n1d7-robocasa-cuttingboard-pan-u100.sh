#!/usr/bin/env bash
#SBATCH --job-name=gr00t-n17-cutpan-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

repository_root="${ART_EMBODIED_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${repository_root}" ]]; then
  repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
python="${repository_root}/.venv-gr00t-n1d7/bin/python"
auditor="${repository_root}/tools/audit_gr00t_one_task_u100.py"
config="${repository_root}/examples/embodied/robocasa/development_experiments/gr1_tabletop_cuttingboard_pan_noise01_u100_continuation_v1.yaml"
preregistration="${repository_root}/examples/embodied/robocasa/development_experiments/gr1_tabletop_cuttingboard_pan_noise01_u100_continuation_v1.json"
output_dir="${repository_root}/outputs/gr00t-n1d7-robocasa-single-task/cuttingboard-pan-noise01-r010-r075-r64-lr3e5-u20-v1"
adjudication="${output_dir}/development_adjudication_u100.json"
audit_report="${output_dir}/development_evidence_audit_u100.json"
expected_auditor_sha256="a47437e6d006760b8cc43ef17dd59a73850b811e3053439b0df085f76134ef78"

[[ -x "${python}" ]] || { echo "Missing GR00T N1.7 runtime" >&2; exit 1; }
[[ -f "${auditor}" ]] || { echo "Missing independent evidence auditor" >&2; exit 1; }
[[ "$(sha256sum "${auditor}" | awk '{print $1}')" == "${expected_auditor_sha256}" ]] || {
  echo "Independent evidence auditor drifted after submission" >&2
  exit 1
}
[[ ! -e "${audit_report}" ]] || {
  echo "Independent evidence audit already exists; refusing to overwrite" >&2
  exit 1
}

cd "${repository_root}"
export ART_EMBODIED_REPO_ROOT="${repository_root}"
export PYTHONPATH="${repository_root}/src:${repository_root}"
export PYTHONSAFEPATH=1

exec "${python}" "${auditor}" \
  --config "${config}" \
  --preregistration "${preregistration}" \
  --adjudication "${adjudication}" \
  --output "${audit_report}"
