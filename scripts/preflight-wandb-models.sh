#!/usr/bin/env bash

set -euo pipefail

repository_root="${ART_EMBODIED_REPO_ROOT:-}"
if [[ -z "${repository_root}" ]]; then
  repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fi
env_file="${ART_EMBODIED_ENV_FILE:-${repository_root}/.env}"
if [[ -f "${env_file}" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${env_file}"
  set +a
fi

python="${ART_EMBODIED_PREFLIGHT_PYTHON:-${repository_root}/.venv-gr00t-n1d7/bin/python}"
entity="${ART_EMBODIED_WANDB_PREFLIGHT_ENTITY:-wandb-japan}"
project="${ART_EMBODIED_WANDB_PREFLIGHT_PROJECT:-art-embodied-observability-canary}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output="${ART_EMBODIED_WANDB_PREFLIGHT_OUTPUT:-${repository_root}/outputs/observability-canary/roundtrip-${timestamp}.json}"

[[ -x "${python}" ]] || { echo "Missing preflight Python: ${python}" >&2; exit 1; }
[[ -n "${WANDB_API_KEY:-}" ]] || { echo "WANDB_API_KEY is required" >&2; exit 1; }

cd "${repository_root}"
export PYTHONPATH="${repository_root}/src:${repository_root}"
exec "${python}" -P -m art_embodied.wandb_preflight \
  --entity "${entity}" \
  --project "${project}" \
  --output "${output}" \
  --timeout-seconds "${ART_EMBODIED_WANDB_PREFLIGHT_TIMEOUT_SECONDS:-120}" \
  --poll-seconds "${ART_EMBODIED_WANDB_PREFLIGHT_POLL_SECONDS:-10}"
