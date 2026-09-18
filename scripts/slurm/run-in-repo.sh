#!/usr/bin/env bash
set -euo pipefail

# Slurm copies batch scripts into its spool directory. Prefer the explicit
# override, then Slurm's original submission directory, and use the script
# location only for direct execution where BASH_SOURCE still points at the
# checkout.
if [[ -n "${ART_EMBODIED_REPO_ROOT:-}" ]]; then
  repo_root="${ART_EMBODIED_REPO_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  repo_root="${SLURM_SUBMIT_DIR}"
else
  script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="${script_dir}/../.."
fi
repo_root="$(cd -- "${repo_root}" && pwd)"
if [[ ! -f "${repo_root}/pyproject.toml" ]]; then
  echo "ART-Embodied repository root is invalid: ${repo_root}" >&2
  echo "Submit from the checkout or set ART_EMBODIED_REPO_ROOT." >&2
  exit 2
fi
env_file="${ART_EMBODIED_ENV_FILE:-${repo_root}/.env}"
runtime_lib_dir="${ART_EMBODIED_RUNTIME_LIB_DIR:-${repo_root}/.runtime/lib}"

# Python wheels cannot install host GLVND/NVIDIA libraries. A cluster image
# should provide them system-wide; this ignored, repo-relative fallback keeps
# machine-local runtime shims portable without embedding a user's home path.
if [[ -d "${runtime_lib_dir}" ]]; then
  export LD_LIBRARY_PATH="${runtime_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [[ -f "${env_file}" ]]; then
  # Export sourced assignments to the launched process without forcing users to
  # duplicate secrets in Slurm directives or command-line arguments.
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
fi

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 COMMAND [ARG ...]" >&2
  exit 2
fi

cd "${repo_root}"
exec "$@"
