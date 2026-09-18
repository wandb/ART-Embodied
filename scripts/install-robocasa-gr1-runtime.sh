#!/usr/bin/env bash
set -euo pipefail

# RoboCasa deliberately lives outside the GR00T policy environment. Its pinned
# MuJoCo, robosuite, and Gymnasium versions conflict with the N1.7 model stack.
repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${repository_root}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${repository_root}/.env"
  set +a
fi

source_revision="4840e671596f93ca03651524b9f72ffb1aadfeff"
storage_root="${ART_EMBODIED_STORAGE_ROOT:-}"
environment_default="${repository_root}/.venv-robocasa-gr1"
source_default="${repository_root}/.runtime-sources/robocasa-gr1-tabletop-${source_revision}"
if [[ -n "${storage_root}" ]]; then
  environment_default="${storage_root}/environments/robocasa-gr1-tabletop-${source_revision}"
  source_default="${storage_root}/sources/robocasa-gr1-tabletop-tasks-${source_revision}"
fi
environment_path="${1:-${ART_EMBODIED_ROBOCASA_ENVIRONMENT:-${environment_default}}}"
python_version="${ART_EMBODIED_ROBOCASA_PYTHON:-3.12}"
source_path="${ART_EMBODIED_ROBOCASA_SOURCE:-${source_default}}"

# Source snapshots expose shared runtimes through stable symlinks. Build the
# environment at the target rather than asking uv to replace the symlink.
if [[ -L "${environment_path}" ]]; then
  environment_path="$(readlink -f -- "${environment_path}")"
fi

command -v uv >/dev/null || {
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
}

mkdir -p "$(dirname -- "${source_path}")" "$(dirname -- "${environment_path}")"
if [[ ! -d "${source_path}/.git" ]]; then
  git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks "${source_path}"
fi
git -C "${source_path}" fetch origin "${source_revision}"
git -C "${source_path}" checkout --detach "${source_revision}"
actual_revision="$(git -C "${source_path}" rev-parse HEAD)"
if [[ "${actual_revision}" != "${source_revision}" ]]; then
  echo "RoboCasa source revision mismatch: ${actual_revision}" >&2
  exit 1
fi

UV_LINK_MODE=copy uv venv --clear --python "${python_version}" "${environment_path}"
python="${environment_path}/bin/python"

UV_LINK_MODE=copy uv pip install --python "${python}" setuptools wheel
UV_LINK_MODE=copy uv pip install --python "${python}" \
  "git+https://github.com/ARISE-Initiative/robosuite.git@v1.5.1"
UV_LINK_MODE=copy uv pip install --python "${python}" \
  --no-deps -e "${source_path}" --config-settings editable_mode=compat
UV_LINK_MODE=copy uv pip install --python "${python}" \
  "gymnasium==0.29.1" \
  "h5py>=3.14,<4" \
  "hidapi>=0.14,<1" \
  "imageio>=2.37,<3" \
  "lxml>=6,<7" \
  "mujoco==3.2.6" \
  "numba==0.61.2" \
  "numpy==1.26.4" \
  "opencv-python==4.11.0.86" \
  "pillow>=11,<13" \
  "pygame>=2.6,<3" \
  "pynput>=1.8,<2" \
  "pyyaml>=6,<7" \
  "scipy>=1.15,<2" \
  "termcolor>=3,<4" \
  "tqdm>=4.67,<5"

assets_marker="${source_path}/robocasa/models/assets/.art-embodied-tabletop-assets-complete"
if [[ ! -f "${assets_marker}" ]]; then
  "${python}" "${source_path}/robocasa/scripts/download_tabletop_assets.py" -y
  touch "${assets_marker}"
fi

# Runtime installation only verifies imports and version pins. The GPU-backed
# conformance job separately exercises EGL rendering on an allocated device.
PYTHONPATH="${repository_root}/src:${repository_root}" \
MUJOCO_GL=disabled \
"${python}" - <<'PY'
from importlib.metadata import version

import gymnasium
import mujoco
import robocasa
import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
import robosuite

assert mujoco.__version__ == "3.2.6"
assert robosuite.__version__ == "1.5.1"
assert gymnasium.__version__ == "0.29.1"
print(f"robocasa={robocasa.__version__}")
for package in ("gymnasium", "mujoco", "numpy", "robocasa", "robosuite"):
    print(f"{package}={version(package)}")
PY
