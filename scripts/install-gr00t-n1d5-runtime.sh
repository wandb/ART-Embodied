#!/usr/bin/env bash
set -euo pipefail

# GR00T N1.5 is isolated because its pinned Transformers/Torch stack conflicts
# with the LeRobot 0.6 policy runtimes used by PI and SmolVLA.
repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
environment_path="${1:-${repository_root}/.venv-gr00t-n1d5}"
python_version="${ART_EMBODIED_GR00T_PYTHON:-3.11}"
gr00t_revision="4af2b622892f7dcb5aae5a3fb70bcb02dc217b96"

command -v uv >/dev/null || {
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
}

uv venv --python "${python_version}" "${environment_path}"
python="${environment_path}/bin/python"

uv pip install --python "${python}" \
  "gr00t[base] @ git+https://github.com/NVIDIA/Isaac-GR00T.git@${gr00t_revision}"

# NVIDIA's N1.5 Eagle backbone imports this package unconditionally. Use the
# official release wheel matching its pinned Torch 2.5/CUDA 12 runtime; login
# nodes are not required to carry nvcc merely to construct the environment.
flash_attn_wheel="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post4/flash_attn-2.7.1.post4%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
uv pip install --python "${python}" "${flash_attn_wheel}"

# The archived N1.5 package metadata pins W&B 0.18. ART-Embodied intentionally
# uses a newer W&B client so GR00T shares the same metric, media, artifact, and
# Weave contract as every other policy family. These are observability-only
# dependencies; protobuf remains on N1.5's TensorFlow-compatible major version.
uv pip install --python "${python}" \
  -e "${repository_root}[observability]" \
  "protobuf==4.25.1" \
  "wandb==0.24.2" \
  "weave==0.52.37"
uv pip install --python "${python}" "hf-libero==0.1.4"
# NVIDIA's frozen N1.5 package pins this version because numpydantic 1.6.7
# cannot build its array schema with Pydantic 2.13.
uv pip install --python "${python}" "pydantic==2.10.6"
uv pip install --python "${python}" "typing-extensions==4.12.2"
# ``openpipe-art`` currently resolves a newer chardet than requests supports.
# Keep the isolated runtime warning-free rather than relying on requests'
# charset detector fallback.
uv pip install --python "${python}" "chardet==5.2.0"

"${python}" - <<'PY'
import flash_attn
import gr00t
from importlib.metadata import version
import pydantic
import torch
import transformers
import wandb
import weave

print(f"gr00t={gr00t.__file__}")
print(f"pydantic={pydantic.__version__}")
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"typing-extensions={version('typing-extensions')}")
print(f"flash_attn={flash_attn.__version__}")
print(f"wandb={wandb.__version__}")
print(f"weave={weave.__version__}")
PY
