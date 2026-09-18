#!/usr/bin/env bash
set -euo pipefail

# N1.7 requires Python 3.12 and NVIDIA's Torch 2.9/CUDA 12.8 dependency lock.
# Keep it isolated from N1.5, OpenVLA-OFT, PI, and SmolVLA runtimes.
repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
environment_path="${1:-${repository_root}/.venv-gr00t-n1d7}"
python_version="${ART_EMBODIED_GR00T_N1D7_PYTHON:-3.12}"
gr00t_revision="376ba890cff8c9de64d71d982772a9c36185fdd7"
source_root="${repository_root}/.runtime-sources"
source_path="${source_root}/isaac-gr00t-${gr00t_revision}"
ffmpeg_prefix="${repository_root}/.runtime-libs/ffmpeg7"

command -v uv >/dev/null || {
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
}
command -v git-lfs >/dev/null || {
  echo "Git LFS is required: https://git-lfs.com/" >&2
  exit 1
}
for tool in cmake c++ make; do
  command -v "${tool}" >/dev/null || {
    echo "${tool} is required to build native dependencies. Install CMake and C++ build tools." >&2
    exit 1
  }
done

mkdir -p "${source_root}"
if [[ ! -d "${source_path}/.git" ]]; then
  git clone --no-checkout https://github.com/NVIDIA/Isaac-GR00T.git "${source_path}"
fi
git -C "${source_path}" lfs install --local
git -C "${source_path}" fetch origin "${gr00t_revision}"
git -C "${source_path}" checkout --detach "${gr00t_revision}"

uv venv --python "${python_version}" "${environment_path}"
UV_PROJECT_ENVIRONMENT="${environment_path}" \
  uv sync --project "${source_path}" --frozen --no-dev
python="${environment_path}/bin/python"

# ART 0.5.20 requires SciPy >=1.17, while this NVIDIA revision pins 1.15.3.
# Keep the qualified native runtime rather than overriding NVIDIA's dependency.
constraints="$(mktemp)"
diffusers_override="${repository_root}/constraints/diffusers-security.txt"
trap 'rm -f "${constraints}"' EXIT
printf '%s\n' 'openpipe-art==0.5.18' 'scipy==1.15.3' \
  'weave==0.52.37' 'wandb==0.24.2' > "${constraints}"
# Do not inherit checkout-wide overrides into NVIDIA's separate runtime.
uv pip install --no-config --python "${python}" --constraint "${constraints}" \
  --overrides "${diffusers_override}" \
  -e "${repository_root}[observability]"
uv pip install --no-config --python "${python}" \
  --overrides "${diffusers_override}" \
  "hf-libero==0.1.4" \
  "numpy==1.26.4" \
  "opencv-python==4.11.0.86" \
  "scipy==1.15.3" \
  "weave==0.52.37"

# Replace NVIDIA's older Diffusers pin without modifying its source or metadata.
# Other dependency overrides are not authorized by this file.
uv pip install --no-config --python "${python}" --overrides "${diffusers_override}" \
  "diffusers==0.38.0" "safetensors==0.8.0"
echo "Diffusers security override: NVIDIA declares 0.35.1, installed 0.38.0." >&2
echo "This intentional metadata mismatch is documented in README.md installation notes." >&2

# TorchCodec 0.8 dynamically links FFmpeg 4-7.  Headless training nodes do not
# necessarily provide those shared libraries even when a static ffmpeg binary
# is available through imageio-ffmpeg.
if [[ ! -e "${ffmpeg_prefix}/lib/libavcodec.so.61" ]]; then
  command -v micromamba >/dev/null || {
    echo "micromamba is required to install the pinned TorchCodec FFmpeg runtime." >&2
    exit 1
  }
  micromamba create --yes --prefix "${ffmpeg_prefix}" --channel conda-forge \
    "ffmpeg=7.1.1"
fi

LD_LIBRARY_PATH="${ffmpeg_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${python}" - <<'PY'
from importlib.metadata import version

import gr00t
import torch
import transformers
from torchcodec.decoders import VideoDecoder  # noqa: F401

print(f"gr00t={gr00t.__file__}")
for package in (
    "diffusers",
    "safetensors",
    "flash-attn",
    "gr00t",
    "hf-libero",
    "numpy",
    "peft",
    "torch",
    "torchvision",
    "transformers",
    "wandb",
    "weave",
):
    print(f"{package}={version(package)}")
assert version("diffusers") == "0.38.0", "Diffusers security override was not applied"
assert version("safetensors") == "0.8.0", "Untested Safetensors version"
print(f"cuda_available={torch.cuda.is_available()}")
print(f"transformers_runtime={transformers.__version__}")
print("torchcodec_ffmpeg_runtime=ok")
PY
