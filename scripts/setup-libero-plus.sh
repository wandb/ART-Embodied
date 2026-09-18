#!/usr/bin/env bash
# Install the pinned official benchmark outside the ART-Embodied repository.

set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
storage_root="${ART_EMBODIED_STORAGE_ROOT:?ART_EMBODIED_STORAGE_ROOT is required}"
source_root="${ART_EMBODIED_LIBERO_PLUS_ROOT:-${storage_root}/external/LIBERO-plus}"
imagemagick_prefix="${ART_EMBODIED_LIBERO_PLUS_IMAGEMAGICK_PREFIX:-${storage_root}/external/libero-plus-imagemagick}"
download_root="${storage_root}/external/LIBERO-plus-assets"
source_revision=4976dc30028e805ff8094b55501d532c48fec182
dataset_revision=dd2bd61b7d9a6fef1abc52d606e983b41886a149
archive_sha256=96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf
prefix=inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets
python="${repository_root}/.venv/bin/python"
hf="${repository_root}/.venv/bin/hf"
imagemagick_version=7.1.2_30
imagemagick_build=imagemagick_h2c03864_0

[[ -x "${python}" && -x "${hf}" ]] || {
  echo "ART-Embodied virtual environment is not ready" >&2
  exit 1
}
command -v micromamba >/dev/null || {
  echo "micromamba is required to install the pinned native ImageMagick runtime" >&2
  exit 1
}
mkdir -p "$(dirname -- "${source_root}")" "${download_root}"

if [[ ! -x "${imagemagick_prefix}/bin/magick" ]]; then
  MAMBA_ROOT_PREFIX="${storage_root}/cache/micromamba" \
    micromamba create -y -p "${imagemagick_prefix}" -c conda-forge \
    "imagemagick=${imagemagick_version}=${imagemagick_build}"
fi
actual_imagemagick_version="$(${imagemagick_prefix}/bin/magick -version | sed -n 's/^Version: ImageMagick \([^ ]*\).*/\1/p')"
[[ "${actual_imagemagick_version}" == "7.1.2-30" ]] || {
  echo "Unexpected ImageMagick runtime version: ${actual_imagemagick_version}" >&2
  exit 1
}

if [[ ! -d "${source_root}/.git" ]]; then
  git clone https://github.com/sylvestf/LIBERO-plus.git "${source_root}"
fi
git -C "${source_root}" fetch origin "${source_revision}"
git -C "${source_root}" checkout --detach "${source_revision}"
actual_revision="$(git -C "${source_root}" rev-parse HEAD)"
[[ "${actual_revision}" == "${source_revision}" ]] || {
  echo "Unexpected LIBERO-Plus source revision: ${actual_revision}" >&2
  exit 1
}

"${hf}" download Sylvest/LIBERO-plus assets.zip \
  --revision "${dataset_revision}" \
  --local-dir "${download_root}"
archive="${download_root}/assets.zip"
echo "${archive_sha256}  ${archive}" | sha256sum --check --status || {
  echo "LIBERO-Plus asset archive checksum failed" >&2
  exit 1
}

assets="${source_root}/libero/libero/assets"
if [[ ! -f "${assets}/art_embodied_libero_plus_assets.json" ]]; then
  [[ ! -e "${assets}" ]] || {
    echo "Unverified LIBERO-Plus assets already exist: ${assets}" >&2
    exit 1
  }
  extract_root="$(mktemp -d "${source_root}/.asset-extract.XXXXXX")"
  unzip -q "${archive}" "${prefix}/*" -d "${extract_root}"
  mv "${extract_root}/${prefix}" "${assets}"
  cat >"${assets}/art_embodied_libero_plus_assets.json" <<EOF
{
  "asset_archive_sha256": "${archive_sha256}",
  "huggingface_dataset": "Sylvest/LIBERO-plus",
  "huggingface_revision": "${dataset_revision}"
}
EOF
fi

echo "LIBERO-Plus is ready at ${source_root}"
echo "export ART_EMBODIED_LIBERO_PLUS_ROOT=${source_root}"
echo "source ${repository_root}/scripts/libero-plus-runtime-env.sh"
