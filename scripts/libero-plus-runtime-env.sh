#!/usr/bin/env bash
# Activate the native ImageMagick runtime required by official LIBERO-Plus.

storage_root="${ART_EMBODIED_STORAGE_ROOT:?ART_EMBODIED_STORAGE_ROOT is required}"
imagemagick_prefix="${ART_EMBODIED_LIBERO_PLUS_IMAGEMAGICK_PREFIX:-${storage_root}/external/libero-plus-imagemagick}"

if [[ ! -x "${imagemagick_prefix}/bin/magick" ]]; then
  echo "Missing LIBERO-Plus ImageMagick runtime: ${imagemagick_prefix}" >&2
  echo "Run scripts/setup-libero-plus.sh first." >&2
  return 1 2>/dev/null || exit 1
fi

export ART_EMBODIED_LIBERO_PLUS_IMAGEMAGICK_PREFIX="${imagemagick_prefix}"
export MAGICK_HOME="${imagemagick_prefix}"
export PATH="${imagemagick_prefix}/bin:${PATH}"
export LD_LIBRARY_PATH="${imagemagick_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
