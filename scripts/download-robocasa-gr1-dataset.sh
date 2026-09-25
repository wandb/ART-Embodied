#!/usr/bin/env bash
set -euo pipefail

repository_root="${ART_EMBODIED_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${repository_root}" ]]; then
  repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fi
if [[ -f "${repository_root}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${repository_root}/.env"
  set +a
fi

dataset_id="nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim"
dataset_revision="09c6de8af50168090e7e9cc01e1ec3bce788de24"
storage_root="${ART_EMBODIED_STORAGE_ROOT:-${repository_root}}"
dataset_root="${ART_EMBODIED_ROBOCASA_DATASET_ROOT:-${storage_root}/datasets/${dataset_id}/${dataset_revision}}"
hf_cli="${repository_root}/.venv-gr00t-n1d7/bin/hf"
max_workers="${ART_EMBODIED_HF_DOWNLOAD_WORKERS:-4}"
max_attempts="${ART_EMBODIED_HF_DOWNLOAD_ATTEMPTS:-12}"
expected_file_count=48193

dataset_license_url="https://creativecommons.org/licenses/by-nc/4.0/"
dataset_license_url="https://creativecommons.org/licenses/by-nc/4.0/"
cat >&2 <<EOF
NOTICE: ${dataset_id} is a third-party NVIDIA dataset licensed under CC BY-NC 4.0.
It is not included in ART-Embodied and is not covered by ART-Embodied's Apache-2.0 license.
Review and comply with NVIDIA's dataset terms before downloading or using it.
License: ${dataset_license_url}
EOF

[[ -x "${hf_cli}" ]] || { echo "Missing Hugging Face CLI: ${hf_cli}" >&2; exit 1; }
[[ "${max_workers}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid worker count: ${max_workers}" >&2; exit 1; }
[[ "${max_attempts}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid attempt count: ${max_attempts}" >&2; exit 1; }

mkdir -p "${dataset_root}"
rm -f "${dataset_root}/download.status"

verify_snapshot() {
  local actual_file_count parquet_count video_count task_count incomplete_count
  actual_file_count="$(
    find "${dataset_root}" -type f \
      ! -path "${dataset_root}/.cache/*" \
      ! -name download.status \
      -printf . | wc -c
  )"
  parquet_count="$(find "${dataset_root}/LeRobot" -type f -name '*.parquet' -printf . 2>/dev/null | wc -c)"
  video_count="$(find "${dataset_root}/LeRobot" -type f -name '*.mp4' -printf . 2>/dev/null | wc -c)"
  task_count="$(find "${dataset_root}/LeRobot" -mindepth 1 -maxdepth 1 -type d -printf . 2>/dev/null | wc -c)"
  incomplete_count="$(find "${dataset_root}" -type f -name '*.incomplete' -printf . | wc -c)"

  echo "RoboCasa snapshot audit: files=${actual_file_count}/${expected_file_count} tasks=${task_count}/24 parquet=${parquet_count}/24000 videos=${video_count}/24000 incomplete=${incomplete_count}"
  [[ "${actual_file_count}" == "${expected_file_count}" ]] &&
    [[ "${task_count}" == 24 ]] &&
    [[ "${parquet_count}" == 24000 ]] &&
    [[ "${video_count}" == 24000 ]] &&
    [[ "${incomplete_count}" == 0 ]]
}

# Hub snapshots are content-addressed and resumable. A retry scans the revision,
# reuses completed files, and resumes only missing or partial objects. Xet's
# adaptive controller handles transfer-stream backpressure; the outer workers
# keep enough independent small files in flight to avoid per-file API latency.
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_CLIENT_RETRY_MAX_ATTEMPTS="${HF_XET_CLIENT_RETRY_MAX_ATTEMPTS:-8}"
attempt=1
while true; do
  echo "RoboCasa dataset download attempt ${attempt}/${max_attempts} (workers=${max_workers}, xet_high_performance=${HF_XET_HIGH_PERFORMANCE})"
  download_succeeded=false
  if "${hf_cli}" download "${dataset_id}" \
    --repo-type dataset \
    --revision "${dataset_revision}" \
    --include 'LeRobot/**' README.md \
    --local-dir "${dataset_root}" \
    --max-workers "${max_workers}"; then
    download_succeeded=true
  fi
  # huggingface_hub can return the existing local directory after a transient
  # Hub failure. Treat the command result as advisory and the pinned snapshot
  # contents as the actual completion contract.
  if [[ "${download_succeeded}" == true ]] && verify_snapshot; then
    printf 'complete\n' >"${dataset_root}/download.status"
    echo "RoboCasa dataset download complete: ${dataset_root}"
    break
  fi

  if (( attempt >= max_attempts )); then
    echo "RoboCasa dataset download failed after ${max_attempts} resumable attempts" >&2
    exit 1
  fi

  # 2, 4, 8, then at most 15 minutes. This clears transient Hub/Xet 429s
  # without turning a permanent authentication error into an unbounded job.
  delay=$((120 * (1 << (attempt - 1))))
  (( delay > 900 )) && delay=900
  echo "Download attempt failed; preserving partial files and retrying in ${delay}s" >&2
  sleep "${delay}"
  ((attempt += 1))
done
