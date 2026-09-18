#!/usr/bin/env bash
#SBATCH --job-name=gr00t-n17-robocasa-sft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=48:00:00
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

storage_root="${ART_EMBODIED_STORAGE_ROOT:-${repository_root}}"
python="${repository_root}/.venv-gr00t-n1d7/bin/python"
source_root="${repository_root}/.runtime-sources/isaac-gr00t-376ba890cff8c9de64d71d982772a9c36185fdd7"
dataset_root="${ART_EMBODIED_ROBOCASA_DATASET_ROOT:-${storage_root}/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim/09c6de8af50168090e7e9cc01e1ec3bce788de24}"
manifest="${repository_root}/examples/embodied/robocasa/gr1_tabletop_tasks.yaml"
output_root="${ART_EMBODIED_ROBOCASA_SFT_OUTPUT_ROOT:-${repository_root}/outputs/gr00t-n1d7-robocasa-sft}"
prepared_root="${ART_EMBODIED_ROBOCASA_PREPARED_ROOT:-${storage_root}/prepared-datasets/gr00t-n1d7-robocasa-gr1-tabletop-v1}"
max_steps="${ART_EMBODIED_SFT_MAX_STEPS:-60000}"
global_batch_size="${ART_EMBODIED_SFT_GLOBAL_BATCH_SIZE:-512}"
experiment_name="${ART_EMBODIED_SFT_EXPERIMENT_NAME:-gr00t-n1d7-robocasa-gr1-tabletop-sft-u${max_steps}}"
run_dir="${output_root}/${experiment_name}"
ffmpeg_library_dir="${repository_root}/.runtime-libs/ffmpeg7/lib"

[[ -x "${python}" ]] || { echo "Missing ${python}" >&2; exit 1; }
[[ -f "${dataset_root}/download.status" ]] || {
  echo "RoboCasa dataset download is incomplete: ${dataset_root}" >&2
  exit 1
}
[[ "$(<"${dataset_root}/download.status")" == "complete" ]] || {
  echo "RoboCasa dataset status is not complete" >&2
  exit 1
}
[[ -f "${manifest}" ]] || { echo "Missing ${manifest}" >&2; exit 1; }
[[ -f "${source_root}/gr00t/experiment/launch_finetune.py" ]] || {
  echo "Missing pinned Isaac-GR00T source" >&2
  exit 1
}
[[ -e "${ffmpeg_library_dir}/libavcodec.so.61" ]] || {
  echo "Missing pinned FFmpeg 7 runtime" >&2
  exit 1
}
[[ -n "${HF_TOKEN:-}" ]] || {
  echo "HF_TOKEN is required to resolve the pinned GR00T checkpoint" >&2
  exit 1
}
[[ -n "${WANDB_API_KEY:-}" ]] || {
  echo "WANDB_API_KEY is required for the evidence-bearing SFT run" >&2
  exit 1
}
[[ ! -e "${run_dir}" ]] || {
  echo "Refusing to mix SFT runs in existing directory: ${run_dir}" >&2
  exit 1
}

mkdir -p "${output_root}" "${repository_root}/outputs/slurm"
dataset_report="${output_root}/dataset-conformance.json"
PYTHONPATH="${repository_root}/src:${repository_root}${PYTHONPATH:+:${PYTHONPATH}}" \
"${python}" -m examples.embodied.robocasa.verify_dataset \
  --root "${dataset_root}" \
  --manifest "${manifest}" \
  --output "${dataset_report}"

prepared_report="${output_root}/prepared-dataset-conformance.json"
PYTHONPATH="${repository_root}/src:${repository_root}${PYTHONPATH:+:${PYTHONPATH}}" \
"${python}" -m examples.embodied.robocasa.prepare_n1d7_dataset \
  --root "${dataset_root}" \
  --manifest "${manifest}" \
  --output-root "${prepared_root}" \
  --report "${prepared_report}"

dataset_list="${output_root}/prepared-dataset-paths.txt"
"${python}" - "${manifest}" "${prepared_root}" "${dataset_list}" <<'PY'
from pathlib import Path
import sys

import yaml

manifest = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
paths = []
for task in manifest["tasks"]:
    path = root / f"gr1_unified.{task['id']}"
    for relative in ("meta/info.json", "meta/modality.json", "meta/stats.json"):
        if not (path / relative).is_file():
            raise SystemExit(f"Incomplete RoboCasa dataset: {path / relative}")
    paths.append(str(path.resolve()))
if len(paths) != 24:
    raise SystemExit(f"Expected 24 RoboCasa datasets, found {len(paths)}")
Path(sys.argv[3]).write_text("\n".join(paths) + "\n", encoding="utf-8")
PY

# Generate N1.7-compatible normalization caches once, before torchrun starts.
# Letting every rank generate them concurrently multiplies I/O and can expose
# partially populated legacy metadata to the mixture statistics merger.
while IFS= read -r dataset_path; do
  PYTHONPATH="${source_root}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python}" - "${dataset_path}" <<'PY'
from pathlib import Path
import sys

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.stats import main

main(Path(sys.argv[1]), EmbodimentTag.ROBOCASA_GR1_TABLETOP)
PY
done < "${dataset_list}"
dataset_paths="$(${python} - "${dataset_list}" <<'PY'
from pathlib import Path
import os
import sys

paths = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
print(os.pathsep.join(paths))
PY
)"

export LD_LIBRARY_PATH="${ffmpeg_library_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${repository_root}:${source_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export WANDB_ENTITY="${WANDB_ENTITY:-wandb-japan}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
wandb_project="${ART_EMBODIED_WANDB_PROJECT:-art-embodied-gr00t-n1d7-robocasa-sft}"
wandb_run_id="${ART_EMBODIED_SFT_WANDB_RUN_ID:-gr00t-n17-robocasa-sft-${SLURM_JOB_ID}}"
export WANDB_RUN_ID="${wandb_run_id}"
export WANDB_RESUME=never
wandb_run_url="https://wandb.ai/${WANDB_ENTITY}/${wandb_project}/runs/${wandb_run_id}"

base_model="$(${python} - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download(
    repo_id="nvidia/GR00T-N1.7-3B",
    revision="2fc962b973bccdd5d8ce4f67cc63b264d6886495",
))
PY
)"

cd "${source_root}"
master_port="$((20000 + SLURM_JOB_ID % 20000))"
"${repository_root}/.venv-gr00t-n1d7/bin/torchrun" \
  --nproc_per_node=8 \
  --master_port="${master_port}" \
  gr00t/experiment/launch_finetune.py \
  --base-model-path "${base_model}" \
  --dataset-path "${dataset_paths}" \
  --embodiment-tag ROBOCASA_GR1_TABLETOP \
  --num-gpus 8 \
  --output-dir "${output_root}" \
  --experiment-name "${experiment_name}" \
  --max-steps "${max_steps}" \
  --global-batch-size "${global_batch_size}" \
  --gradient-accumulation-steps 1 \
  --learning-rate 1e-4 \
  --warmup-ratio 0.05 \
  --weight-decay 1e-5 \
  --dataloader-num-workers 4 \
  --episode-sampling-rate 0.1 \
  --state-dropout-prob 0.2 \
  --save-steps 2000 \
  --save-total-limit 2 \
  --save-only-model \
  --use-wandb \
  --wandb-project "${wandb_project}"

checkpoint="${run_dir}/checkpoint-${max_steps}"
compgen -G "${checkpoint}/model*.safetensors" >/dev/null || {
  echo "SFT checkpoint is missing model tensors: ${checkpoint}" >&2
  exit 1
}
[[ -f "${checkpoint}/processor_config.json" ]] || {
  echo "SFT checkpoint is missing processor config: ${checkpoint}" >&2
  exit 1
}
"${python}" - \
  "${checkpoint}" \
  "${dataset_root}" \
  "${base_model}" \
  "${max_steps}" \
  "${WANDB_ENTITY}" \
  "${wandb_project}" \
  "${wandb_run_id}" \
  "${wandb_run_url}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

checkpoint = Path(sys.argv[1])
dataset_root = Path(sys.argv[2])
base_model = Path(sys.argv[3])
model_files = sorted(path.name for path in checkpoint.glob("model*.safetensors"))
marker = {
    "schema_version": 1,
    "status": "complete",
    "training": "official_gr00t_n1d7_supervised_finetune",
    "optimizer_steps": int(sys.argv[4]),
    "dataset_root": str(dataset_root.resolve()),
    "dataset_revision": "09c6de8af50168090e7e9cc01e1ec3bce788de24",
    "base_model": str(base_model.resolve()),
    "model_files": model_files,
    "processor_config_sha256": hashlib.sha256(
        (checkpoint / "processor_config.json").read_bytes()
    ).hexdigest(),
    "wandb": {
        "entity": sys.argv[5],
        "project": sys.argv[6],
        "run_id": sys.argv[7],
        "run_url": sys.argv[8],
    },
}
(checkpoint / "art_embodied_sft_complete.json").write_text(
    json.dumps(marker, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

printf 'RoboCasa 24-task SFT complete: %s\n' "${checkpoint}"
