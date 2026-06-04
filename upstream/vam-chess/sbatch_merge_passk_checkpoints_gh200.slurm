#!/bin/bash
#SBATCH --job-name=chess-passk-merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm/slurm-%x-%j.out
#SBATCH --error=slurm/slurm-%x-%j.err
#SBATCH --partition=workq
#SBATCH --account=brics.a5l
#SBATCH --export=ALL

set -euo pipefail

CHESS_RL_PROJECTS_ROOT="${CHESS_RL_PROJECTS_ROOT:-/projects/a5l/ziyan/chess_rl}"
PROJECTS_SCRATCH_OK=0
if [ -d "/projects" ]; then
  if mkdir -p "${CHESS_RL_PROJECTS_ROOT}" 2>/dev/null && [ -w "${CHESS_RL_PROJECTS_ROOT}" ]; then
    PROJECTS_SCRATCH_OK=1
  fi
fi
if [ "${PROJECTS_SCRATCH_OK}" -eq 1 ]; then
  export CHESS_RL_SINGULARITY_TMPDIR="${CHESS_RL_SINGULARITY_TMPDIR:-${CHESS_RL_PROJECTS_ROOT%/}/singularity}"
  export CHESS_RL_TMPDIR="${CHESS_RL_TMPDIR:-${CHESS_RL_PROJECTS_ROOT%/}/tmp}"
  export CHESS_RL_APPTAINER_CACHEDIR="${CHESS_RL_APPTAINER_CACHEDIR:-${CHESS_RL_PROJECTS_ROOT%/}/apptainer_cache}"
  mkdir -p "${CHESS_RL_SINGULARITY_TMPDIR}" "${CHESS_RL_TMPDIR}" "${CHESS_RL_APPTAINER_CACHEDIR}" || true
fi
if [ -n "${CHESS_RL_TMPDIR:-}" ]; then
  export TMPDIR="${CHESS_RL_TMPDIR}"
fi
if [ -n "${CHESS_RL_SINGULARITY_TMPDIR:-}" ]; then
  export SINGULARITY_TMPDIR="${CHESS_RL_SINGULARITY_TMPDIR}"
  export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SINGULARITY_TMPDIR}}"
fi
if [ -n "${CHESS_RL_APPTAINER_CACHEDIR:-}" ]; then
  export APPTAINER_CACHEDIR="${CHESS_RL_APPTAINER_CACHEDIR%/}/apptainer_cache_${SLURM_JOB_ID:-$$}"
  export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${APPTAINER_CACHEDIR}}"
fi

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

echo "[SLURM] Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "[SLURM] Job name:    ${SLURM_JOB_NAME:-chess-passk-merge}"
echo "[SLURM] Node list:   ${SLURM_JOB_NODELIST:-N/A}"
echo "[SLURM] GPUs:        ${SLURM_GPUS_ON_NODE:-1} (requested via --gres=gpu:1)"
echo "[SLURM] Submit dir:  ${SLURM_SUBMIT_DIR:-$(pwd)}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  THIS_DIR="${SLURM_SUBMIT_DIR}"
else
  THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${THIS_DIR}"
echo "[SLURM] Repo dir:    ${THIS_DIR}"

HOST_ARCH="$(uname -m 2>/dev/null || echo unknown)"
echo "[SLURM] Host arch:   ${HOST_ARCH}"
if [ "${HOST_ARCH}" != "aarch64" ] && [ "${HOST_ARCH}" != "arm64" ]; then
  die "This script targets GH200 (arm64/aarch64) nodes, but uname reports '${HOST_ARCH}'."
fi

SIF_IMAGE="${CHESS_RL_SIF_IMAGE:-${HOME}/sif-images/chess_rl_v1-arm-stockfish-flashinfer.sif}"
CONTAINER_REPO_DIR="${CHESS_RL_CONTAINER_REPO_DIR:-/workspace/chess_rl}"
if [ ! -f "${SIF_IMAGE}" ]; then
  die "Missing SIF image: ${SIF_IMAGE}"
fi
echo "[SLURM] SIF_IMAGE=${SIF_IMAGE}"
echo "[SLURM] Repo mount: ${CONTAINER_REPO_DIR}"

PROJECT_ROOT="${CHESS_RL_PROJECT_ROOT:-/projects/a5l/ziyan}"
HF_CACHE_ROOT="${CHESS_RL_HF_CACHE_ROOT:-${PROJECT_ROOT%/}/hf_cache}"
mkdir -p "${HF_CACHE_ROOT}"
export HF_HOME="${HF_CACHE_ROOT%/}/hf_home"
export HUGGINGFACE_HUB_CACHE="${HF_HOME%/}/hub"
export TRANSFORMERS_CACHE="${HF_HOME%/}/transformers"
mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"
echo "[SLURM] HF_HOME=${HF_HOME}"

MERGE_OUT_ROOT="${MERGE_OUT_ROOT:-${PROJECT_ROOT%/}/chess_rl_outputs/passk_eval_models/checkpoint_sweep}"
mkdir -p "${MERGE_OUT_ROOT}"
export MERGE_OUT_ROOT
echo "[SLURM] MERGE_OUT_ROOT=${MERGE_OUT_ROOT}"

CMD="unset PYTHONHOME PYTHONPATH; cd \"${CONTAINER_REPO_DIR}\" && /usr/bin/python3 - <<'PY'
import json
import os
import subprocess
from pathlib import Path

def looks_like_hf_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if not (path / \"config.json\").exists():
        return False
    patterns = [\"model.safetensors\", \"model-*.safetensors\", \"pytorch_model*.bin\"]
    for pat in patterns:
        if list(path.glob(pat)):
            return True
    return False

runs = {
    \"82fpo6l0\": Path(\"/projects/a5l/ziyan/chess_rl_outputs/cr1_expectedscore_wdl_fulleval_20260128_114750/checkpoints\"),
    \"u2cuw56a\": Path(\"/projects/a5l/ziyan/chess_rl_outputs/cr1_expectedscore_wdl_fulleval_20260128_114800/checkpoints\"),
    \"f5guq4ti\": Path(\"/projects/a5l/ziyan/chess_rl_outputs/cr1_passk_qwen3b_full_20260210_202348/checkpoints\"),
}

merge_out_root = Path(os.environ[\"MERGE_OUT_ROOT\"])
merge_out_root.mkdir(parents=True, exist_ok=True)

records = []
for run_id, ckpt_root in runs.items():
    step_dirs = sorted(
        [p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith(\"global_step_\")],
        key=lambda p: int(p.name.split(\"_\")[-1]),
    )
    for step_dir in step_dirs:
        step_num = int(step_dir.name.split(\"_\")[-1])
        actor_dir = step_dir / \"actor\"
        target_dir = merge_out_root / run_id / f\"{step_dir.name}_merged_hf\"
        target_dir.parent.mkdir(parents=True, exist_ok=True)

        status = \"\"
        if looks_like_hf_model_dir(target_dir):
            status = \"already_merged\"
        else:
            legacy_dir = Path(f\"/projects/a5l/ziyan/chess_rl_outputs/passk_eval_models/{run_id}_step{step_num}_merged_hf\")
            if looks_like_hf_model_dir(legacy_dir):
                if target_dir.exists() and target_dir.is_symlink():
                    target_dir.unlink()
                elif target_dir.exists() and not target_dir.is_dir():
                    target_dir.unlink()
                if not target_dir.exists():
                    target_dir.symlink_to(legacy_dir, target_is_directory=True)
                status = \"reused_legacy_symlink\"
            else:
                cmd = [
                    \"python3\",
                    \"-m\",
                    \"verl.model_merger\",
                    \"merge\",
                    \"--backend\",
                    \"fsdp\",
                    \"--local_dir\",
                    str(actor_dir),
                    \"--target_dir\",
                    str(target_dir),
                    \"--trust-remote-code\",
                    \"--use_cpu_initialization\",
                ]
                print(f\"[merge] running: {' '.join(cmd)}\", flush=True)
                subprocess.check_call(cmd)
                status = \"merged_now\"

        if not looks_like_hf_model_dir(target_dir):
            raise RuntimeError(f\"Merged dir missing model weights: {target_dir}\")

        records.append(
            {
                \"run_id\": run_id,
                \"step\": step_dir.name,
                \"actor_dir\": str(actor_dir),
                \"merged_dir\": str(target_dir),
                \"status\": status,
            }
        )
        print(f\"[merge] ok run={run_id} step={step_dir.name} status={status} merged_dir={target_dir}\", flush=True)

manifest_json = merge_out_root / f\"merge_manifest_{os.environ.get('SLURM_JOB_ID', 'manual')}.json\"
manifest_tsv = merge_out_root / f\"merge_manifest_{os.environ.get('SLURM_JOB_ID', 'manual')}.tsv\"
with manifest_json.open(\"w\", encoding=\"utf-8\") as f:
    json.dump(records, f, indent=2)
with manifest_tsv.open(\"w\", encoding=\"utf-8\") as f:
    f.write(\"run_id\\tstep\\tstatus\\tmerged_dir\\n\")
    for r in records:
        f.write(f\"{r['run_id']}\\t{r['step']}\\t{r['status']}\\t{r['merged_dir']}\\n\")

print(f\"[merge] wrote_manifest_json={manifest_json}\", flush=True)
print(f\"[merge] wrote_manifest_tsv={manifest_tsv}\", flush=True)
print(f\"[merge] total_checkpoints={len(records)}\", flush=True)
PY"

echo "[SLURM] Container CMD: ${CMD}"

if command -v srun >/dev/null 2>&1; then
  srun --export=ALL \
    apptainer exec --nv \
      --bind "${THIS_DIR}:${CONTAINER_REPO_DIR}" \
      --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
      --bind "${HF_CACHE_ROOT}:${HF_CACHE_ROOT}" \
      "${SIF_IMAGE}" \
      bash -c "${CMD}"
else
  apptainer exec --nv \
    --bind "${THIS_DIR}:${CONTAINER_REPO_DIR}" \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --bind "${HF_CACHE_ROOT}:${HF_CACHE_ROOT}" \
    "${SIF_IMAGE}" \
    bash -c "${CMD}"
fi

echo "[SLURM] Merge job completed."
