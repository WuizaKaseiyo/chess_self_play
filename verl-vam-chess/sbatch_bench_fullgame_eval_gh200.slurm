#!/bin/bash
#SBATCH --job-name=chess-rl-bench-gh200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=144
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/slurm-%x-%j.out
#SBATCH --error=slurm/slurm-%x-%j.err
#SBATCH --partition=workq
#SBATCH --account=brics.a5l
#SBATCH --export=ALL

# GH200 standalone benchmark launcher (no training):
# - Runs `scripts/bench_fullgame_eval_gh200.py` inside the ARM64 container image
#   `gabr1e1/chess_rl:v1-arm-stockfish-flashinfer` via Apptainer/Singularity.
# - Benchmarks baseline vs optimized full-game evaluation speed, and checks ACPL alignment.
#
# Usage:
#   # One-liner (submits the job, waits, prints summary):
#   bash ./sbatch_bench_fullgame_eval_gh200.slurm
#
#   # Or submit directly:
#   sbatch --export=ALL,CHESS_RL_BENCH_MODEL=Qwen/Qwen2.5-7B-Instruct ./sbatch_bench_fullgame_eval_gh200.slurm
#
# Common overrides:
#   # Use a local path (e.g., merged HF dir or repo-mounted checkpoint):
#   sbatch --export=ALL,CHESS_RL_BENCH_MODEL=/workspace/chess_rl/outputs/.../merged_hf_model ./sbatch_bench_fullgame_eval_gh200.slurm
#
#   # Change evaluation sizing:
#   sbatch --export=ALL,CHESS_RL_BENCH_ROUNDS=5,CHESS_RL_BENCH_GAMES_PER_ROUND=50 ./sbatch_bench_fullgame_eval_gh200.slurm
#
#   # Change baseline sizing (default is 1 round × 50 games = 50 games per depth):
#   sbatch --export=ALL,CHESS_RL_BENCH_BASELINE_ROUNDS=1,CHESS_RL_BENCH_BASELINE_GAMES_PER_ROUND=50 ./sbatch_bench_fullgame_eval_gh200.slurm
#
#   # ACPL parallelism:
#   sbatch --export=ALL,CHESS_RL_BENCH_OPT_ACPL_WORKERS=72,CHESS_RL_BENCH_ACPL_THREADS=2 ./sbatch_bench_fullgame_eval_gh200.slurm
#
#   # Isolate ACPL-only changes (baseline also batched inference):
#   sbatch --export=ALL,CHESS_RL_BENCH_BASELINE_BATCHED_INFERENCE=True ./sbatch_bench_fullgame_eval_gh200.slurm

set -euo pipefail

# Preferred shared scratch (persistent, non-tmpfs on Isambard):
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
  export CHESS_RL_WANDB_DIR="${CHESS_RL_WANDB_DIR:-${CHESS_RL_PROJECTS_ROOT%/}/wandb}"
  mkdir -p "${CHESS_RL_SINGULARITY_TMPDIR}" "${CHESS_RL_TMPDIR}" "${CHESS_RL_APPTAINER_CACHEDIR}" "${CHESS_RL_WANDB_DIR}" || true
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
if [ -n "${CHESS_RL_WANDB_DIR:-}" ]; then
  export WANDB_DIR="${CHESS_RL_WANDB_DIR}"
fi

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

# Always run relative to the repo root so relative paths work.
# Note: under Slurm, the script may be executed from a spool directory, so prefer SLURM_SUBMIT_DIR
# when running *inside* a job.
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  THIS_DIR="${SLURM_SUBMIT_DIR}"
else
  THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${THIS_DIR}"

# Convenience wrapper: allow running via `bash sbatch_bench_fullgame_eval_gh200.slurm`.
if [ -z "${SLURM_JOB_ID:-}" ]; then
  if ! command -v sbatch >/dev/null 2>&1; then
    die "sbatch not found on PATH. Run this on a Slurm login node, or submit manually with: sbatch ${THIS_DIR}/sbatch_bench_fullgame_eval_gh200.slurm"
  fi

  mkdir -p slurm

  # Default to the same model default as train_chess.sh if none provided.
  export CHESS_RL_BENCH_MODEL="${CHESS_RL_BENCH_MODEL:-${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}}"
  export CHESS_RL_BENCH_TRUST_REMOTE_CODE="${CHESS_RL_BENCH_TRUST_REMOTE_CODE:-True}"

  echo "[SUBMIT] Submitting GH200 benchmark job..."
  echo "[SUBMIT]   CHESS_RL_BENCH_MODEL=${CHESS_RL_BENCH_MODEL}"
  set +e
  job_id_raw="$(sbatch --wait --export=ALL --parsable "${THIS_DIR}/sbatch_bench_fullgame_eval_gh200.slurm")"
  sbatch_rc=$?
  set -e
  # Slurm federation can return "jobid;cluster". Arrays can return "jobid_task".
  job_id="${job_id_raw%%;*}"
  echo "[SUBMIT]   job_id=${job_id}"
  if [ "${sbatch_rc}" -ne 0 ]; then
    echo "[WARN] sbatch returned non-zero exit code ${sbatch_rc}. Job may have failed; attempting to show logs/results anyway." >&2
  fi

  out_log="${THIS_DIR}/slurm/slurm-chess-rl-bench-gh200-${job_id}.out"
  err_log="${THIS_DIR}/slurm/slurm-chess-rl-bench-gh200-${job_id}.err"
  echo "[SUBMIT]   out_log=${out_log}"
  echo "[SUBMIT]   err_log=${err_log}"

  # Locate the results JSON written by the benchmark script.
  CONTAINER_REPO_DIR="${CHESS_RL_CONTAINER_REPO_DIR:-/workspace/chess_rl}"
  OUT_ROOT_CONTAINER="${CHESS_RL_BENCH_OUT_ROOT:-${CONTAINER_REPO_DIR}/outputs/gh200_fullgame_bench}"
  if [[ "${OUT_ROOT_CONTAINER}" == "${CONTAINER_REPO_DIR}"* ]]; then
    OUT_ROOT_HOST="${THIS_DIR}${OUT_ROOT_CONTAINER#${CONTAINER_REPO_DIR}}"
  elif [[ "${OUT_ROOT_CONTAINER}" != /* ]]; then
    OUT_ROOT_HOST="${THIS_DIR}/${OUT_ROOT_CONTAINER}"
  else
    OUT_ROOT_HOST="${OUT_ROOT_CONTAINER}"
  fi
  results_json="${OUT_ROOT_HOST}/bench_results_${job_id}.json"

  echo "[DONE] Job ${job_id} finished."
  echo "[DONE] Results JSON: ${results_json}"

  # Best-effort wait for NFS flush / delayed writes.
  for _ in $(seq 1 12); do
    if [ -f "${results_json}" ]; then
      break
    fi
    sleep 5
  done

  if [ ! -f "${results_json}" ]; then
    echo "[DONE] Results JSON not found; check logs:" >&2
    echo "  ${out_log}" >&2
    echo "  ${err_log}" >&2
    exit 1
  fi

  export CHESS_RL_HOST_REPO_DIR="${THIS_DIR}"
  export CHESS_RL_CONTAINER_REPO_DIR="${CONTAINER_REPO_DIR}"
  RESULTS_JSON="${results_json}" python3 - <<'PY'
import json
import os
from pathlib import Path

host_repo = str(os.environ.get("CHESS_RL_HOST_REPO_DIR") or "")
container_repo = str(os.environ.get("CHESS_RL_CONTAINER_REPO_DIR") or "")

path = Path(os.environ["RESULTS_JSON"])
obj = json.loads(path.read_text(encoding="utf-8"))


def map_path(p: object) -> str:
    if not isinstance(p, str):
        return str(p)
    if container_repo and host_repo and p.startswith(container_repo):
        return host_repo + p[len(container_repo) :]
    return p


def fmt_float(x):
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


print("[RESULT] run_id=", obj.get("run_id"))
print("[RESULT] model_requested=", obj.get("model", {}).get("requested"))
print("[RESULT] model_resolved_for_vllm=", obj.get("model", {}).get("resolved_for_vllm"))

for mode in ("baseline", "batched_serial", "optimized"):
    if mode not in obj:
        continue
    wall = obj[mode].get("wall_s")
    paths = obj[mode].get("paths", {}) or {}
    print(
        f"[RESULT] {mode}.wall_s={fmt_float(wall)} "
        f"summary_json={map_path(paths.get('summary_json'))} "
        f"games_jsonl={map_path(paths.get('games_jsonl'))}"
    )
    by_depth = obj[mode].get("summary_by_depth", {}) or {}
    for depth_key, row in sorted(by_depth.items()):
        print(
            f"[RESULT] {mode}.{depth_key} games={row.get('num_games')} "
            f"W/L/D={row.get('wins')}/{row.get('losses')}/{row.get('draws')} "
            f"acpl_mean={fmt_float(row.get('acpl_mean'))} "
            f"acpl_mean_per_move={fmt_float(row.get('acpl_mean_per_move'))} "
            f"moves={row.get('acpl_moves')}"
        )

print("[RESULT] speedup_total_est=", fmt_float(obj.get("speedup_total_est")))
if "speedup_total" in obj or "speedup" in obj:
    print("[RESULT] speedup_total=", fmt_float(obj.get("speedup_total", obj.get("speedup"))))
if "speedup_infer" in obj:
    print("[RESULT] speedup_infer=", fmt_float(obj.get("speedup_infer")))
if "speedup_infer_est_full" in obj:
    print("[RESULT] speedup_infer_est_full=", fmt_float(obj.get("speedup_infer_est_full")))
if "speedup_acpl_est" in obj:
    print("[RESULT] speedup_acpl_est=", fmt_float(obj.get("speedup_acpl_est")))
if "acpl_parallel_time_s" in obj:
    print("[RESULT] acpl_parallel_time_s=", fmt_float(obj.get("acpl_parallel_time_s")))
if "infer_s_per_pos_baseline" in obj:
    print("[RESULT] infer_s_per_pos_baseline=", fmt_float(obj.get("infer_s_per_pos_baseline")))
if "infer_s_per_pos_optimized" in obj:
    print("[RESULT] infer_s_per_pos_optimized=", fmt_float(obj.get("infer_s_per_pos_optimized")))
if "infer_serial_est_s_for_optimized" in obj:
    print("[RESULT] infer_serial_est_s_for_optimized=", fmt_float(obj.get("infer_serial_est_s_for_optimized")))
if "other_time_s_optimized" in obj:
    print("[RESULT] other_time_s_optimized=", fmt_float(obj.get("other_time_s_optimized")))
if "acpl_serial_full_est_s_for_optimized" in obj:
    print("[RESULT] acpl_serial_full_est_s_for_optimized=", fmt_float(obj.get("acpl_serial_full_est_s_for_optimized")))
if "baseline_total_est_s_for_optimized" in obj:
    print("[RESULT] baseline_total_est_s_for_optimized=", fmt_float(obj.get("baseline_total_est_s_for_optimized")))

al = obj.get("acpl_alignment_serial_sample_vs_parallel", {}) or {}
if al and al.get("enabled"):
    d = al.get("delta_serial_minus_parallel") or {}
    ad = al.get("abs_delta") or {}
    print(
        "[RESULT] acpl_alignment_serial_sample_vs_parallel "
        f"games={al.get('sample_games')} plies={al.get('sample_plies')} serial_wall_s={fmt_float(al.get('serial_wall_s'))} "
        f"delta_mean={fmt_float(d.get('mean'))} abs_delta_mean={fmt_float(ad.get('mean'))}"
    )

pgn1 = obj.get("pgn_alignment_baseline_vs_batched_serial", {}) or {}
if pgn1:
    keys = ("shared_games", "pgn_mismatch_count", "pgn_match_frac")
    parts = " ".join(f"{k}={pgn1.get(k)}" for k in keys if k in pgn1)
    print("[RESULT] pgn_alignment_baseline_vs_batched_serial", parts)
pgn2 = obj.get("pgn_alignment_batched_serial_vs_optimized", {}) or {}
if pgn2:
    keys = ("shared_games", "pgn_mismatch_count", "pgn_match_frac")
    parts = " ".join(f"{k}={pgn2.get(k)}" for k in keys if k in pgn2)
    print("[RESULT] pgn_alignment_batched_serial_vs_optimized", parts)

print("[RESULT] full_results_json=", str(path))
PY

  if [ -f "${out_log}" ]; then
    echo "[EVIDENCE] First infer lines:"
    grep -m 5 "\\[fullgame\\]\\[infer\\]" "${out_log}" 2>/dev/null || true
    echo "[EVIDENCE] First ACPL lines:"
    grep -m 5 "\\[fullgame\\]\\[acpl\\]" "${out_log}" 2>/dev/null || true
  fi

  exit 0
fi

echo "[SLURM] Job ID:     ${SLURM_JOB_ID:-N/A}"
echo "[SLURM] Node list:  ${SLURM_JOB_NODELIST:-N/A}"
echo "[SLURM] GPUs:       ${SLURM_GPUS_ON_NODE:-1} (requested via --gres=gpu:1)"
echo "[SLURM] CPUs:       ${SLURM_CPUS_PER_TASK:-144} (requested via --cpus-per-task)"
echo "[SLURM] Repo dir:   ${THIS_DIR}"

# GH200 nodes are arm64/aarch64.
HOST_ARCH="$(uname -m 2>/dev/null || echo unknown)"
echo "[SLURM] Host arch:  ${HOST_ARCH}"
if [ "${HOST_ARCH}" != "aarch64" ] && [ "${HOST_ARCH}" != "arm64" ]; then
  die "This script targets GH200 (arm64/aarch64) nodes, but uname reports '${HOST_ARCH}'."
fi

mkdir -p slurm

SINGULARITY_BIN="${SINGULARITY_BIN:-}"
if [ -z "${SINGULARITY_BIN}" ]; then
  if command -v apptainer >/dev/null 2>&1; then
    SINGULARITY_BIN="apptainer"
  elif command -v singularity >/dev/null 2>&1; then
    SINGULARITY_BIN="singularity"
  fi
fi
if [ -z "${SINGULARITY_BIN}" ]; then
  die "Could not find a container runtime on PATH (expected 'apptainer' or 'singularity')."
fi
echo "[SLURM] Container runtime: ${SINGULARITY_BIN} ($(${SINGULARITY_BIN} --version 2>/dev/null || echo 'version-unknown'))"

# Container image.
SIF_URI="${CHESS_RL_IMAGE_URI:-docker://gabr1e1/chess_rl:v1-arm-stockfish-flashinfer}"
SIF_IMAGE="${CHESS_RL_SIF_IMAGE:-${HOME}/sif-images/chess_rl_v1-arm-stockfish-flashinfer.sif}"
CONTAINER_REPO_DIR="${CHESS_RL_CONTAINER_REPO_DIR:-/workspace/chess_rl}"
mkdir -p "$(dirname "${SIF_IMAGE}")"
echo "[SLURM] Image URI:  ${SIF_URI}"
echo "[SLURM] Image SIF:  ${SIF_IMAGE}"
echo "[SLURM] Repo mount: ${CONTAINER_REPO_DIR}"

# Apptainer/Singularity tmp directory:
# - Rootless OCI->SIF builds often require xattrs; prefer node-local scratch.
CHESS_RL_SINGULARITY_TMPDIR="${CHESS_RL_SINGULARITY_TMPDIR:-}"
if [ -n "${CHESS_RL_SINGULARITY_TMPDIR}" ]; then
  DEFAULT_SINGULARITY_TMP="${CHESS_RL_SINGULARITY_TMPDIR}"
elif [ -n "${SLURM_TMPDIR:-}" ]; then
  DEFAULT_SINGULARITY_TMP="${SLURM_TMPDIR}/singularity"
else
  DEFAULT_SINGULARITY_TMP="/tmp/${USER}/singularity"
fi
export SINGULARITY_TMPDIR="${DEFAULT_SINGULARITY_TMP}"
export APPTAINER_TMPDIR="${SINGULARITY_TMPDIR}"
mkdir -p "${SINGULARITY_TMPDIR}" || true
echo "[SLURM] SINGULARITY_TMPDIR=${SINGULARITY_TMPDIR}"

if [ ! -f "${SIF_IMAGE}" ]; then
  echo "[SLURM] Building SIF: ${SIF_IMAGE}"
  "${SINGULARITY_BIN}" build "${SIF_IMAGE}" "${SIF_URI}"
fi

# Required benchmark args.
CHESS_RL_BENCH_MODEL="${CHESS_RL_BENCH_MODEL:-${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}}"
if [ -z "${CHESS_RL_BENCH_MODEL}" ]; then
  die "CHESS_RL_BENCH_MODEL is required (e.g., export CHESS_RL_BENCH_MODEL=Qwen/Qwen2.5-7B-Instruct)."
fi

# Benchmark knobs (env -> CLI).
CHESS_RL_BENCH_OUT_ROOT="${CHESS_RL_BENCH_OUT_ROOT:-${CONTAINER_REPO_DIR}/outputs/gh200_fullgame_bench}"
CHESS_RL_BENCH_ROUNDS="${CHESS_RL_BENCH_ROUNDS:-5}"
CHESS_RL_BENCH_GAMES_PER_ROUND="${CHESS_RL_BENCH_GAMES_PER_ROUND:-50}"
CHESS_RL_BENCH_BASELINE_ROUNDS="${CHESS_RL_BENCH_BASELINE_ROUNDS:-1}"
CHESS_RL_BENCH_BASELINE_GAMES_PER_ROUND="${CHESS_RL_BENCH_BASELINE_GAMES_PER_ROUND:-50}"

# Align with training-style DP/FSDP setups (no tensor parallel by default).
CHESS_RL_BENCH_TP_SIZE="${CHESS_RL_BENCH_TP_SIZE:-1}"
CHESS_RL_BENCH_GPU_MEM_UTIL="${CHESS_RL_BENCH_GPU_MEM_UTIL:-0.90}"
CHESS_RL_BENCH_MAX_MODEL_LEN="${CHESS_RL_BENCH_MAX_MODEL_LEN:-5120}"
CHESS_RL_BENCH_MAX_NUM_SEQS="${CHESS_RL_BENCH_MAX_NUM_SEQS:-2048}"
CHESS_RL_BENCH_TRUST_REMOTE_CODE="${CHESS_RL_BENCH_TRUST_REMOTE_CODE:-True}"

CHESS_RL_BENCH_OPT_ACPL_WORKERS="${CHESS_RL_BENCH_OPT_ACPL_WORKERS:-72}"
# Baseline runs skip ACPL by default to stay within the 1-day QoS walltime limit; we do a
# sampled serial ACPL recompute inside the benchmark script for alignment instead.
CHESS_RL_BENCH_BASE_ACPL_WORKERS="${CHESS_RL_BENCH_BASE_ACPL_WORKERS:-0}"
CHESS_RL_BENCH_ACPL_THREADS="${CHESS_RL_BENCH_ACPL_THREADS:-2}"
CHESS_RL_BENCH_BASELINE_BATCHED_INFERENCE="${CHESS_RL_BENCH_BASELINE_BATCHED_INFERENCE:-False}"

CHESS_RL_BENCH_STOCKFISH_PATH="${CHESS_RL_BENCH_STOCKFISH_PATH:-/usr/local/bin/stockfish}"

export PYTHONUNBUFFERED=1
# Avoid picking up user-site packages via Apptainer's default $HOME bind.
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

TRUST_REMOTE_CODE_FLAG=""
case "${CHESS_RL_BENCH_TRUST_REMOTE_CODE}" in
  1|true|True|TRUE|yes|YES|y|Y) TRUST_REMOTE_CODE_FLAG="--trust-remote-code" ;;
esac
BASELINE_BATCHED_FLAG=""
case "${CHESS_RL_BENCH_BASELINE_BATCHED_INFERENCE}" in
  1|true|True|TRUE|yes|YES|y|Y) BASELINE_BATCHED_FLAG="--baseline-batched-inference" ;;
esac

HF_BIND_ARGS=()
if [ -d "${HOME}/.cache/huggingface" ]; then
  HF_BIND_ARGS=(--bind "${HOME}/.cache/huggingface:/root/.cache/huggingface")
fi

echo "[SLURM] Launching benchmark inside container..."
"${SINGULARITY_BIN}" exec --nv \
  --bind "${THIS_DIR}:${CONTAINER_REPO_DIR}" \
  "${HF_BIND_ARGS[@]}" \
  "${SIF_IMAGE}" \
  bash -lc "
    python3 -c \"import flashinfer; print('[bench] flashinfer', flashinfer.__version__)\" \
    && \
    python3 ${CONTAINER_REPO_DIR}/scripts/bench_fullgame_eval_gh200.py \
      --model '${CHESS_RL_BENCH_MODEL}' \
      --out-root '${CHESS_RL_BENCH_OUT_ROOT}' \
      --run-id '${SLURM_JOB_ID}' \
      --rounds ${CHESS_RL_BENCH_ROUNDS} \
      --games-per-round ${CHESS_RL_BENCH_GAMES_PER_ROUND} \
      --baseline-rounds ${CHESS_RL_BENCH_BASELINE_ROUNDS} \
      --baseline-games-per-round ${CHESS_RL_BENCH_BASELINE_GAMES_PER_ROUND} \
      --tensor-parallel-size ${CHESS_RL_BENCH_TP_SIZE} \
      --gpu-memory-utilization ${CHESS_RL_BENCH_GPU_MEM_UTIL} \
      --max-model-len ${CHESS_RL_BENCH_MAX_MODEL_LEN} \
      --max-num-seqs ${CHESS_RL_BENCH_MAX_NUM_SEQS} \
      --stockfish-path '${CHESS_RL_BENCH_STOCKFISH_PATH}' \
      --baseline-acpl-workers ${CHESS_RL_BENCH_BASE_ACPL_WORKERS} \
      --optimized-acpl-workers ${CHESS_RL_BENCH_OPT_ACPL_WORKERS} \
      --acpl-threads ${CHESS_RL_BENCH_ACPL_THREADS} \
      ${TRUST_REMOTE_CODE_FLAG} \
      ${BASELINE_BATCHED_FLAG}
  "
