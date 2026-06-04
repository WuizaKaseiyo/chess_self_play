#!/bin/bash
# Sweep pass@8 eval over every teacher ckpt saved by job 8723.
#
# - Iterates outputs/44e1d0cf255c0621/checkpoints/global_step_*
# - For each ckpt:
#     1. Merges FSDP shards -> HF (verl.model_merger) if not already merged
#     2. Runs scripts.eval_chess_passk with k_max=8 on test.parquet
#     3. Skips ckpts that already produced their JSON
# - 4-GPU data-parallel via per-shard fan-out (mirrors sbatch_eval_chess_passk_dp4_smoke_gh200.slurm's approach)
# - Summarises and prints BEST teacher at end
#
# Required env (set by sbatch wrapper):
#   PROJECT_DIR   : repo root (default $HOME/chess/vam-chess)
#
# Optional:
#   ONLY_STEP=320  -> evaluate only that ckpt
#   K_MAX, BATCH_SIZE, MAX_*  -> mirror eval_chess_passk flags

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/chess/vam-chess}"
CKPT_BASE="${PROJECT_DIR}/outputs/44e1d0cf255c0621/checkpoints"
PARQUET="${PARQUET:-${PROJECT_DIR}/data/chess_puzzles_chessr1_aligned_sharded_baseline/test.parquet}"
RESULTS="${RESULTS:-${PROJECT_DIR}/eval_results/teacher_passk}"
WORK_ROOT="${WORK_ROOT:-${PROJECT_DIR}/eval_results/teacher_passk/_work}"

K_MAX="${K_MAX:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1536}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-3584}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DP_SHARDS="${DP_SHARDS:-4}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
SEED="${SEED:-0}"

mkdir -p "${RESULTS}" "${WORK_ROOT}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

echo "[sweep] PROJECT_DIR=${PROJECT_DIR}"
echo "[sweep] CKPT_BASE=${CKPT_BASE}"
echo "[sweep] PARQUET=${PARQUET}"
echo "[sweep] RESULTS=${RESULTS}"
echo "[sweep] K_MAX=${K_MAX}  DP_SHARDS=${DP_SHARDS}  TP=${TENSOR_PARALLEL_SIZE}"

# ---- Helpers -------------------------------------------------------------

ensure_hf_merged() {
    local ckpt_dir="$1"   # .../global_step_N
    local actor_dir="${ckpt_dir}/actor"
    local hf_dir="${actor_dir}/huggingface"

    if [ ! -d "${actor_dir}" ]; then
        echo "[merge] SKIP ${ckpt_dir} (no actor/ subdir)"
        return 1
    fi
    # Heuristic: merged HF dir has a weight file alongside config.json.
    if [ -f "${hf_dir}/config.json" ] && ( \
            ls "${hf_dir}"/*.safetensors >/dev/null 2>&1 \
            || ls "${hf_dir}"/pytorch_model*.bin >/dev/null 2>&1 \
            || ls "${hf_dir}"/model.safetensors* >/dev/null 2>&1 ); then
        echo "[merge] cached HF dir at ${hf_dir}"
        return 0
    fi
    if ! ls "${actor_dir}"/model_world_size_*_rank_*.pt >/dev/null 2>&1; then
        echo "[merge] SKIP ${ckpt_dir} (no FSDP shards; ckpt incomplete)"
        return 1
    fi

    echo "[merge] FSDP -> HF for ${actor_dir}"
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${actor_dir}" \
        --target_dir "${hf_dir}"
}

shard_parquet() {
    # Split parquet into DP_SHARDS shards under WORK_ROOT/shards/<step>/
    local step="$1"
    local shard_dir="${WORK_ROOT}/shards/step${step}"
    if [ -f "${shard_dir}/.ready" ]; then
        echo "[shard] cached at ${shard_dir}"
        echo "${shard_dir}"
        return 0
    fi
    mkdir -p "${shard_dir}"
    PARQUET="${PARQUET}" SHARD_DIR="${shard_dir}" NUM_SHARDS="${DP_SHARDS}" python - <<'PY'
import os
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

src = os.environ["PARQUET"]
shard_dir = Path(os.environ["SHARD_DIR"])
n = int(os.environ["NUM_SHARDS"])
shard_dir.mkdir(parents=True, exist_ok=True)
# Read with the columns eval needs.
table = ds.dataset(src, format="parquet").to_table(columns=["prompt", "reward_model", "extra_info"])
rows = table.to_pylist()
shards = [[] for _ in range(n)]
for i, row in enumerate(rows):
    shards[i % n].append(row)
for i, sh in enumerate(shards):
    out = shard_dir / f"shard_{i}.parquet"
    pq.write_table(pa.Table.from_pylist(sh), out)
    print(f"[shard] {i}: rows={len(sh)} out={out}", flush=True)
PY
    touch "${shard_dir}/.ready"
    echo "${shard_dir}"
}

run_eval_dp() {
    # Run k_max=K_MAX pass@k on the merged HF model. Each shard pinned to one GPU.
    local model_dir="$1"  # actor/huggingface dir
    local step="$2"
    local shard_dir="$3"
    local out_dir="${WORK_ROOT}/eval_step${step}"
    mkdir -p "${out_dir}"

    local pids=()
    for rank in $(seq 0 $((DP_SHARDS - 1))); do
        local shard_json="${out_dir}/result_shard_${rank}.json"
        if [ -f "${shard_json}" ]; then
            echo "[eval] step=${step} shard=${rank} cached"
            continue
        fi
        echo "[eval] step=${step} shard=${rank} -> GPU ${rank}"
        CUDA_VISIBLE_DEVICES="${rank}" CUDA_DEVICE_ORDER=PCI_BUS_ID \
            python -m scripts.eval_chess_passk \
                --model "${model_dir}" \
                --parquet "${shard_dir}/shard_${rank}.parquet" \
                --k_max "${K_MAX}" \
                --do_sample \
                --seed "${SEED}" \
                --seed_mode engine \
                --temperature "${TEMPERATURE}" \
                --top_p "${TOP_P}" \
                --batch_size "${BATCH_SIZE}" \
                --max_num_seqs "${MAX_NUM_SEQS}" \
                --max_prompt_length "${MAX_PROMPT_LENGTH}" \
                --max_response_length "${MAX_RESPONSE_LENGTH}" \
                --max_model_len "${MAX_MODEL_LEN}" \
                --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
                --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
                --acc_key acc \
                --out_json "${shard_json}" \
                > "${out_dir}/shard_${rank}.log" 2>&1 &
        pids+=($!)
    done

    if [ ${#pids[@]} -gt 0 ]; then
        local fail=0
        for pid in "${pids[@]}"; do
            if ! wait "${pid}"; then
                fail=1
            fi
        done
        if [ "${fail}" -ne 0 ]; then
            echo "[eval] step=${step} at least one shard FAILED — see ${out_dir}/shard_*.log"
            return 1
        fi
    fi

    # Reduce shards -> one JSON
    local final_json="${RESULTS}/passk_step${step}.json"
    OUT_DIR="${out_dir}" DP_SHARDS="${DP_SHARDS}" FINAL_JSON="${final_json}" \
        python - <<'PY'
import json, os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
n = int(os.environ["DP_SHARDS"])
final = Path(os.environ["FINAL_JSON"])

shards = []
for i in range(n):
    p = out_dir / f"result_shard_{i}.json"
    with p.open() as f:
        shards.append(json.load(f))

num_prompts = [int(s["num_prompts"]) for s in shards]
total = sum(num_prompts)
if total <= 0:
    raise RuntimeError("No prompts in shard outputs")
w = [n_i / total for n_i in num_prompts]

k = shards[0]["k"]
for s in shards[1:]:
    if s["k"] != k:
        raise RuntimeError("Shard k lists differ")

def wmean_list(key):
    vals = [s["curves"][key] for s in shards]
    return [float(sum(v[j] * w[i] for i, v in enumerate(vals))) for j in range(len(k))]

curves = {}
for key in shards[0]["curves"].keys():
    curves[key] = wmean_list(key)

summary = {}
for key in shards[0]["summary"].keys():
    v0 = shards[0]["summary"][key]
    if isinstance(v0, (int, float)):
        summary[key] = float(sum(float(s["summary"][key]) * w[i] for i, s in enumerate(shards)))
    else:
        summary[key] = v0  # acc_key etc.

# Convenience top-level fields the workflow doc references.
pass_at_1 = float(curves["acc_pass_mean"][0])
pass_at_k = float(curves["acc_pass_mean"][-1])

reduced = {
    "config": shards[0].get("config"),
    "num_prompts": total,
    "k": k,
    "curves": curves,
    "summary": summary,
    "pass@1": pass_at_1,
    f"pass@{k[-1]}": pass_at_k,
}
final.parent.mkdir(parents=True, exist_ok=True)
tmp = final.with_suffix(".json.tmp")
with tmp.open("w") as f:
    json.dump(reduced, f, indent=2)
tmp.replace(final)
print(f"[reduce] wrote {final}  pass@1={pass_at_1:.4f}  pass@{k[-1]}={pass_at_k:.4f}")
PY
}

# ---- Main loop -----------------------------------------------------------

if [ ! -d "${CKPT_BASE}" ]; then
    echo "[fatal] CKPT_BASE missing: ${CKPT_BASE}"
    exit 2
fi

if [ -n "${ONLY_STEP:-}" ]; then
    CKPTS=( "${CKPT_BASE}/global_step_${ONLY_STEP}" )
else
    # Sort numerically by step.
    mapfile -t CKPTS < <(ls -d "${CKPT_BASE}"/global_step_* 2>/dev/null \
        | awk -F'global_step_' '{print $2"\t"$0}' \
        | sort -n \
        | cut -f2)
fi

if [ ${#CKPTS[@]} -eq 0 ]; then
    echo "[fatal] no ckpts under ${CKPT_BASE}"
    exit 2
fi

echo "[sweep] ${#CKPTS[@]} ckpts to consider"

# Stage the parquet shards once (independent of ckpt).
SHARD_DIR_GLOBAL="$(shard_parquet shared)"

for ckpt in "${CKPTS[@]}"; do
    step="$(basename "${ckpt}" | sed 's/global_step_//')"
    final_json="${RESULTS}/passk_step${step}.json"
    if [ -f "${final_json}" ]; then
        echo "[skip] step=${step} already evaluated -> ${final_json}"
        continue
    fi
    if ! ensure_hf_merged "${ckpt}"; then
        echo "[skip] step=${step} (merge prerequisites unmet)"
        continue
    fi
    if ! run_eval_dp "${ckpt}/actor/huggingface" "${step}" "${SHARD_DIR_GLOBAL}"; then
        echo "[warn] step=${step} eval failed; continuing"
        continue
    fi
done

# ---- Summarise + pick best -----------------------------------------------
RESULTS="${RESULTS}" python - <<'PY'
import glob, json, os, re
from pathlib import Path

results_dir = Path(os.environ["RESULTS"])
rows = []
for p in sorted(glob.glob(str(results_dir / "passk_step*.json"))):
    m = re.search(r"passk_step(\d+)\.json$", p)
    if not m:
        continue
    step = int(m.group(1))
    with open(p) as f:
        d = json.load(f)
    p1 = float(d.get("pass@1", d["curves"]["acc_pass_mean"][0]))
    k_final = d["k"][-1]
    pk = float(d.get(f"pass@{k_final}", d["curves"]["acc_pass_mean"][-1]))
    rows.append((step, p1, pk, k_final))

if not rows:
    print("[summary] no eval JSONs found")
    raise SystemExit(0)

rows.sort()
k_used = rows[0][3]
header = f"{'step':>6}  {'pass@1':>8}  {'pass@'+str(k_used):>8}"
print(header)
print("-" * len(header))
for step, p1, pk, _ in rows:
    print(f"{step:>6}  {p1:>8.4f}  {pk:>8.4f}")

best = max(rows, key=lambda r: r[2])
print()
print(f"[BEST TEACHER] step={best[0]}  pass@1={best[1]:.4f}  pass@{best[3]}={best[2]:.4f}")

# Pin the choice for downstream Phase D.
best_step = best[0]
ckpt_path = Path(os.environ.get("PROJECT_DIR") or os.path.expanduser("~/chess/vam-chess")) \
    / "outputs/44e1d0cf255c0621/checkpoints" / f"global_step_{best_step}/actor/huggingface"
out = results_dir / "BEST_TEACHER"
out.write_text(str(ckpt_path) + "\n")
print(f"[BEST TEACHER] pinned -> {out}")
PY

echo "[sweep] done."
