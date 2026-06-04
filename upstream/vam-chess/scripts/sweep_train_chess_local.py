import argparse
import csv
import os
import re
import subprocess
import time
from pathlib import Path

from tqdm import tqdm


STEP_TIME_RE = re.compile(r"step:(\d+).*perf/time_per_step:([0-9.eE+\-]+)")
PROGRESS_RE = re.compile(r"Training Progress:.*?\|\s*(\d+)/(\d+)\s*\[([0-9:]+)<")


def _bool_to_str(value: bool) -> str:
    return "True" if value else "False"


def parse_step_times_s(log_text: str) -> list[tuple[int, float]]:
    # Prefer tqdm progress-bar elapsed times (robust in Ray where very-long console metric lines
    # may be dropped), then fall back to console metrics if present.
    by_completed: dict[int, float] = {}
    for m in PROGRESS_RE.finditer(log_text.replace("\r", "\n")):
        completed = int(m.group(1))
        if completed == 0:
            continue
        elapsed_str = m.group(3)
        parts = [int(p) for p in elapsed_str.split(":")]
        if len(parts) == 2:
            elapsed_s = parts[0] * 60 + parts[1]
        else:
            elapsed_s = parts[0] * 3600 + parts[1] * 60 + parts[2]
        by_completed[completed] = float(elapsed_s)

    if len(by_completed) >= 1:
        step_times: list[tuple[int, float]] = []
        prev_elapsed = 0.0
        for completed in sorted(by_completed.keys()):
            elapsed = by_completed[completed]
            step_times.append((completed, elapsed - prev_elapsed))
            prev_elapsed = elapsed
        return step_times

    out: list[tuple[int, float]] = []
    for m in STEP_TIME_RE.finditer(log_text):
        out.append((int(m.group(1)), float(m.group(2))))
    return out


def detect_oom(log_text: str) -> bool:
    t = log_text.lower()
    return (
        ("cuda out of memory" in t)
        or ("torch.cuda.outofmemoryerror" in t)
        or ("torch.outofmemoryerror: cuda out of memory" in t)
        or ("no available memory for the cache blocks" in t)
        or ("available kv cache memory" in t)
    )


def cleanup(repo_root: Path) -> None:
    # Kill any leftover Ray workers between runs (timeouts can leave GPU-resident workers alive).
    subprocess.run(
        ["ray", "stop", "--force"],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)


def cfg_to_vec(cfg: tuple[bool, bool, bool, int, int, int, int]) -> tuple[int, int, int, int, int]:
    (
        param_offload,
        optimizer_offload,
        _enforce_eager,
        _tensor_model_parallel_size,
        rollout_logprob_micro,
        ref_logprob_micro,
        ppo_micro,
    ) = cfg
    return (
        ppo_micro,
        rollout_logprob_micro,
        ref_logprob_micro,
        0 if param_offload else 1,
        0 if optimizer_offload else 1,
    )


def vec_leq(vec_a: tuple[int, ...], vec_b: tuple[int, ...]) -> bool:
    return all(a <= b for a, b in zip(vec_a, vec_b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/sweep_local", type=str)
    parser.add_argument("--timeout_min", default=30, type=int)
    parser.add_argument("--training_steps", default=2, type=int)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cleanup(repo_root)
    out_root = repo_root / args.output_dir
    logs_dir = out_root / "logs"
    runs_dir = out_root / "runs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_root / "results.csv"
    write_header = not results_csv.exists()

    micro_batch_sizes = [1, 2, 4, 8, 16]
    offload_bools = [False, True]
    eager_bools = [True, False]  # (enforce_eager=True, free_cache_engine=True) vs (False, False)
    tensor_model_parallel_sizes = [1, 2]

    all_cfgs: list[tuple[bool, bool, bool, int, int, int, int]] = []
    for param_offload in offload_bools:
        for optimizer_offload in offload_bools:
            for enforce_eager in eager_bools:
                for tensor_model_parallel_size in tensor_model_parallel_sizes:
                    for rollout_logprob_micro in micro_batch_sizes:
                        for ref_logprob_micro in micro_batch_sizes:
                            for ppo_micro in micro_batch_sizes:
                                all_cfgs.append(
                                    (
                                        param_offload,
                                        optimizer_offload,
                                        enforce_eager,
                                        tensor_model_parallel_size,
                                        rollout_logprob_micro,
                                        ref_logprob_micro,
                                        ppo_micro,
                                    )
                                )

    vecs = {cfg: cfg_to_vec(cfg) for cfg in all_cfgs}

    def order_key(cfg: tuple[bool, bool, bool, int, int, int, int]) -> tuple[int, ...]:
        po, oo, eager, tp, rlp, ref_lp, ppo = cfg
        max_micro = max(ppo, rlp, ref_lp)
        sum_micro = ppo + rlp + ref_lp
        return (
            max_micro,
            sum_micro,
            ppo,
            rlp,
            ref_lp,
            0 if po else 1,
            0 if oo else 1,
            0 if eager else 1,
            0 if tp == 2 else 1,
        )

    ordered_cfgs = sorted(all_cfgs, key=order_key)
    total_cfgs = len(all_cfgs)

    stats = {"tested": 0, "dominated": 0, "ok": 0, "oom": 0, "timeout": 0, "error": 0}

    with results_csv.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "param_offload",
                "optimizer_offload",
                "tensor_model_parallel_size",
                "rollout_log_prob_micro_batch_size_per_gpu",
                "ref_log_prob_micro_batch_size_per_gpu",
                "ppo_micro_batch_size_per_gpu",
                "enforce_eager",
                "free_cache_engine",
                "status",
                "exit_code",
                "step_times_s",
                "log_path",
                "verl_base_dir",
            ],
        )
        if write_header:
            writer.writeheader()

        with tqdm(total=total_cfgs, desc="sweep", unit="cfg") as pbar:
            oom_vecs_by_group: dict[tuple[int, bool], list[tuple[int, ...]]] = {}
            for cfg in ordered_cfgs:
                cfg_vec = vecs[cfg]
                po, oo, eager, tp, rlp, ref_lp, ppo = cfg
                group_key = (tp, eager)
                if any((vec_leq(v, cfg_vec) and v != cfg_vec) for v in oom_vecs_by_group.get(group_key, [])):
                    po, oo, eager, tp, rlp, ref_lp, ppo = cfg
                    writer.writerow(
                        {
                            "param_offload": int(po),
                            "optimizer_offload": int(oo),
                            "tensor_model_parallel_size": tp,
                            "rollout_log_prob_micro_batch_size_per_gpu": rlp,
                            "ref_log_prob_micro_batch_size_per_gpu": ref_lp,
                            "ppo_micro_batch_size_per_gpu": ppo,
                            "enforce_eager": int(eager),
                            "free_cache_engine": int(eager),
                            "status": "dominated",
                            "exit_code": "",
                            "step_times_s": "",
                            "log_path": "",
                            "verl_base_dir": "",
                        }
                    )
                    f.flush()
                    stats["dominated"] += 1
                    pbar.update(1)
                    pbar.set_postfix(**stats)
                    continue

                (
                    param_offload,
                    optimizer_offload,
                    enforce_eager,
                    tensor_model_parallel_size,
                    rollout_logprob_micro,
                    ref_logprob_micro,
                    ppo_micro,
                ) = cfg
                free_cache_engine = enforce_eager

                run_id = (
                    f"po{int(param_offload)}_oo{int(optimizer_offload)}"
                    f"_tp{tensor_model_parallel_size}"
                    f"_rlp{rollout_logprob_micro}_ref{ref_logprob_micro}_ppo{ppo_micro}"
                    f"_e{int(enforce_eager)}"
                )
                verl_base_dir = str(runs_dir / run_id)
                log_path = logs_dir / f"{run_id}.log"

                env = os.environ.copy()
                env["PARAM_OFFLOAD"] = _bool_to_str(param_offload)
                env["OPTIMIZER_OFFLOAD"] = _bool_to_str(optimizer_offload)
                env["PPO_MICRO_BATCH_SIZE"] = str(ppo_micro)
                env["ROLLOUT_LOGPROB_MICRO_BATCH_SIZE"] = str(rollout_logprob_micro)
                env["REF_LOGPROB_MICRO_BATCH_SIZE"] = str(ref_logprob_micro)
                env["TENSOR_MODEL_PARALLEL_SIZE"] = str(tensor_model_parallel_size)
                env["ENFORCE_EAGER"] = _bool_to_str(enforce_eager)
                env["FREE_CACHE_ENGINE"] = _bool_to_str(free_cache_engine)
                env["TOTAL_TRAINING_STEPS"] = str(args.training_steps)
                env["VERL_BASE_DIR"] = verl_base_dir

                cmd = [
                    "timeout",
                    f"{args.timeout_min}m",
                    "bash",
                    "train_chess.sh",
                ]
                tqdm.write(
                    f"RUN {run_id} | "
                    f"po={int(param_offload)} oo={int(optimizer_offload)} "
                    f"tp={tensor_model_parallel_size} "
                    f"rlp={rollout_logprob_micro} ref={ref_logprob_micro} ppo={ppo_micro} "
                    f"eager={int(enforce_eager)} free_cache={int(free_cache_engine)}"
                )
                with log_path.open("wb") as log_f:
                    completed = subprocess.run(
                        cmd,
                        cwd=repo_root,
                        env=env,
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                    )

                log_text = log_path.read_text(errors="replace")
                step_times = parse_step_times_s(log_text)

                status = "ok"
                if completed.returncode != 0:
                    status = "oom" if detect_oom(log_text) else ("timeout" if completed.returncode == 124 else "error")

                writer.writerow(
                    {
                        "param_offload": int(param_offload),
                        "optimizer_offload": int(optimizer_offload),
                        "tensor_model_parallel_size": tensor_model_parallel_size,
                        "rollout_log_prob_micro_batch_size_per_gpu": rollout_logprob_micro,
                        "ref_log_prob_micro_batch_size_per_gpu": ref_logprob_micro,
                        "ppo_micro_batch_size_per_gpu": ppo_micro,
                        "enforce_eager": int(enforce_eager),
                        "free_cache_engine": int(free_cache_engine),
                        "status": status,
                        "exit_code": completed.returncode,
                        "step_times_s": ",".join([f"{s}:{t}" for s, t in step_times]),
                        "log_path": str(log_path),
                        "verl_base_dir": verl_base_dir,
                    }
                )
                f.flush()

                stats["tested"] += 1
                stats[status] += 1
                pbar.update(1)
                pbar.set_postfix(**stats)

                if status == "oom":
                    oom_vecs_by_group.setdefault((tensor_model_parallel_size, enforce_eager), []).append(cfg_vec)
                cleanup(repo_root)


if __name__ == "__main__":
    main()
