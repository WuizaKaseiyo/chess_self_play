#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def _extract_baseline_passk(baseline_json: dict[str, Any]) -> float:
    # scripts/eval_chess_passk.py uses historical key names: "k32_acc_mean" is the final k.
    s = baseline_json.get("summary", {}) or {}
    v = s.get("k32_acc_mean", None)
    if v is None:
        raise KeyError("baseline summary missing 'k32_acc_mean'")
    return float(v)


def _extract_forced_passk(forced_json: dict[str, Any]) -> float:
    s = forced_json.get("summary", {}) or {}
    v = s.get("passk_acc_mean", None)
    if v is None:
        raise KeyError("forced-prefix summary missing 'passk_acc_mean'")
    return float(v)


def _plot_hist(
    *,
    df: pd.DataFrame,
    column: str,
    title: str,
    out_path: Path,
    bins: int,
    kde: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column], bins=bins, kde=kde, stat="density")
    mean = float(df[column].mean())
    median = float(df[column].median())
    plt.axvline(mean, color="red", linestyle="--", linewidth=1, label=f"mean={mean:.3f}")
    plt.axvline(median, color="black", linestyle=":", linewidth=1, label=f"median={median:.3f}")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_passk_bar(
    *,
    labels: list[str],
    values: list[float],
    title: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    sns.barplot(x=labels, y=values)
    plt.ylim(0.0, 1.0)
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom")
    plt.title(title)
    plt.ylabel("pass@8")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_passk_json", default="plots/passk_baseline_qwen2p5_7b_test_k8_seed0_pci3.json")
    parser.add_argument("--forced_guess_json", default="artifacts/passk_forcedprefix_guess_qwen2p5_7b_test1000_k8_seed0.json")
    parser.add_argument("--logprob_guess_jsonl", default="artifacts/offpolicy_logprob_guess_qwen2p5_7b_test1000.jsonl")
    parser.add_argument("--out_dir", default="plots")
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--kde", action="store_true", default=False)
    parser.add_argument("--out_md", default="artifacts/forced_prefix_offpolicy_summary.md")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = _read_json(str(args.baseline_passk_json))
    forced_guess = _read_json(str(args.forced_guess_json))

    baseline_pass8 = _extract_baseline_passk(baseline)
    guess_pass8 = _extract_forced_passk(forced_guess)

    df_guess = _read_jsonl(str(args.logprob_guess_jsonl))
    if "move_logprob_mean" not in df_guess.columns or "move_logprob_sum" not in df_guess.columns:
        raise KeyError("logprob JSONL must contain 'move_logprob_mean' and 'move_logprob_sum' columns")

    _plot_hist(
        df=df_guess,
        column="move_logprob_mean",
        title="Guess forced-prefix scheme: move token logprob (mean per token)",
        out_path=out_dir / "offpolicy_logprob_guess_move_logprob_mean_hist.png",
        bins=int(args.bins),
        kde=bool(args.kde),
    )
    _plot_hist(
        df=df_guess,
        column="move_logprob_sum",
        title="Guess forced-prefix scheme: move logprob (sum over tokens)",
        out_path=out_dir / "offpolicy_logprob_guess_move_logprob_sum_hist.png",
        bins=int(args.bins),
        kde=bool(args.kde),
    )

    _plot_passk_bar(
        labels=["baseline", "forced_guess"],
        values=[baseline_pass8, guess_pass8],
        title="pass@8: baseline vs forced <guess> prefix (GT forced)",
        out_path=out_dir / "passk_k8_baseline_vs_forcedprefix_guess.png",
    )

    guess_mean = float(np.asarray(df_guess["move_logprob_mean"], dtype=np.float64).mean())
    guess_median = float(np.asarray(df_guess["move_logprob_mean"], dtype=np.float64).median())

    # Write a short reproducibility note with updated commands (no legacy --scheme flags).
    md_path = Path(args.out_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_md = md_path.with_suffix(md_path.suffix + ".tmp")
    with open(tmp_md, "w", encoding="utf-8") as f:
        f.write("# Forced-prefix off-policy summary (guess scheme)\n\n")
        f.write("## Commands (exact)\n\n")
        f.write("### Baseline pass@8 (no forced prefix)\n")
        f.write("```bash\n")
        f.write(
            "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false \\\n"
            "  conda run -n verl python -m scripts.eval_chess_passk \\\n"
            "    --model Qwen/Qwen2.5-7B-Instruct \\\n"
            "    --parquet data/chess_puzzles/test.parquet \\\n"
            "    --k_max 8 --do_sample --seed 0 --seed_mode engine \\\n"
            "    --temperature 0.6 --top_p 0.95 \\\n"
            "    --batch_size 32 --max_num_seqs 1024 \\\n"
            "    --max_prompt_length 1024 --max_response_length 2000 --max_model_len 4096 \\\n"
            "    --gpu_memory_utilization 0.8 --tensor_parallel_size 1 \\\n"
            "    --out_json plots/passk_baseline_qwen2p5_7b_test_k8_seed0_pci3.json\n"
        )
        f.write("```\n\n")

        f.write("### Move logprobs (teacher forcing)\n")
        f.write("```bash\n")
        f.write(
            "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false \\\n"
            "  conda run -n verl python -m scripts.analyze_chess_forced_prefix_logprobs \\\n"
            "    --model Qwen/Qwen2.5-7B-Instruct \\\n"
            "    --parquet data/chess_puzzles/test.parquet \\\n"
            "    --batch_size 8 \\\n"
            "    --out_jsonl artifacts/offpolicy_logprob_guess_qwen2p5_7b_test1000.jsonl \\\n"
            "    --out_stats_json artifacts/offpolicy_logprob_guess_qwen2p5_7b_test1000_stats.json\n"
        )
        f.write("```\n\n")

        f.write("### Pass@8 with forced prefix containing the ground-truth move\n")
        f.write("```bash\n")
        f.write(
            "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false \\\n"
            "  conda run -n verl python -m scripts.eval_chess_passk_forced_prefix \\\n"
            "    --model Qwen/Qwen2.5-7B-Instruct \\\n"
            "    --parquet data/chess_puzzles/test.parquet \\\n"
            "    --k 8 --seed 0 --seed_mode engine \\\n"
            "    --temperature 0.6 --top_p 0.95 \\\n"
            "    --batch_size 32 --max_num_seqs 1024 \\\n"
            "    --max_prompt_length 1024 --max_response_length 2000 --max_model_len 4096 \\\n"
            "    --gpu_memory_utilization 0.8 --tensor_parallel_size 1 \\\n"
            "    --out_json artifacts/passk_forcedprefix_guess_qwen2p5_7b_test1000_k8_seed0.json \\\n"
            "    --out_jsonl_gz artifacts/passk_forcedprefix_guess_qwen2p5_7b_test1000_k8_seed0.jsonl.gz\n"
        )
        f.write("```\n\n")

        f.write("## Summary\n")
        f.write(f"- Baseline pass@8: {baseline_pass8:.6f} (`{args.baseline_passk_json}`)\n")
        f.write(f"- Forced-guess pass@8: {guess_pass8:.6f} (`{args.forced_guess_json}`)\n")
        f.write(f"- Guess move_logprob_mean: mean={guess_mean:.6f}, median={guess_median:.6f} (`{args.logprob_guess_jsonl}`)\n\n")
        f.write("## Plots\n")
        f.write(f"- `{out_dir / 'offpolicy_logprob_guess_move_logprob_mean_hist.png'}`\n")
        f.write(f"- `{out_dir / 'offpolicy_logprob_guess_move_logprob_sum_hist.png'}`\n")
        f.write(f"- `{out_dir / 'passk_k8_baseline_vs_forcedprefix_guess.png'}`\n")

    tmp_md.replace(md_path)
    print(f"Wrote {md_path}")
    print("pass@8:", json.dumps({"baseline": baseline_pass8, "forced_guess": guess_pass8}, indent=2))
    print("move_logprob_mean:", json.dumps({"guess_mean": guess_mean, "guess_median": guess_median}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

