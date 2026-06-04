#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _model_label(payload: dict[str, Any]) -> str:
    cfg = payload.get("config", {})
    model = cfg.get("model", "unknown")
    k1_acc = payload.get("summary", {}).get("k1_acc_mean")
    k1_reward = payload.get("summary", {}).get("k1_reward_mean")
    if k1_acc is None or k1_reward is None:
        return model
    return f"{model} (k=1 acc={100*k1_acc:.1f}%, r={k1_reward:.3f})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_paths",
        nargs="+",
        help="One or more JSON outputs from scripts/eval_chess_passk.py",
    )
    parser.add_argument("--out_dir", default="plots")
    parser.add_argument("--out_prefix", default="passk")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads = [_load_json(p) for p in args.json_paths]
    payloads = sorted(payloads, key=lambda d: d.get("config", {}).get("model", ""))

    fig, (ax_r, ax_a) = plt.subplots(1, 2, figsize=(14, 5))

    for payload in payloads:
        k = payload["k"]
        curves = payload["curves"]
        label = _model_label(payload)

        ax_r.plot(k, curves["reward_best_mean"], linewidth=2, label=label)
        ax_a.plot(k, [100.0 * x for x in curves["acc_pass_mean"]], linewidth=2, label=label)

    ax_r.set_title("Best-of-k Reward vs k")
    ax_r.set_xlabel("k")
    ax_r.set_ylabel("mean(best reward among k)")
    ax_r.grid(True, alpha=0.3)
    ax_r.set_xlim(min(payloads[0]["k"]), max(payloads[0]["k"]))

    ax_a.set_title("Pass@k Accuracy vs k")
    ax_a.set_xlabel("k")
    ax_a.set_ylabel("pass@k accuracy (%)")
    ax_a.grid(True, alpha=0.3)
    ax_a.set_xlim(min(payloads[0]["k"]), max(payloads[0]["k"]))

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=1, frameon=False)
    fig.tight_layout()

    out_path = out_dir / f"{args.out_prefix}_reward_acc_curves.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

