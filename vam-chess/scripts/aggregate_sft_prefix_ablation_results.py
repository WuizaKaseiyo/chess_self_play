#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _maybe_get_passk(payload: dict[str, Any]) -> float | None:
    # scripts/eval_chess_passk.py always stores the pass@k mean under the summary key `k32_acc_mean`,
    # even when k_max != 32. We keep the key name for backwards compatibility.
    s = payload.get("summary", {})
    v = s.get("k32_acc_mean", None)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _maybe_get_pass1(payload: dict[str, Any]) -> float | None:
    s = payload.get("summary", {})
    v = s.get("k1_acc_mean", None)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _maybe_get_summary_float(payload: dict[str, Any], key: str) -> float | None:
    s = payload.get("summary", {})
    v = s.get(key, None)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _pick_eval_point(eval_points: list[dict[str, Any]], *, step: int) -> dict[str, Any] | None:
    for p in eval_points:
        try:
            if int(p.get("global_step")) == int(step):
                return p
        except Exception:
            continue
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate + plot chess SFT experiment results (prefix ablation + extensions).")
    p.add_argument(
        "--run_root",
        required=True,
        help="Output root containing per-run subdirs with results.json "
        "(e.g., /projects/.../sft_prefix_ablation_<id> or /projects/.../sft_sampled_move_<id>).",
    )
    p.add_argument(
        "--variants",
        default=None,
        help="Optional comma-separated list of subdir names to include (default: auto-discover all */results.json).",
    )
    p.add_argument(
        "--base_eval_json",
        default=None,
        help="Optional: base-model eval JSON produced by scripts/eval_chess_passk.py (for a baseline line).",
    )
    p.add_argument(
        "--out_csv",
        default="plots/sft_prefix_ablation_pass32.csv",
        help="Wide summary CSV (one row per variant; includes epoch1/epoch2).",
    )
    p.add_argument(
        "--out_points_csv",
        default="plots/sft_prefix_ablation_pass32_points.csv",
        help="Long CSV with one row per (variant, eval_point).",
    )
    p.add_argument("--out_png", default="plots/sft_prefix_ablation_pass32.png")
    p.add_argument("--out_pass1_png", default="plots/sft_prefix_ablation_pass1.png")
    p.add_argument(
        "--out_metrics_png",
        default="plots/sft_prefix_ablation_metrics.png",
        help="Optional: combined diversity/quality plot (valid count, unique moves, expected score sum).",
    )
    args = p.parse_args()

    run_root = Path(args.run_root)
    if args.variants:
        variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    else:
        variants = sorted([p.name for p in run_root.iterdir() if (p / "results.json").exists()])
    if not variants:
        raise FileNotFoundError(f"No results.json found under {run_root}")

    rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    for v in variants:
        path = run_root / v / "results.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        r = _load_json(path)
        mode = str(r.get("mode") or "offline").strip().lower()
        run_name = str(r.get("run_name") or v)
        offline_sampling = r.get("offline_sampling") or {}
        online_cfg = r.get("online") or {}
        metrics = r.get("metrics") or {}
        eval_points = metrics.get("eval_points") or []
        sft = r.get("sft") or {}
        epoch_steps = (sft.get("epoch_steps") or {}) if isinstance(sft, dict) else {}
        epoch1_step = int(epoch_steps.get("epoch1", 0) or 0)
        epoch2_step = int(epoch_steps.get("epoch2", 0) or 0)
        p_e1 = _pick_eval_point(eval_points, step=epoch1_step) if epoch1_step else None
        p_e2 = _pick_eval_point(eval_points, step=epoch2_step) if epoch2_step else None
        if isinstance(eval_points, list) and eval_points:
            for pnt in eval_points:
                point_rows.append(
                    {
                        "variant": v,
                        "run_name": run_name,
                        "mode": mode,
                        "epoch_frac": pnt.get("epoch_frac"),
                        "global_step": pnt.get("global_step"),
                        "pass@k_acc": pnt.get("pass@k_acc"),
                        "k1_acc": pnt.get("k1_acc"),
                        "k32_reward_mean": pnt.get("k32_reward_mean"),
                        "valid_count_mean": pnt.get("valid_count_mean"),
                        "unique_valid_moves_mean": pnt.get("unique_valid_moves_mean"),
                        "expected_score_sum_mean": pnt.get("expected_score_sum_mean"),
                        "out_dir": str((run_root / v).resolve()),
                    }
                )

        # Unified "final" metrics for mixed experiment types.
        final_passk = None
        final_pass1 = None
        final_valid = None
        final_unique = None
        final_expected_sum = None

        if mode == "online":
            final_passk = (r.get("metrics") or {}).get("k32_acc_mean")
            final_pass1 = (r.get("metrics") or {}).get("k1_acc_mean")
            final_valid = (r.get("metrics") or {}).get("k32_valid_count_mean")
            final_unique = (r.get("metrics") or {}).get("k32_unique_valid_moves_mean")
            final_expected_sum = (r.get("metrics") or {}).get("k32_expected_score_sum_mean")
        else:
            final_passk = ((r.get("metrics") or {}).get("epoch2") or {}).get("pass@k_acc")
            final_pass1 = (p_e2 or {}).get("k1_acc")
            final_valid = ((r.get("metrics") or {}).get("epoch2") or {}).get("valid_count_mean")
            final_unique = ((r.get("metrics") or {}).get("epoch2") or {}).get("unique_valid_moves_mean")
            final_expected_sum = ((r.get("metrics") or {}).get("epoch2") or {}).get("expected_score_sum_mean")

        rows.append(
            {
                "variant": v,
                "run_name": run_name,
                "mode": mode,
                "forced_prefix_template": r.get("forced_prefix_template"),
                "strip_phrase_template": r.get("strip_phrase_template"),
                "model_base": r.get("model_base") or r.get("model_base", None) or r.get("model", None),
                "final_pass@k_acc": final_passk,
                "final_pass@1_acc": final_pass1,
                "final_valid_count_mean": final_valid,
                "final_unique_valid_moves_mean": final_unique,
                "final_expected_score_sum_mean": final_expected_sum,
                # Offline-only (epoch endpoints + curves).
                "eval_k": (r.get("eval") or {}).get("k"),
                "epoch1_pass_acc": ((r.get("metrics") or {}).get("epoch1") or {}).get("pass@k_acc"),
                "epoch2_pass_acc": ((r.get("metrics") or {}).get("epoch2") or {}).get("pass@k_acc"),
                "epoch1_pass1_acc": (p_e1 or {}).get("k1_acc"),
                "epoch2_pass1_acc": (p_e2 or {}).get("k1_acc"),
                # Variant bookkeeping (if present).
                "sample_ordering": offline_sampling.get("sample_ordering") or online_cfg.get("sample_ordering"),
                "sft_weighting": offline_sampling.get("sft_weighting") or online_cfg.get("sft_weighting"),
                "num_move_samples": offline_sampling.get("num_move_samples"),
                "out_dir": str((run_root / v).resolve()),
            }
        )

    df = pd.DataFrame(rows).sort_values("variant")
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    if point_rows:
        df_points = pd.DataFrame(point_rows).sort_values(["variant", "epoch_frac", "global_step"])
        out_points_csv = Path(args.out_points_csv)
        out_points_csv.parent.mkdir(parents=True, exist_ok=True)
        df_points.to_csv(out_points_csv, index=False)
        print(f"Wrote {out_points_csv}")
    else:
        df_points = None

    base_acc = None
    base_label = None
    base_pass1 = None
    base_valid = None
    base_unique = None
    base_expected_sum = None
    if args.base_eval_json:
        base_payload = _load_json(Path(args.base_eval_json))
        base_acc = _maybe_get_passk(base_payload)
        base_pass1 = _maybe_get_pass1(base_payload)
        base_valid = _maybe_get_summary_float(base_payload, "k32_valid_count_mean")
        base_unique = _maybe_get_summary_float(base_payload, "k32_unique_valid_moves_mean")
        base_expected_sum = _maybe_get_summary_float(base_payload, "k32_expected_score_sum_mean")
        base_label = (base_payload.get("config") or {}).get("model", "base")

    # Plot pass@k.
    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(df)), 4.5))
    do_curves = df_points is not None and not df_points.empty and len(variants) <= 10
    if do_curves:
        x_min = float(df_points["epoch_frac"].min())
        x_max = float(df_points["epoch_frac"].max())
        for v in variants:
            sub = df_points[df_points["variant"] == v]
            if sub.empty:
                # Online runs don't have checkpoint curves; draw a horizontal line at the final value
                # so mixed offline/online experiment roots render cleanly.
                y = df[df["variant"] == v]["final_pass@k_acc"].iloc[0]
                ax.plot([x_min, x_max], [y, y], linestyle="--", linewidth=2, label=str(v))
            else:
                ax.plot(sub["epoch_frac"], sub["pass@k_acc"], marker="o", linewidth=2, label=str(v))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("pass@k exact-match (mean)")
        ax.set_title("Chess SFT (prefix-free eval; checkpoints)")
        ax.grid(True, axis="both", alpha=0.3)
    else:
        x = list(range(len(df)))
        ax.bar(x, df["final_pass@k_acc"], width=0.8, label="final")
        ax.set_xticks(x)
        ax.set_xticklabels(df["run_name"].tolist(), rotation=90)
        ax.set_ylabel("pass@k exact-match (mean)")
        ax.set_title("Chess SFT (prefix-free eval; final)")
        ax.grid(True, axis="y", alpha=0.3)

    if base_acc is not None:
        ax.axhline(base_acc, linestyle="--", linewidth=2, color="black", alpha=0.8, label=f"base ({base_label})")

    if do_curves:
        y_max = float(df_points["pass@k_acc"].max()) if not df_points.empty else 0.0
    else:
        y_max = float(df["final_pass@k_acc"].max()) if not df.empty else 0.0
    ax.set_ylim(0.0, max(0.05, y_max * 1.2))
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_png}")

    # Separate pass@1 plot.
    fig2, ax2 = plt.subplots(figsize=(max(10, 0.35 * len(df)), 4.5))
    if do_curves:
        for v in variants:
            sub = df_points[df_points["variant"] == v]
            if sub.empty:
                y = df[df["variant"] == v]["final_pass@1_acc"].iloc[0]
                ax2.plot([x_min, x_max], [y, y], linestyle="--", linewidth=2, label=str(v))
            else:
                ax2.plot(sub["epoch_frac"], sub["k1_acc"], marker="o", linewidth=2, label=str(v))
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("pass@1 exact-match (mean)")
        ax2.set_title("Chess SFT (prefix-free eval; pass@1)")
        ax2.grid(True, axis="both", alpha=0.3)
        y_max2 = float(df_points["k1_acc"].max())
        ax2.set_ylim(0.0, max(0.05, y_max2 * 1.2))
    else:
        x = list(range(len(df)))
        ax2.bar(x, df["final_pass@1_acc"], width=0.8, label="final")
        ax2.set_xticks(x)
        ax2.set_xticklabels(df["run_name"].tolist(), rotation=90)
        ax2.set_ylabel("pass@1 exact-match (mean)")
        ax2.set_title("Chess SFT (prefix-free eval; pass@1 final)")
        ax2.grid(True, axis="y", alpha=0.3)
        y_max2 = float(df["final_pass@1_acc"].max()) if not df.empty else 0.0
        ax2.set_ylim(0.0, max(0.05, y_max2 * 1.2))

    if base_pass1 is not None:
        ax2.axhline(base_pass1, linestyle="--", linewidth=2, color="black", alpha=0.8, label=f"base ({base_label})")
    ax2.legend(loc="upper left", frameon=False)
    fig2.tight_layout()

    out_pass1_png = Path(args.out_pass1_png)
    out_pass1_png.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(out_pass1_png, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_pass1_png}")

    # Extra diversity/quality plot.
    if args.out_metrics_png:
        fig3, axs = plt.subplots(1, 3, figsize=(max(12, 0.4 * len(df)), 4.0), sharex=False)
        metrics_specs = [
            ("valid_count_mean", "final_valid_count_mean", "valid_count_mean"),
            ("unique_valid_moves_mean", "final_unique_valid_moves_mean", "unique_valid_moves_mean"),
            ("expected_score_sum_mean", "final_expected_score_sum_mean", "expected_score_sum_mean"),
        ]
        for ax3, (title, col_final, col_points) in zip(axs, metrics_specs):
            if do_curves and df_points is not None:
                for v in variants:
                    sub = df_points[df_points["variant"] == v]
                    if sub.empty:
                        y = df[df["variant"] == v][col_final].iloc[0]
                        ax3.plot([x_min, x_max], [y, y], linestyle="--", linewidth=2, label=str(v))
                    else:
                        ax3.plot(sub["epoch_frac"], sub[col_points], marker="o", linewidth=2, label=str(v))
                ax3.set_xlabel("Epoch")
                ax3.set_title(title)
                ax3.grid(True, axis="both", alpha=0.3)
            else:
                x = list(range(len(df)))
                ax3.bar(x, df[col_final], width=0.8)
                ax3.set_xticks(x)
                ax3.set_xticklabels(df["run_name"].tolist(), rotation=90)
                ax3.set_title(title)
                ax3.grid(True, axis="y", alpha=0.3)

            # Optional baseline horizontal line (if base eval JSON contains the metric).
            baseline_map = {
                "valid_count_mean": base_valid,
                "unique_valid_moves_mean": base_unique,
                "expected_score_sum_mean": base_expected_sum,
            }
            base_y = baseline_map.get(title, None)
            if base_y is not None:
                ax3.axhline(
                    base_y,
                    linestyle="--",
                    linewidth=2,
                    color="black",
                    alpha=0.8,
                    label=(f"base ({base_label})" if title == "valid_count_mean" else None),
                )

        axs[0].set_ylabel("Mean (over prompts)")
        if do_curves:
            axs[0].legend(loc="upper left", frameon=False)
        fig3.tight_layout()

        out_metrics_png = Path(args.out_metrics_png)
        out_metrics_png.parent.mkdir(parents=True, exist_ok=True)
        fig3.savefig(out_metrics_png, dpi=200, bbox_inches="tight")
        print(f"Wrote {out_metrics_png}")


if __name__ == "__main__":
    main()
