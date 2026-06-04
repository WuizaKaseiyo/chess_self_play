#!/usr/bin/env python3
"""
Download a selected subset of W&B run files (instead of the full file tree).

Typical use for this investigation:
  - baseline run: rollout_logs/<step>.jsonl
  - iterative run: allowed_move_elim_rounds/<step>_round<r>.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb


def _build_wanted_names(mode: str, steps: list[int]) -> list[str]:
    if mode == "baseline":
        return [f"rollout_logs/{s}.jsonl" for s in steps]
    if mode == "iterative":
        return [f"allowed_move_elim_rounds/{s}_round{r}.jsonl" for s in steps for r in (1, 2, 3, 4)]
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mode", choices=["baseline", "iterative"], required=True)
    ap.add_argument("--steps-start", type=int, default=20)
    ap.add_argument("--steps-end", type=int, default=360)
    ap.add_argument("--steps-stride", type=int, default=20)
    args = ap.parse_args()

    steps = list(range(int(args.steps_start), int(args.steps_end) + 1, int(args.steps_stride)))
    if not steps:
        raise ValueError("No steps selected")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    run = api.run(f"{args.entity}/{args.project}/{args.run}")
    files = {f.name: f for f in run.files()}

    wanted = _build_wanted_names(args.mode, steps)
    # Always grab these small provenance files if present.
    always = ["wandb-metadata.json", "config_hash.json", "requirements.txt"]
    wanted_all = always + wanted

    missing = [name for name in wanted_all if name not in files]
    if missing:
        print(f"[WARN] missing {len(missing)} requested files (showing up to 20): {missing[:20]}")

    downloaded = 0
    for idx, name in enumerate(wanted_all, start=1):
        f = files.get(name)
        if f is None:
            continue
        local = outdir / name
        if local.exists() and local.stat().st_size > 0:
            continue
        f.download(root=str(outdir), replace=True)
        downloaded += 1
        if idx % 20 == 0:
            print(f"[INFO] processed {idx}/{len(wanted_all)}")

    print(f"[OK] run={args.run} mode={args.mode} steps={steps[0]}..{steps[-1]} stride={args.steps_stride}")
    print(f"[OK] requested={len(wanted_all)} downloaded_now={downloaded} outdir={outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

