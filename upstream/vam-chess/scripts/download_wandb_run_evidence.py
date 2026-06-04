#!/usr/bin/env python3
"""
Download W&B run evidence (config, metrics history, logs/files, artifacts) into a
local folder for reproducible analysis.

This script is intentionally self-contained so investigations can be repeated
without relying on the W&B UI.

Usage:
  python3 scripts/download_wandb_run_evidence.py \
    --entity gabr1e11 \
    --project chess_rl \
    --run px0sxz3v \
    --outdir analysis/wandb_evidence/px0sxz3v
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import wandb
import yaml


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _yaml_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=True, allow_unicode=True), encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_for_parquet(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--run", required=True, help="W&B run id (e.g., px0sxz3v)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--download-files",
        action="store_true",
        help="Download all run files (rollout logs, output.log, config.yaml, etc.)",
    )
    ap.add_argument(
        "--download-artifacts",
        action="store_true",
        help="Download all logged artifacts (may be large).",
    )
    ap.add_argument(
        "--history-page-size",
        type=int,
        default=10000,
        help="W&B scan_history page size.",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    run_path = f"{args.entity}/{args.project}/{args.run}"
    run = api.run(run_path)

    meta = {
        "downloaded_at": _now_iso(),
        "run_path": run_path,
        "run_id": run.id,
        "run_name": run.name,
        "run_state": run.state,
        "created_at": getattr(run, "created_at", None),
        "url": getattr(run, "url", None),
        "wandb_library": getattr(wandb, "__version__", None),
    }
    _json_dump(outdir / "run_meta.json", meta)

    # API config is *not* necessarily the same as a Hydra/OMEGACONF config.yaml file
    # saved by the training script; keep both.
    config = dict(run.config or {})
    _json_dump(outdir / "config_api.json", config)
    _yaml_dump(outdir / "config_api.yaml", config)

    # Summary: wandb summary is a custom object; `_json_dict` is the canonical export.
    summary = {}
    try:
        summary = dict(run.summary._json_dict)  # noqa: SLF001
    except Exception:
        summary = {k: run.summary[k] for k in run.summary.keys()}
    _json_dump(outdir / "summary.json", summary)

    # History (full): store raw JSONL, then a best-effort columnar version.
    history_jsonl = outdir / "history.jsonl"
    n_rows = 0
    with history_jsonl.open("w", encoding="utf-8") as f:
        for row in run.scan_history(page_size=args.history_page_size):
            n_rows += 1
            f.write(json.dumps(row, default=str) + "\n")

    _json_dump(outdir / "history_meta.json", {"rows": n_rows, "downloaded_at": _now_iso()})

    try:
        df = pd.read_json(history_jsonl, lines=True)
        for col in list(df.columns):
            if df[col].dtype == "object":
                df[col] = df[col].map(_sanitize_for_parquet)
        df.to_parquet(outdir / "history.parquet", index=False)
        df.to_csv(outdir / "history.csv.gz", index=False, compression="gzip")
    except Exception as e:
        _json_dump(outdir / "history_tabular_error.json", {"error": str(e)})

    # Manifest of files logged by the run.
    files = list(run.files())
    files_manifest = {
        "downloaded_at": _now_iso(),
        "count": len(files),
        "files": [{"name": f.name, "size": getattr(f, "size", None), "md5": getattr(f, "md5", None)} for f in files],
    }
    _json_dump(outdir / "files_manifest.json", files_manifest)

    if args.download_files:
        files_root = outdir / "files"
        files_root.mkdir(parents=True, exist_ok=True)
        for wf in files:
            wf.download(root=str(files_root), replace=True)
        _json_dump(outdir / "files_downloaded.json", {"downloaded_at": _now_iso(), "root": str(files_root)})

    # Logged artifacts: optional (often large).
    if args.download_artifacts:
        artifacts_root = outdir / "artifacts"
        artifacts_root.mkdir(parents=True, exist_ok=True)
        artifacts = []
        try:
            artifacts = list(run.logged_artifacts())
        except Exception:
            artifacts = []
        _json_dump(
            outdir / "artifacts_manifest.json",
            {
                "downloaded_at": _now_iso(),
                "count": len(artifacts),
                "artifacts": [
                    {
                        "name": a.name,
                        "type": a.type,
                        "version": getattr(a, "version", None),
                        "aliases": list(getattr(a, "aliases", []) or []),
                        "size": getattr(a, "size", None),
                        "digest": getattr(a, "digest", None),
                    }
                    for a in artifacts
                ],
            },
        )
        for a in artifacts:
            # Use a stable directory name and avoid characters that are annoying in shells.
            safe_name = a.name.replace("/", "__")
            dst = artifacts_root / safe_name
            dst.mkdir(parents=True, exist_ok=True)
            # Artifact.download() does not support `replace=` in all wandb versions.
            a.download(root=str(dst))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
