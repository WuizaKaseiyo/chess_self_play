#!/usr/bin/env python3
"""
Export a W&B run into a reproducible local folder.

Writes:
  - run_info.json
  - config.json (raw run.config dict)
  - summary.json (raw run.summary._json_dict)
  - files_manifest.json (name/size/md5 for run files)
  - artifacts_manifest.json (logged/used artifacts metadata)
  - history.jsonl (raw time series via run.scan_history)
  - downloads under: files/ and artifacts/

Example:
  python3 scripts/export_wandb_run.py \
    --entity gabr1e11 --project chess_rl --run ymyvoypx \
    --out outputs/wandb/ymyvoypx
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import wandb


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _json_dump(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _iter_scan_history(run: Any, page_size: int) -> Iterable[dict[str, Any]]:
    # scan_history returns an iterator yielding dicts
    # page_size helps limit API request sizes
    for row in run.scan_history(page_size=page_size):
        if not isinstance(row, dict):
            continue
        yield row


def _download_run_file(*, run: Any, file_name: str, expected_size: int | None, dst_root: Path) -> Path:
    local_path = dst_root / file_name
    if local_path.exists() and local_path.is_file():
        if expected_size is None or local_path.stat().st_size == int(expected_size):
            return local_path

    _ensure_dir(local_path.parent)
    run.file(file_name).download(root=str(dst_root), replace=True)
    return local_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run", required=True, help="Run id, e.g. ymyvoypx")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--history-page-size", type=int, default=10000)
    parser.add_argument("--download-files", action="store_true", default=True)
    parser.add_argument("--no-download-files", action="store_false", dest="download_files")
    parser.add_argument("--download-artifacts", action="store_true", default=True)
    parser.add_argument("--no-download-artifacts", action="store_false", dest="download_artifacts")
    args = parser.parse_args()

    out_root: Path = args.out
    _ensure_dir(out_root)

    api = wandb.Api()
    run_path = f"{args.entity}/{args.project}/{args.run}"
    run = api.run(run_path)

    run_info = {
        "exported_at": _now_iso(),
        "run_path": run_path,
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "created_at": str(run.created_at),
        "heartbeat_at": str(getattr(run, "heartbeat_at", None)),
        "updated_at": str(getattr(run, "updated_at", None)),
        "tags": list(getattr(run, "tags", []) or []),
        "notes": getattr(run, "notes", None),
        "group": getattr(run, "group", None),
        "job_type": getattr(run, "job_type", None),
        "host": getattr(run, "host", None),
        "username": getattr(run, "username", None),
    }
    _json_dump(out_root / "run_info.json", run_info)
    _json_dump(out_root / "config.json", dict(run.config))
    _json_dump(out_root / "summary.json", dict(run.summary._json_dict))

    # Files
    files_manifest: list[dict[str, Any]] = []
    files = list(run.files())
    for f in files:
        files_manifest.append(
            {
                "name": f.name,
                "size": f.size,
                "md5": getattr(f, "md5", None),
                "updated_at": str(getattr(f, "updated_at", None)),
            }
        )
    _json_dump(out_root / "files_manifest.json", {"n": len(files_manifest), "files": files_manifest})

    if args.download_files:
        files_root = out_root / "files"
        _ensure_dir(files_root)
        for f in files:
            _download_run_file(run=run, file_name=f.name, expected_size=f.size, dst_root=files_root)

    # Artifacts (metadata + download)
    artifacts_manifest: dict[str, Any] = {
        "logged": [],
        "used": [],
    }
    logged_arts = list(run.logged_artifacts())
    used_arts = list(run.used_artifacts())
    for a in logged_arts:
        artifacts_manifest["logged"].append(
            {
                "name": a.name,
                "type": a.type,
                "version": a.version,
                "size": a.size,
                "state": getattr(a, "state", None),
                "created_at": str(getattr(a, "created_at", None)),
                "updated_at": str(getattr(a, "updated_at", None)),
            }
        )
    for a in used_arts:
        artifacts_manifest["used"].append(
            {
                "name": a.name,
                "type": a.type,
                "version": a.version,
                "size": a.size,
                "state": getattr(a, "state", None),
                "created_at": str(getattr(a, "created_at", None)),
                "updated_at": str(getattr(a, "updated_at", None)),
            }
        )
    _json_dump(out_root / "artifacts_manifest.json", artifacts_manifest)

    if args.download_artifacts and logged_arts:
        arts_root = out_root / "artifacts"
        _ensure_dir(arts_root)
        for a in logged_arts:
            # Artifact names can contain ':' and '/'.
            safe_name = a.name.replace("/", "__").replace(":", "__")
            target_dir = arts_root / safe_name
            _ensure_dir(target_dir)
            # This downloads the artifact contents (usually a directory).
            a.download(root=str(target_dir))

    # History (full time series)
    history_path = out_root / "history.jsonl"
    if not history_path.exists():
        tmp_path = history_path.with_suffix(".jsonl.tmp")
        keys: set[str] = set()
        n = 0
        _ensure_dir(tmp_path.parent)
        with tmp_path.open("w", encoding="utf-8") as f:
            for row in _iter_scan_history(run, page_size=int(args.history_page_size)):
                n += 1
                keys.update(row.keys())
                f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                if n % 100000 == 0:
                    f.flush()
        tmp_path.replace(history_path)
        _json_dump(out_root / "history_keys.json", {"n_rows": n, "n_keys": len(keys), "keys": sorted(keys)})

    # Light metadata snapshot so we can tell if exports were partial.
    _json_dump(out_root / "export_meta.json", {"exported_at": _now_iso(), "pid": os.getpid()})

    print(f"[OK] Exported {run_path} to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

