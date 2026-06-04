#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Iterable


def _get_hf_token() -> str | None:
    # Never print this token.
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(k)
        if v:
            return str(v)
    return None


def _chunked(it: list, chunk_size: int) -> Iterable[list]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be >0 (got {chunk_size})")
    for i in range(0, len(it), chunk_size):
        yield it[i : i + chunk_size]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Delete *all* files from a Hugging Face Hub repo.\n\n"
            "This is intentionally explicit + confirmation-gated. It is designed for cleaning up\n"
            "large checkpoint repos where HF is the intended source-of-truth.\n"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo_id", default="Gabr1e11/a_lot_of_models", help="Hugging Face repo ID (namespace/name)")
    parser.add_argument("--repo_type", default="model", choices=["model", "dataset", "space"], help="Repo type")
    parser.add_argument(
        "--keep",
        action="append",
        default=[],
        help="Path in repo to keep (repeatable). Example: --keep README.md",
    )
    parser.add_argument(
        "--commit_message",
        default="Wipe repository contents (delete all files)",
        help="Commit message to use for the deletion commit(s)",
    )
    parser.add_argument(
        "--yes_really_delete",
        action="store_true",
        help="Required. Without this flag, the script will only print what it would do.",
    )
    parser.add_argument(
        "--allow_chunked_commits",
        action="store_true",
        help=(
            "If the Hub rejects a single mega-commit (too many operations), allow deleting in multiple commits.\n"
            "This does NOT rewrite history; it just splits the deletions."
        ),
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="Delete operations per commit when --allow_chunked_commits is enabled",
    )
    parser.add_argument(
        "--recreate_repo",
        action="store_true",
        help=(
            "Dangerous alternative: delete the entire repo and recreate it empty.\n"
            "This is a true wipe but changes history irreversibly."
        ),
    )
    parser.add_argument(
        "--yes_really_recreate_repo",
        action="store_true",
        help="Required when --recreate_repo is set.",
    )
    args = parser.parse_args()

    token = _get_hf_token()
    if not token:
        print(
            "[ERROR] No HF token found. Set one of: HF_TOKEN / HUGGINGFACE_HUB_TOKEN / HUGGINGFACE_TOKEN.\n"
            "Do NOT paste tokens into commands that will be logged; prefer setting the env var in your shell.",
            file=sys.stderr,
        )
        return 2

    from huggingface_hub import CommitOperationDelete, HfApi

    api = HfApi(token=token)

    if args.recreate_repo:
        if not args.yes_really_recreate_repo:
            print(
                f"[DRY RUN] Would delete and recreate: {args.repo_type}:{args.repo_id}\n"
                "Re-run with --yes_really_recreate_repo to proceed.",
                file=sys.stderr,
            )
            return 0
        if not args.yes_really_delete:
            print(
                "[ERROR] --recreate_repo requires --yes_really_delete as an extra safety latch.",
                file=sys.stderr,
            )
            return 2

        # NOTE: create_repo(...) preserves private/public if you pass it; we keep default settings.
        api.delete_repo(repo_id=args.repo_id, repo_type=args.repo_type)
        api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, exist_ok=True)
        print(f"[OK] Recreated empty repo: {args.repo_type}:{args.repo_id}")
        return 0

    files = api.list_repo_files(repo_id=args.repo_id, repo_type=args.repo_type)
    keep = set(args.keep or [])
    to_delete = [f for f in files if f not in keep]

    print(f"[INFO] Repo: {args.repo_type}:{args.repo_id}")
    print(f"[INFO] Total files: {len(files)}")
    print(f"[INFO] Keep list: {sorted(keep) if keep else '[]'}")
    print(f"[INFO] Files to delete: {len(to_delete)}")

    if not args.yes_really_delete:
        preview = to_delete[:50]
        if preview:
            print("[DRY RUN] First 50 files that would be deleted:")
            for f in preview:
                print(f"  - {f}")
        print("\nRe-run with --yes_really_delete to proceed.")
        return 0

    if not to_delete:
        print("[OK] Nothing to delete.")
        return 0

    ops = [CommitOperationDelete(path_in_repo=f) for f in to_delete]
    try:
        api.create_commit(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            operations=ops,
            commit_message=args.commit_message,
        )
        print(f"[OK] Deleted {len(to_delete)} files in a single commit.")
        return 0
    except Exception as e:
        if not args.allow_chunked_commits:
            print(
                f"[ERROR] create_commit failed ({type(e).__name__}: {e}).\n"
                "If this is due to too many operations, re-run with --allow_chunked_commits.",
                file=sys.stderr,
            )
            return 1

        print(f"[WARN] Single commit failed; retrying with chunked commits (chunk_size={args.chunk_size}).")
        for chunk in _chunked(ops, args.chunk_size):
            api.create_commit(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                operations=chunk,
                commit_message=args.commit_message,
            )
            print(f"[OK] Deleted {len(chunk)} files in one commit (chunk).")
        print(f"[OK] Deleted {len(to_delete)} files across multiple commits.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

