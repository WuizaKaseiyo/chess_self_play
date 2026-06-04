"""Utilities to fetch the Searchless Chess critic code/checkpoint on demand."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / ".third_party_cache" / "searchless_chess"
REPO_URL = "https://github.com/google-deepmind/searchless_chess.git"
HF_REPO_ID = "Gabr1e11/chess"
HF_PARAMS_SUBDIR = "searchless_critic_270M_params"
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # provide via env; never commit a token


def _clone_repo() -> None:
  """Clone the upstream Searchless Chess repo into the cache."""
  CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
  if CACHE_DIR.exists():
    return
  subprocess.run(
      ["git", "clone", "--depth", "1", REPO_URL, str(CACHE_DIR)],
      check=True,
  )


def _download_checkpoint_from_hf(target_params: Path) -> None:
  """Download the critic checkpoint directory from Hugging Face."""
  try:
    from huggingface_hub import snapshot_download
  except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "huggingface_hub is required to download the Searchless checkpoint.\n"
        "Install it with `pip install huggingface_hub` in the active environment."
    ) from exc

  snapshot_path = Path(
      snapshot_download(
          repo_id=HF_REPO_ID,
          repo_type="dataset",
          allow_patterns=f"{HF_PARAMS_SUBDIR}/**",
          token=HF_TOKEN,
      )
  )
  source_dir = snapshot_path / HF_PARAMS_SUBDIR
  if not source_dir.exists():
    raise FileNotFoundError(
        f"Unable to locate {HF_PARAMS_SUBDIR} inside {snapshot_path}"
    )
  shutil.copytree(source_dir, target_params, dirs_exist_ok=True)


def _ensure_checkpoint() -> None:
  """Make sure the 270M checkpoint exists under the cached repo."""
  checkpoints_dir = CACHE_DIR / "checkpoints"
  target_params = checkpoints_dir / "270M" / "6400000" / "params"
  if target_params.exists():
    return
  target_params.parent.mkdir(parents=True, exist_ok=True)
  _download_checkpoint_from_hf(target_params)


def prepare_searchless_src() -> Path:
  """Returns the path to searchless_chess/src, cloning/extracting if needed."""
  _clone_repo()
  _ensure_checkpoint()
  src_dir = CACHE_DIR / "src"
  if not src_dir.exists():
    raise FileNotFoundError(
        f"searchless_chess/src not found under {CACHE_DIR}. "
        "Clone may have failed."
    )
  return src_dir


@contextlib.contextmanager
def pushd(path: Path):
  prev_cwd = Path.cwd()
  os.chdir(path)
  try:
    yield path
  finally:
    os.chdir(prev_cwd)
