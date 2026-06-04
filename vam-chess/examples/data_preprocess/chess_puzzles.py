"""
Preprocess a local Chess puzzles dataset to VERL RLHF parquet format.

Input parquet schema (expected):
- `system_prompt` (str): Instructional system message guiding reasoning and answer tags.
- `user_prompt` (str): Contains the FEN string and legal moves in SAN.
- `best_move_uci` (str): Ground-truth best move in UCI notation.
- `move_expectations_json` (str): JSON mapping UCI moves -> float values (e.g., engine eval / Q-values).

Output parquet schema (per VERL docs): one row per sample with keys
- `data_source` (str)
- `prompt` (list[dict]): HuggingFace chat template messages
- `ability` (str): e.g., "chess"
- `reward_model` (dict): contains `style`, `ground_truth`, and optional fields used by the reward fn
- `extra_info` (dict): metadata such as split and index

Usage example:

  python examples/data_preprocess/chess_puzzles.py \
    --local_dataset_path data/puzzles.parquet \
    --local_save_dir data/chess_puzzles \
    --test_ratio 0.1 --seed 42

This will write `train.parquet` and `test.parquet` under `data/chess_puzzles/`.

NOTE (repo contract):
- The current chess reward function (`recipe/chess/reward_fn.py`) expects the model's final move inside
  `<uci_move> ... </uci_move>` tags (not legacy `<answer> ... </answer>`).
- This script rewrites any `<answer>` tag mentions in the prompt text to `<uci_move>` so that newly
  generated parquets are compatible with the strict reward parser.
"""

import argparse
import copy as pycopy
import json
import os
import re
from typing import Any, Dict, Optional, List, Tuple

import datasets
import chess

# Optional dependency on VERL's HDFS helpers. On environments with Python<3.10,
# importing the top-level `verl` module may fail due to newer typing features.
# Provide a lightweight fallback so local preprocessing still works.
try:
    from verl.utils.hdfs_io import copy, makedirs  # type: ignore
except Exception:  # pragma: no cover - runtime convenience
    import shutil

    def makedirs(path: Optional[str]):
        if path is None:
            return
        os.makedirs(path, exist_ok=True)

    def copy(src: str, dst: str):
        # best-effort local copy; no HDFS support in fallback
        if os.path.isdir(src):
            # copytree requires dst not exist
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)


FEN_PATTERN = re.compile(r"Current FEN string:\s*(?P<fen>[^\n]+)")
LEGAL_PATTERN = re.compile(r"Legal moves:\s*(?P<moves>[^\n]+)")
UCI_CANON_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")


def extract_fen_and_legal(user_prompt: str) -> Tuple[Optional[str], Optional[List[str]]]:
    """Parse FEN and legal move list (SAN strings) from the `user_prompt` field.

    The dataset's `user_prompt` contains lines like:

      Current FEN string: <FEN>\n
      Legal moves: <comma-separated SAN tokens>\n

    Returns: (fen, legal_moves) where `legal_moves` is a list of strings; any missing
    item returns as (None, None).
    """
    fen_match = FEN_PATTERN.search(user_prompt)
    legal_match = LEGAL_PATTERN.search(user_prompt)
    fen: Optional[str] = fen_match.group("fen").strip() if fen_match else None
    legal_moves: Optional[List[str]] = None
    if legal_match:
        # split by comma, strip whitespace
        legal_moves = [m.strip() for m in legal_match.group("moves").split(",") if m.strip()]
    return fen, legal_moves


def canonicalize_uci(move: Optional[str]) -> Optional[str]:
    """Normalize a UCI move string (e.g., e2e4, e7e8q)."""
    if not move:
        return None
    m = move.strip().strip("`'\"").rstrip(".!?;,")
    m = m.replace("=", "").lower()
    if UCI_CANON_RE.fullmatch(m):
        return m
    return None


def normalize_move_values_json(json_str: str) -> Optional[str]:
    """Return a canonical JSON string of the move->value mapping (UCI keys), or None.

    Storing as string avoids Arrow schema explosion from nested dicts with
    variable keys across samples.
    """
    try:
        obj = json.loads(json_str)
        if not isinstance(obj, dict):
            return None
        cleaned: Dict[str, float] = {}
        for k, v in obj.items():
            try:
                canon = canonicalize_uci(str(k))
                if canon is None:
                    continue
                cleaned[canon] = float(v)
            except Exception:
                pass
        # sort keys for determinism
        return json.dumps(cleaned, sort_keys=True, separators=(",", ":"))
    except Exception:
        return None


def legal_moves_from_fen(fen: Optional[str]) -> Optional[List[str]]:
    """Return legal moves in UCI for a FEN string."""
    if not fen:
        return None
    try:
        board = chess.Board(fen)
        return [m.uci().lower() for m in board.legal_moves]
    except Exception:
        return None


def build_user_message(system_prompt: str, user_prompt: str) -> str:
    """Return a single chat message combining system+user text."""
    system_prompt = (system_prompt or "").strip()
    user_prompt = (user_prompt or "").strip()
    if system_prompt:
        return f"{system_prompt}\n\n{user_prompt}".strip()
    return user_prompt


def rewrite_prompt_to_uci(text: str, legal_moves_uci: Optional[List[str]]) -> str:
    """Rewrite SAN references and inject UCI legal moves."""
    if not text:
        return ""
    out = text
    out = re.sub(
        r"The answer must be in SAN notation[^\n]*",
        "The answer must be in UCI notation, using the from-square and to-square (e.g., e2e4, g1f3, a7a8q).",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"best move in SAN notation within <answer> </answer> tags\. i\.e\., <answer>[^<]*</answer>",
        "best move in UCI notation within <uci_move> </uci_move> tags. i.e., <uci_move> e2e4 </uci_move>",
        out,
        flags=re.IGNORECASE,
    )
    if legal_moves_uci:
        out = re.sub(
            r"Legal moves:[^\n]*",
            f"Legal moves (UCI): {', '.join(legal_moves_uci)}",
            out,
            flags=re.IGNORECASE,
        )
    out = re.sub(r"\bSAN\b", "UCI", out)
    # Rewrite legacy tag mentions to the strict `<uci_move>` contract.
    out = re.sub(r"<\s*/\s*answer\s*>", "</uci_move>", out, flags=re.IGNORECASE)
    out = re.sub(r"<\s*answer\s*>", "<uci_move>", out, flags=re.IGNORECASE)
    # If the source prompt doesn't mention any output tags at all, inject a minimal
    # `<uci_move>` instruction so the generated dataset remains compatible with the
    # current strict chess reward parser.
    out_lower = out.lower()
    if ("<uci_move>" not in out_lower) and ("</uci_move>" not in out_lower):
        out = (
            out.rstrip()
            + "\n\nAfter analyzing the position, clearly state the best move in UCI notation "
            "within <uci_move> </uci_move> tags. i.e., <uci_move> e2e4 </uci_move>"
        )

    # Adopt the `<guess>` scheme: request an explicit guess line before the `<think>` block.
    #
    # Note: scoring ignores `<guess>...</guess>`, but prompting it improves consistency
    # and aligns training with evaluation-time experiments.
    if "<guess>" not in out_lower:
        out = (
            out.rstrip()
            + "\n\n"
            "IMPORTANT (format): Before your <think> block, output exactly one guess line:\n"
            "<guess> GUESS_UCI </guess>\n"
            "Then output the usual strict answer format:\n"
            "<think> ... </think><uci_move> ... </uci_move>\n"
            "Do not write any other text outside these tags.\n"
        )
    return out


def parse_move_map(move_values_json: Optional[str]) -> Dict[str, float]:
    """Load the expectation map and normalize keys."""
    if not move_values_json:
        return {}
    try:
        obj = json.loads(move_values_json)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    move_map: Dict[str, float] = {}
    for k, v in obj.items():
        try:
            canon = canonicalize_uci(str(k))
            if canon is None:
                continue
            move_map[canon] = float(v)
        except Exception:
            continue
    return move_map


def select_wrong_hint_move(move_map: Dict[str, float], gt_move: str) -> Optional[str]:
    """Select the highest-valued move (deterministically) that differs from gt_move."""
    candidates = [
        (value, move)
        for move, value in move_map.items()
        if move != gt_move
    ]
    if not candidates:
        return None
    # Sort by value desc, then lexicographically for determinism.
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def render_hint_text(move_uci: str) -> str:
    """Create the user-visible hint string in UCI format."""
    return (
        f"Hint: Consider exploring the move {move_uci} (UCI). "
        "This hint may be misleading—verify it against the position before answering."
    )


def inject_hint_into_prompt(user_prompt: str, hint_text: Optional[str]) -> str:
    """Append the hint (if any) to the base user prompt."""
    base = (user_prompt or "").rstrip()
    if hint_text:
        return f"{base}\n\n{hint_text}"
    return base


def expand_dataset_with_hints(
    dataset: datasets.Dataset,
    *,
    hint_data_source: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], int]:
    """Return: (all rows, rows grouped by hint type, skipped count)."""
    rows: List[Dict[str, Any]] = []
    rows_by_variant: Dict[str, List[Dict[str, Any]]] = {"correct": [], "none": [], "wrong": []}
    skipped = 0
    for row in dataset:
        reward_model = row.get("reward_model") or {}
        move_map = parse_move_map(reward_model.get("move_values_json"))
        gt_uci = (reward_model.get("ground_truth") or "").strip().lower()
        if not move_map or gt_uci not in move_map:
            skipped += 1
            continue
        wrong_hint = select_wrong_hint_move(move_map, gt_uci)
        if wrong_hint is None:
            skipped += 1
            continue

        extra_info = row.get("extra_info") or {}
        system_prompt = extra_info.get("system_prompt", "")
        base_user_prompt = extra_info.get("user_prompt", "")

        variant_specs = [
            ("correct", gt_uci),
            ("none", None),
            ("wrong", wrong_hint),
        ]
        for variant_idx, (variant_type, move_uci) in enumerate(variant_specs):
            variant = pycopy.deepcopy(row)
            hint_text = render_hint_text(move_uci) if move_uci else ""
            user_prompt_with_hint = inject_hint_into_prompt(base_user_prompt, hint_text if hint_text else None)
            variant["prompt"] = [
                {
                    "role": "user",
                    "content": build_user_message(system_prompt, user_prompt_with_hint),
                }
            ]
            variant["data_source"] = f"{hint_data_source}/{variant_type}"
            rm_variant = variant.get("reward_model", {})
            rm_variant["hint_move_uci"] = move_uci
            rm_variant["hint_quality"] = variant_type
            rm_variant["hint_present"] = bool(move_uci)
            variant["reward_model"] = rm_variant

            extra_variant = variant.get("extra_info", {})
            extra_variant["hint_type"] = variant_type
            extra_variant["hint_move_uci"] = move_uci or ""
            extra_variant["hint_text"] = hint_text
            extra_variant["hint_variant_index"] = variant_idx
            extra_variant["user_prompt_with_hint"] = user_prompt_with_hint
            variant["extra_info"] = extra_variant

            rows.append(variant)
            rows_by_variant.setdefault(variant_type, []).append(variant)
    return rows, rows_by_variant, skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert chess puzzles parquet to VERL RLHF parquet format")
    parser.add_argument(
        "--local_dataset_path",
        default="data/puzzles.parquet",
        help="Path to source puzzles parquet.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/chess_puzzles",
        help="Directory to write processed parquet files.",
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS destination directory.")
    parser.add_argument("--data_source", default="local/chess_puzzles", help="Name used to tag data_source field.")
    parser.add_argument("--ability", default="chess", help="Ability tag for this dataset.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Fraction of data for test split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")

    args = parser.parse_args()

    # Load raw puzzles as a HuggingFace Dataset from parquet
    ds_raw = datasets.load_dataset("parquet", data_files=args.local_dataset_path)["train"]

    # Deterministic split into train/test
    split = ds_raw.train_test_split(test_size=args.test_ratio, seed=args.seed)
    train_ds = split["train"]
    test_ds = split["test"]

    # Construct chat prompt and reward payload for each row
    def make_map_fn(split_name: str):
        def process_fn(example: Dict[str, Any], idx: int):
            # Pop raw fields to keep the resulting schema clean and small
            system_prompt = (example.pop("system_prompt", "") or "").strip()
            user_prompt = (example.pop("user_prompt", "") or "").strip()
            # Build a single-user message (safer across chat templates)
            fen, _ = extract_fen_and_legal(user_prompt)
            legal_moves_uci = legal_moves_from_fen(fen) or []
            move_values_json = normalize_move_values_json(example.pop("move_expectations_json", "") or "")

            best_move_uci = canonicalize_uci(example.pop("best_move_uci", "") or "") or ""

            system_prompt = rewrite_prompt_to_uci(system_prompt, legal_moves_uci)
            user_prompt = rewrite_prompt_to_uci(user_prompt, legal_moves_uci)
            user_message = (system_prompt + "\n\n" + user_prompt).strip() if system_prompt else user_prompt

            data = {
                "data_source": args.data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": user_message,
                    }
                ],
                "ability": args.ability,
                # Store all details needed for a rule-based reward function
                "reward_model": {
                    "style": "rule",
                    # Ground truth uses UCI
                    "ground_truth": best_move_uci,
                    # Additional fields useful for scoring
                    "fen": fen,
                    "legal_moves_uci": legal_moves_uci,
                    "move_values_json": move_values_json,
                },
                "extra_info": {
                    "split": split_name,
                    "index": idx,
                    # Keep raw pieces for debugging/eval
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "legal_moves_uci": legal_moves_uci,
                },
            }
            return data

        return process_fn

    train_ds = train_ds.map(function=make_map_fn("train"), with_indices=True)
    test_ds = test_ds.map(function=make_map_fn("test"), with_indices=True)

    # Save to parquet files
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    train_ds.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_ds.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    # Expand to hint-augmented datasets (correct/no-hint/wrong variants per base row)
    hint_data_source = f"{args.data_source}/hint"
    train_hint_rows, train_hint_variants, train_hint_skipped = expand_dataset_with_hints(
        train_ds, hint_data_source=hint_data_source
    )
    test_hint_rows, test_hint_variants, test_hint_skipped = expand_dataset_with_hints(
        test_ds, hint_data_source=hint_data_source
    )
    if train_hint_rows:
        datasets.Dataset.from_list(train_hint_rows).to_parquet(os.path.join(local_save_dir, "train_hint.parquet"))
    # Dedicated evaluation splits: correct-hint and wrong-hint only.
    def _write_hint_subset(rows: List[Dict[str, Any]], filename: str):
        if not rows:
            return
        datasets.Dataset.from_list(rows).to_parquet(os.path.join(local_save_dir, filename))

    _write_hint_subset(test_hint_variants.get("correct", []), "test_correct_hint.parquet")
    _write_hint_subset(test_hint_variants.get("wrong", []), "test_wrong_hint.parquet")
    print(
        f"Hint dataset stats — train: kept {len(train_hint_rows)//3} rows "
        f"(skipped {train_hint_skipped}), test: kept {len(test_hint_rows)//3} rows "
        f"(skipped {test_hint_skipped})"
    )

    # Optional: copy to HDFS if requested
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)

    print(f"Wrote train/test parquet to: {local_save_dir}")
