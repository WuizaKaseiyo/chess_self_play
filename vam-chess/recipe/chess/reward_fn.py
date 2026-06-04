"""
Chess reward for restricted-moves ("selection") training/eval.

Contract (selection framing; source-of-truth: `restricted_moves.md`)
- Responses must contain a closed `<reason>...</reason>` block followed later by
  exactly one `<uci_move>...</uci_move>` tag. Text/whitespace is allowed
  between `</reason>` and `<uci_move>`.
- The `<uci_move>` payload must be strict lowercase UCI with no surrounding
  whitespace.
- The parsed move must be **one of the provided candidate moves** ("considered moves") **only**
  when the prompt is a selection prompt (it contains the keyword `allowed_moves`).
  - Selection use-subset can be explicitly overridden via
    `extra_info["use_considered_moves_uci"]` (bool-like).
  - Otherwise we detect selection prompts via `extra_info["prompt_text"]`
    (attached by the reward manager).
  - If no prompt is available, we fall back to the presence of `reward_model.considered_moves_uci`.
  - For baseline prompts (no `allowed_moves`), we ignore `considered_moves_uci`
    and use the full legal list.

Target definition (μ-based; subset-relative)
- Let S be the candidate set (considered moves).
- Let μ(m) be the per-move expected score, taken from:
    1) `move_expected_scores_json` if present, else
    2) `move_values_json`
- The target move is `argmax_{m ∈ S} μ(m)` with deterministic tie-break:
    - higher μ wins; if μ ties, lexicographically smaller UCI wins.

Reward
- `score = 1.0` iff the prediction equals the μ-target within the candidate set.
- `score = 0.0` for an in-subset but non-target move.
- `score = -1.0` for format errors / invalid UCI / out-of-subset.

Additional reward modes (training-time shaping; see `CHESS_REWARD_FN`)
- `gt_gated`: "dead-group gating" on the dataset ground-truth move.
  - Valid in-subset move: `score=1.0` iff `pred_move == gt_uci`, else `0.0`.
  - Keeps the same strict `-1.0` penalties for format/out-of-subset.
- `gt_expected_threshold`: relaxed GT gating based on expected-score proximity.
  - Requires `move_expected_scores_json` to be present.
  - Let `expected_gt = expected_score[gt_uci]`, `expected_pred = expected_score[pred_move]`.
  - If `abs(expected_pred - expected_gt) <= gt_expected_score_diff_threshold`: `score=expected_pred`, else `0.0`.
  - Keeps the same strict `-1.0` penalties for format/out-of-subset.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Optional

# Parsing: reward requires a closed reasoning block followed later by exactly one
# strict `<uci_move>...</uci_move>` span. Text/whitespace between the reasoning
# block and `<uci_move>` is allowed. Qwen3-4B-Base uses `<reason>` to avoid the
# model-special `<think>` behavior; legacy prompt templates may still use `<think>`.
REASON_RESPONSE_RE = re.compile(
    r"^<reason>[\s\S]*?</reason>(?:(?!<uci_move>)[\s\S])*<uci_move>(?P<ans>[a-h][1-8][a-h][1-8][qrbn]?)</uci_move>$",
    re.DOTALL,
)
THINK_RESPONSE_RE = re.compile(
    r"^<think>[\s\S]*?</think>(?:(?!<uci_move>)[\s\S])*<uci_move>(?P<ans>[a-h][1-8][a-h][1-8][qrbn]?)</uci_move>$",
    re.DOTALL,
)
# Base-model prompts can end with the opening `<reason>` as literal prompt text.
# Because reward receives only decoded response tokens, this regex validates the
# response fragment that follows that open prompt-side tag.
OPEN_REASON_PROMPT_RESPONSE_RE = re.compile(
    r"^[\s\S]*?</reason>(?:(?!<uci_move>)[\s\S])*<uci_move>(?P<ans>[a-h][1-8][a-h][1-8][qrbn]?)</uci_move>$",
    re.DOTALL,
)
OPEN_THINK_PROMPT_RESPONSE_RE = re.compile(
    r"^[\s\S]*?</think>(?:(?!<uci_move>)[\s\S])*<uci_move>(?P<ans>[a-h][1-8][a-h][1-8][qrbn]?)</uci_move>$",
    re.DOTALL,
)
# Legacy span regex kept for full-game eval / analysis utilities that intentionally
# parse a `<uci_move>` span without enforcing the training reward's full contract.
UCI_MOVE_ONLY_RE = re.compile(
    r"<\s*uci_move\s*>(?P<ans>[\s\S]*?)<\s*/\s*uci_move\s*>",
    re.DOTALL | re.IGNORECASE,
)
UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$", re.IGNORECASE)

# Match dataset scoring in `scripts/rescore_puzzles_cp.py`.
K_CP_TO_WINPROB = 0.00368208


def _to_uci(token: str) -> Optional[str]:
    """Return normalized UCI string or None."""
    if not token:
        return None
    t = token.strip()
    t = t.strip("`'\"")
    t = t.rstrip(".!?;,:")
    # Normalize promotions with '=' (e7e8=Q -> e7e8q), then lowercase
    if re.fullmatch(r"[a-h][1-8][a-h][1-8]=[QRBNqrbn]", t):
        return t.replace("=", "").lower()
    if UCI_RE.fullmatch(t):
        return t.lower()
    return None


def _extract_prompt_text(extra: Any) -> Optional[str]:
    if not isinstance(extra, dict):
        return None
    prompt_raw = extra.get("prompt_text") or extra.get("prompt") or extra.get("prompt_str")
    if prompt_raw is None:
        return None
    if isinstance(prompt_raw, list):
        parts: list[str] = []
        for msg in prompt_raw:
            if isinstance(msg, dict):
                parts.append(str(msg.get("content") or ""))
            else:
                parts.append(str(msg))
        prompt_text = "\n".join(parts)
    elif isinstance(prompt_raw, dict):
        prompt_text = str(prompt_raw.get("content") or "")
    else:
        prompt_text = str(prompt_raw)
    if not prompt_text.strip():
        return None
    return prompt_text


def _prompt_ends_with_open_reason(extra: Any) -> bool:
    prompt_text = _extract_prompt_text(extra)
    return bool(prompt_text and prompt_text.rstrip().endswith("<reason>"))


def _prompt_ends_with_open_think(extra: Any) -> bool:
    prompt_text = _extract_prompt_text(extra)
    return bool(prompt_text and prompt_text.rstrip().endswith("<think>"))


def _extract_uci_move_payload(
    solution_str: str,
    *,
    prompt_ends_with_open_reason: bool = False,
    prompt_ends_with_open_think: bool = False,
) -> tuple[Optional[str], float]:
    """Extract a strict reasoning-tag + `<uci_move>...</uci_move>` payload.

    Returns: (uci_move_payload_or_none, format_reward).

    `format_reward=1.0` iff the whole response contains a closed top-level
    reasoning block followed by one strict UCI move tag. If the prompt text
    itself ends with an open reasoning tag, the decoded response is allowed to
    begin inside that reasoning block but must still close it before the strict
    `<uci_move>` tag.
    """
    s0 = solution_str or ""
    match = None
    if match is None and prompt_ends_with_open_reason:
        match = OPEN_REASON_PROMPT_RESPONSE_RE.fullmatch(s0)
    if match is None and prompt_ends_with_open_think:
        match = OPEN_THINK_PROMPT_RESPONSE_RE.fullmatch(s0)
    if match is None and not (prompt_ends_with_open_reason or prompt_ends_with_open_think):
        match = REASON_RESPONSE_RE.fullmatch(s0) or THINK_RESPONSE_RE.fullmatch(s0)
    if match is None:
        return None, 0.0
    payload = match.group("ans")
    return payload, 1.0


# Back-compat shim (legacy name used by some analysis scripts). The "guess" payload is always None now.
def _extract_guess_then_uci_move(solution_str: str) -> tuple[Optional[str], Optional[str], float]:
    payload, format_reward = _extract_uci_move_payload(solution_str)
    return None, payload, float(format_reward)


def _resolve_move(token: Optional[str], fen: str) -> tuple[Optional[str], Optional[str]]:
    """Return (uci_move, penalty_reason)."""
    if not token:
        return None, "format_error"

    pred_uci = _to_uci(token)
    if pred_uci:
        return pred_uci, None

    # Strict: SAN is disallowed; only UCI is accepted.
    return None, "bad_move"


def _normalize_moves(moves: Any) -> list[str]:
    """Best-effort normalize a move list to lowercase UCI-like strings (order-preserving)."""
    if moves is None:
        return []
    if isinstance(moves, str):
        s = moves.strip().lower()
        return [s] if s else []
    out: list[str] = []
    try:
        for m in moves:
            s = str(m).strip().lower()
            if s:
                out.append(s)
    except Exception:
        return []
    return out


def _parse_move_values_json(move_values_json: Any) -> Dict[str, float]:
    if not move_values_json:
        return {}

    parsed: Any = None
    if isinstance(move_values_json, dict):
        parsed = move_values_json
    elif isinstance(move_values_json, str):
        try:
            parsed = json.loads(move_values_json)
        except Exception:
            return {}
    else:
        return {}

    if not isinstance(parsed, dict):
        return {}

    out: Dict[str, float] = {}
    for k, v in parsed.items():
        if k is None:
            continue
        key = str(k).strip().lower()
        if not key:
            continue
        try:
            out[key] = float(v)
        except Exception:
            continue
    return out


def _sigmoid_stable(z: float) -> float:
    # Numerically stable sigmoid.
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def centipawn_to_win_prob(cp: float) -> float:
    # win% = 100/(1 + exp(-k*cp)) -> win_prob = sigmoid(k*cp)
    return _sigmoid_stable(K_CP_TO_WINPROB * float(cp))


def _logit(p: float, *, eps: float) -> float:
    p2 = min(max(float(p), eps), 1.0 - eps)
    return math.log(p2 / (1.0 - p2))


def _maybe_parse_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _parse_move_float_map(map_json: Any) -> Dict[str, float]:
    # Same contract as `_parse_move_values_json`, but for any float-valued map.
    return _parse_move_values_json(map_json)


def _compute_rank_reward(value_map: Dict[str, float], pred_move: str) -> float:
    # Rank among unique values to treat ties consistently.
    # Worst=0, best=1.
    if not value_map or pred_move not in value_map:
        return float("nan")
    unique = sorted({float(v) for v in value_map.values()})
    if not unique:
        return float("nan")
    if len(unique) == 1:
        return 1.0
    v = float(value_map[pred_move])
    try:
        idx = unique.index(v)
    except ValueError:
        # Should not happen, but guard against float roundoff.
        idx = min(range(len(unique)), key=lambda i: abs(unique[i] - v))
    return float(idx) / float(len(unique) - 1)


def _compute_move_reward(
    *,
    chess_reward_fn: str,
    pred_move: str,
    winprob_after: float,
    winprob_best: float,
    winprob_before: float,
    cp_after: float,
    cp_best: float,
    cp_before: float,
    expected_after: float,
    expected_best: float,
    expected_before: float,
    move_winprob_map: Dict[str, float],
    move_cp_map: Dict[str, float],
    logit_eps: float,
) -> tuple[float, str]:
    """Return (reward, reward_reason)."""
    fn = (chess_reward_fn or "winrate").strip()

    if fn == "winrate":
        return float(winprob_after), ""

    if fn == "delta_winrate":
        return float(winprob_after - winprob_before), ""

    if fn == "winrate_vs_best":
        return float(winprob_after - winprob_best), ""

    if fn == "delta_centipawn":
        return float(cp_after - cp_before), ""

    if fn == "centipawn_vs_best":
        return float(cp_after - cp_best), ""

    if fn == "delta_logit_winrate":
        return float(_logit(winprob_after, eps=logit_eps) - _logit(winprob_before, eps=logit_eps)), ""

    if fn == "logit_winrate_vs_best":
        return float(_logit(winprob_after, eps=logit_eps) - _logit(winprob_best, eps=logit_eps)), ""

    if fn == "delta_expected_score_wdl":
        return float(expected_after - expected_before), ""

    if fn == "expected_score_wdl_vs_best":
        return float(expected_after - expected_best), ""

    if fn == "rank_among_moves":
        # Prefer CP rank when available (more informative in very low/high winprob
        # regimes), otherwise rank by winprob.
        if move_cp_map:
            return float(_compute_rank_reward(move_cp_map, pred_move)), ""
        return float(_compute_rank_reward(move_winprob_map, pred_move)), ""

    # Unknown reward_fn: fall back to the original winrate shaping.
    return float(winprob_after), f"unknown_reward_fn:{fn}"


def compute_score(
    data_source: Any,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    **reward_kwargs: Any,
) -> Dict[str, Any] | float:
    """Chess reward for selection datasets.

    By default this implements the strict selection reward described at the top of this file.

    You can optionally override the reward shaping by passing:
      - `chess_reward_fn=<name>` via `custom_reward_function.reward_kwargs.*`.

    Supported values include:
      - `selection` (default): 1/0/-1 subset-relative correctness.
      - `gt_gated`: 1/0/-1, but positive reward only when `pred_move == gt_uci`.
      - `gt_expected_threshold`: 0..1 / 0 / -1 based on expected-score proximity to `gt_uci`.
      - `winrate`: returns Stockfish win-prob for the predicted move (0..1), with the same -1 penalties.
      - `expected_score_wdl_vs_best`: expected_score(pred) - expected_score(best_legal) (<=0), with the same -1 penalties.

    All modes enforce the same strict parsing. The in-subset constraint is applied only when the prompt
    contains `allowed_moves` (fallback: `considered_moves_uci` present and no prompt string available),
    unless explicitly overridden by `extra_info["use_considered_moves_uci"]`.
    """
    rm = data_source if isinstance(data_source, dict) else {}

    chess_reward_fn = str(reward_kwargs.get("chess_reward_fn") or "selection").strip()
    if not chess_reward_fn:
        chess_reward_fn = "selection"

    gt_uci = (ground_truth or "").strip().lower()
    if not gt_uci and isinstance(rm, dict):
        gt_uci = (rm.get("ground_truth") or "").strip().lower()

    legal_moves_list = _normalize_moves(rm.get("legal_moves_uci"))
    legal_moves_set = set(legal_moves_list)

    def _parse_optional_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value == 1:
                return True
            if value == 0:
                return False
            return None
        if isinstance(value, str):
            s = value.strip().lower()
            if s in {"1", "true", "t", "yes", "y", "on"}:
                return True
            if s in {"0", "false", "f", "no", "n", "off"}:
                return False
        return None

    def _explicit_subset_override(extra: Any) -> Optional[bool]:
        if not isinstance(extra, dict):
            return None
        return _parse_optional_bool(extra.get("use_considered_moves_uci"))

    def _prompt_contains_allowed_moves(extra: Any) -> Optional[bool]:
        prompt_text = _extract_prompt_text(extra)
        if not prompt_text:
            return None
        return "allowed_moves" in prompt_text.lower()

    explicit_subset_override = _explicit_subset_override(extra_info)
    prompt_has_allowed_moves = _prompt_contains_allowed_moves(extra_info)

    # Candidate set for selection prompts: `considered_moves_uci` when present; else fall back to full legal moves.
    considered_raw = rm.get("considered_moves_uci")
    if considered_raw is None:
        considered_raw = rm.get("considered_moves_uci_list")

    use_subset = None
    if explicit_subset_override is not None:
        use_subset = explicit_subset_override
    elif prompt_has_allowed_moves is not None:
        use_subset = prompt_has_allowed_moves
    else:
        use_subset = bool(considered_raw)

    if use_subset:
        considered_list = _normalize_moves(considered_raw)
        # De-dupe while preserving order.
        seen: set[str] = set()
        considered_list = [m for m in considered_list if not (m in seen or seen.add(m))]
        if not considered_list:
            considered_list = list(legal_moves_list) if legal_moves_list else sorted(list(legal_moves_set))
    else:
        considered_list = list(legal_moves_list) if legal_moves_list else sorted(list(legal_moves_set))

    considered_set = set(considered_list)
    if use_subset and legal_moves_set and not considered_set.issubset(legal_moves_set):
        illegal = sorted(list(considered_set - legal_moves_set))[:10]
        raise ValueError(f"Invalid dataset row: considered_moves contains illegal moves: {illegal}")

    # μ map (expected scores preferred).
    mu_expected = _parse_move_float_map(rm.get("move_expected_scores_json"))
    mu_values = _parse_move_values_json(rm.get("move_values_json"))
    mu_map = mu_expected if mu_expected else mu_values
    mu_source = "move_expected_scores_json" if mu_expected else "move_values_json"
    if not mu_map:
        raise ValueError("Empty mu_map (expected move_expected_scores_json or move_values_json).")

    def _best_move_by_mu(mu_map: dict[str, float], moves: list[str]) -> tuple[str, float]:
        best_move = ""
        best_mu = -float("inf")
        for mv in moves:
            key = str(mv).strip().lower()
            mu = float(mu_map.get(key, -float("inf")))
            if (mu > best_mu) or (mu == best_mu and (not best_move or key < best_move)):
                best_move = key
                best_mu = mu
        if not best_move:
            raise ValueError("Empty candidate set when selecting μ-target.")
        return best_move, float(best_mu)

    target_move, target_mu = _best_move_by_mu(mu_map, considered_list)

    prompt_ends_with_open_reason = _prompt_ends_with_open_reason(extra_info)
    prompt_ends_with_open_think = _prompt_ends_with_open_think(extra_info)
    payload, format_reward = _extract_uci_move_payload(
        solution_str or "",
        prompt_ends_with_open_reason=prompt_ends_with_open_reason,
        prompt_ends_with_open_think=prompt_ends_with_open_think,
    )
    pred_move = ""
    penalty_reason = ""
    if payload is None:
        penalty_reason = "format_error"
    else:
        pred_uci = _to_uci(payload)
        if pred_uci is None:
            penalty_reason = "bad_move"
        else:
            pred_move = pred_uci
            if format_reward < 1.0:
                # Multiple tags are a format error even if we can parse the first one.
                penalty_reason = "format_error"

    in_subset = bool(pred_move) and (pred_move in considered_set)
    if penalty_reason == "" and not in_subset:
        if use_subset:
            penalty_reason = "out_of_subset"
        elif legal_moves_set and pred_move not in legal_moves_set:
            penalty_reason = "illegal_move"

    penalty_applied = penalty_reason != ""

    # For debugging: whether the model matched the dataset's (global) ground_truth label.
    exact_match_gt = 1.0 if (gt_uci and pred_move == gt_uci) else 0.0

    mu_pred = float(mu_map.get(pred_move, float("nan"))) if pred_move else float("nan")

    score: float
    reward_reason = ""
    # `acc` is always reported as selection accuracy (subset-target match), even when using other reward shaping.
    acc = 1.0 if (pred_move and pred_move == target_move and not penalty_applied) else 0.0

    if penalty_applied:
        score = -1.0
        reward_reason = penalty_reason
    elif chess_reward_fn == "selection":
        score = float(acc)
        reward_reason = "selection"
    elif chess_reward_fn == "gt_gated":
        # Strict "dead-group gating" on the dataset label: for valid in-subset moves, only
        # reward exact ground-truth hits. Keep penalties strict (-1.0).
        if not gt_uci:
            score = 0.0
            reward_reason = "gt_gated:missing_gt"
        elif pred_move == gt_uci:
            score = 1.0
            reward_reason = "gt_gated:hit"
        else:
            score = 0.0
            reward_reason = "gt_gated:miss"
    elif chess_reward_fn == "gt_expected_threshold":
        # Relaxed gating: allow "near GT" moves based on expected-score distance.
        # NOTE: This intentionally uses `move_expected_scores_json` (not winprob), because
        # it matches the selection target μ definition.
        expected_map = _parse_move_float_map(rm.get("move_expected_scores_json"))
        threshold_raw = reward_kwargs.get("gt_expected_score_diff_threshold", 0.0)
        try:
            threshold = float(threshold_raw)
        except Exception:
            threshold = 0.0
        if not math.isfinite(threshold):
            threshold = 0.0
        threshold = max(0.0, threshold)

        if not gt_uci:
            score = 0.0
            reward_reason = "gt_expected_threshold:missing_gt"
        elif not expected_map:
            score = 0.0
            reward_reason = "gt_expected_threshold:missing_expected_map"
        else:
            expected_gt = expected_map.get(gt_uci)
            expected_pred = expected_map.get(pred_move) if pred_move else None
            if expected_gt is None:
                score = 0.0
                reward_reason = "gt_expected_threshold:missing_expected_gt"
            elif expected_pred is None:
                score = 0.0
                reward_reason = "gt_expected_threshold:missing_expected_pred"
            else:
                try:
                    expected_gt_f = float(expected_gt)
                    expected_pred_f = float(expected_pred)
                except Exception:
                    score = 0.0
                    reward_reason = "gt_expected_threshold:non_float_expected"
                else:
                    if not (math.isfinite(expected_gt_f) and math.isfinite(expected_pred_f)):
                        score = 0.0
                        reward_reason = "gt_expected_threshold:non_finite_expected"
                    else:
                        diff = abs(expected_pred_f - expected_gt_f)
                        if diff <= threshold:
                            score = float(expected_pred_f)
                            reward_reason = f"gt_expected_threshold:hit(diff={diff:.6g},thr={threshold:.6g})"
                        else:
                            score = 0.0
                            reward_reason = f"gt_expected_threshold:miss(diff={diff:.6g},thr={threshold:.6g})"
    else:
        move_winprob_map = _parse_move_float_map(rm.get("move_values_json"))
        move_expected_map = _parse_move_float_map(rm.get("move_expected_scores_json"))
        move_cp_map = _parse_move_float_map(rm.get("move_cps_json"))

        row_id = None
        if isinstance(extra_info, dict):
            row_id = extra_info.get("index")
        fen_ctx = str(rm.get("fen") or "")

        def _fail(msg: str) -> None:
            raise ValueError(f"{msg} (chess_reward_fn={chess_reward_fn}, row_id={row_id}, fen={fen_ctx!r})")

        def _validate_float_map(
            *,
            name: str,
            value_map: dict[str, float],
            require_keys: Optional[set[str]] = None,
            require_pred: bool = True,
            enforce_01: bool = False,
        ) -> None:
            if not value_map:
                _fail(f"Missing/empty reward payload field: {name}")
            if require_pred and pred_move not in value_map:
                _fail(f"{name} is missing predicted move: pred_move={pred_move!r}")
            if require_keys:
                missing = [m for m in require_keys if m not in value_map]
                if missing:
                    _fail(f"{name} is missing {len(missing)} legal moves (e.g. {sorted(missing)[:5]})")
            bad = []
            for k, v in value_map.items():
                try:
                    vf = float(v)
                except Exception:
                    bad.append((k, "non_float"))
                    continue
                if not math.isfinite(vf):
                    bad.append((k, "non_finite"))
                    continue
                if enforce_01 and not (0.0 <= vf <= 1.0):
                    bad.append((k, f"out_of_range:{vf}"))
            if bad:
                head = bad[:5]
                _fail(f"{name} contains invalid values (e.g. {head})")

        # Validate reward-mode-specific required inputs. This is intentionally strict:
        # producing NaN rewards breaks GRPO/optimizer stats in hard-to-debug ways.
        fn = chess_reward_fn
        known = {
            "winrate",
            "delta_winrate",
            "winrate_vs_best",
            "delta_centipawn",
            "centipawn_vs_best",
            "delta_logit_winrate",
            "logit_winrate_vs_best",
            "delta_expected_score_wdl",
            "expected_score_wdl_vs_best",
            "rank_among_moves",
        }
        require_all_legal = set(legal_moves_set) if legal_moves_set else None

        if fn in {"expected_score_wdl_vs_best", "delta_expected_score_wdl"}:
            _validate_float_map(
                name="reward_model.move_expected_scores_json",
                value_map=move_expected_map,
                require_keys=require_all_legal,
                enforce_01=True,
            )

        if fn in {
            "winrate",
            "delta_winrate",
            "winrate_vs_best",
            "delta_logit_winrate",
            "logit_winrate_vs_best",
        } or fn not in known:
            _validate_float_map(
                name="reward_model.move_values_json",
                value_map=move_winprob_map,
                require_keys=require_all_legal,
                enforce_01=True,
            )

        if fn in {"delta_centipawn", "centipawn_vs_best"}:
            _validate_float_map(
                name="reward_model.move_cps_json",
                value_map=move_cp_map,
                require_keys=require_all_legal,
                enforce_01=False,
            )

        if fn == "rank_among_moves":
            # Prefer CP rank, but fall back to winprob rank if CP is absent.
            if move_cp_map:
                _validate_float_map(
                    name="reward_model.move_cps_json",
                    value_map=move_cp_map,
                    require_keys=require_all_legal,
                    enforce_01=False,
                )
            else:
                _validate_float_map(
                    name="reward_model.move_values_json",
                    value_map=move_winprob_map,
                    require_keys=require_all_legal,
                    enforce_01=True,
                )

        def _best_value(value_map: dict[str, float]) -> float:
            if not value_map:
                return float("nan")
            best = -float("inf")
            # Prefer "best among legal moves" when available; otherwise fall back to the map keys.
            candidates = legal_moves_set if legal_moves_set else set(value_map.keys())
            for mv in candidates:
                v = value_map.get(mv)
                if v is None:
                    continue
                try:
                    best = max(best, float(v))
                except Exception:
                    continue
            return float(best) if best != -float("inf") else float("nan")

        winprob_before = _maybe_parse_float(rm.get("position_win_prob"))
        cp_before = _maybe_parse_float(rm.get("position_cp"))
        expected_before = _maybe_parse_float(rm.get("position_expected_score"))

        if fn in {"delta_winrate", "delta_logit_winrate"}:
            if winprob_before is None or (not math.isfinite(float(winprob_before))):
                _fail("Reward mode requires finite reward_model.position_win_prob for delta_* winrate rewards")

        if fn == "delta_centipawn":
            if cp_before is None or (not math.isfinite(float(cp_before))):
                _fail("Reward mode requires finite reward_model.position_cp for delta_centipawn")

        if fn == "delta_expected_score_wdl":
            if expected_before is None or (not math.isfinite(float(expected_before))):
                _fail("Reward mode requires finite reward_model.position_expected_score for delta_expected_score_wdl")

        score, reward_reason = _compute_move_reward(
            chess_reward_fn=chess_reward_fn,
            pred_move=pred_move,
            winprob_after=float(move_winprob_map.get(pred_move, float("nan"))),
            winprob_best=_best_value(move_winprob_map),
            winprob_before=float(winprob_before) if winprob_before is not None else float("nan"),
            cp_after=float(move_cp_map.get(pred_move, float("nan"))),
            cp_best=_best_value(move_cp_map),
            cp_before=float(cp_before) if cp_before is not None else float("nan"),
            expected_after=float(move_expected_map.get(pred_move, float("nan"))),
            expected_best=_best_value(move_expected_map),
            expected_before=float(expected_before) if expected_before is not None else float("nan"),
            move_winprob_map=move_winprob_map,
            move_cp_map=move_cp_map,
            logit_eps=float(reward_kwargs.get("logit_eps", 1e-6)),
        )
        if not math.isfinite(float(score)):
            _fail(f"Non-finite reward computed: score={score} (check reward payload fields)")

    penalty = -1.0 if penalty_applied else 0.0

    return {
        "score": float(score),
        "total_reward": float(score),
        "format_reward": float(format_reward),
        "pred_move": pred_move,
        "target_move": target_move,
        "gt_uci": gt_uci,
        "mu_source": mu_source,
        "mu_pred": float(mu_pred),
        "mu_target": float(target_mu),
        "in_subset": bool(in_subset),
        "n_legal_moves": int(len(legal_moves_list)),
        "n_considered_moves": int(len(considered_list)),
        "chess_reward_fn": str(chess_reward_fn),
        "reward_reason": str(reward_reason),
        "coverage": float(exact_match_gt),
        "exact_match": float(exact_match_gt),
        "acc": float(acc),
        "penalty": float(penalty),
        "penalty_reason": penalty_reason,
        "penalty_applied": bool(penalty_applied),
    }


def compute_score_batch(
    data_sources: list[Any],
    solution_strs: list[str],
    ground_truths: list[Optional[str]] | None = None,
    extra_infos: list[Optional[Dict[str, Any]]] | None = None,
    **reward_kwargs: Any,
):
    """Batch wrapper compatible with BatchRewardManager.

    Intentionally simple: iterate and reuse `compute_score` with strict semantics.
    """
    n = len(solution_strs)
    # Defensive alignment
    def _get(lst, i, default=None):
        try:
            return lst[i]
        except Exception:
            return default

    out = []
    for i in range(n):
        ds_i = _get(data_sources, i, None)
        sol_i = solution_strs[i]
        gt_i = None
        if ground_truths is not None:
            gt_i = _get(ground_truths, i, None)
        if not gt_i and isinstance(ds_i, dict):
            gt_i = (ds_i.get("ground_truth") or "")
        ei_i = None
        if extra_infos is not None:
            ei_i = _get(extra_infos, i, None)

        res = compute_score(
            data_source=ds_i,
            solution_str=sol_i,
            ground_truth=gt_i or "",
            extra_info=ei_i,
            **reward_kwargs,
        )
        out.append(res)
    return out
