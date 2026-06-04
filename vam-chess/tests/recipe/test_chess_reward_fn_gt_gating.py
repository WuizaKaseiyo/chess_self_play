from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_reward_module():
    repo_root = Path(__file__).resolve().parents[2]
    reward_path = repo_root / "recipe" / "chess" / "reward_fn.py"
    spec = importlib.util.spec_from_file_location("chess_reward_fn", reward_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reward_mod():
    return _load_reward_module()


def _mk_solution(move: str) -> str:
    return f"<reason>test</reason><uci_move>{move}</uci_move>"


def _mk_reward_model(*, include_expected: bool = True):
    rm = {
        "fen": "8/8/8/8/8/8/8/8 w - - 0 1",
        "legal_moves_uci": ["e2e4", "d2d4", "a2a3"],
        "considered_moves_uci": ["e2e4", "d2d4", "a2a3"],
        "ground_truth": "e2e4",
        # Provide winprob map as a fallback for μ + other shaping modes.
        "move_values_json": {"e2e4": 0.61, "d2d4": 0.60, "a2a3": 0.05},
    }
    if include_expected:
        rm["move_expected_scores_json"] = {"e2e4": 0.70, "d2d4": 0.69, "a2a3": 0.20}
    return rm


def test_penalties_remain_strict(reward_mod):
    rm = _mk_reward_model()

    res = reward_mod.compute_score(
        data_source=rm,
        solution_str="no tags at all",
        ground_truth=rm["ground_truth"],
        chess_reward_fn="gt_gated",
    )
    assert res["penalty_applied"] is True
    assert res["penalty_reason"] == "format_error"
    assert res["score"] == -1.0

    res = reward_mod.compute_score(
        data_source=rm,
        solution_str=_mk_solution("a1a2"),  # valid UCI, but not in considered_moves_uci
        ground_truth=rm["ground_truth"],
        chess_reward_fn="gt_gated",
    )
    assert res["penalty_applied"] is True
    assert res["penalty_reason"] == "out_of_subset"
    assert res["score"] == -1.0

    # Multiple <uci_move> tags are still a hard format error.
    res = reward_mod.compute_score(
        data_source=rm,
        solution_str="<reason>t</reason><uci_move>e2e4</uci_move><uci_move>d2d4</uci_move>",
        ground_truth=rm["ground_truth"],
        chess_reward_fn="gt_gated",
    )
    assert res["penalty_applied"] is True
    assert res["penalty_reason"] == "format_error"
    assert res["score"] == -1.0

    # Extra text after the final move tag is still a hard format error.
    res = reward_mod.compute_score(
        data_source=rm,
        solution_str="<reason>t</reason><uci_move>e2e4</uci_move> EXTRA",
        ground_truth=rm["ground_truth"],
        chess_reward_fn="gt_gated",
    )
    assert res["penalty_applied"] is True
    assert res["penalty_reason"] == "format_error"
    assert res["format_reward"] == 0.0
    assert res["score"] == -1.0

    # The old <think> tags are not accepted when the prompt opened <reason>.
    res = reward_mod.compute_score(
        data_source=rm,
        solution_str="test</think><uci_move>e2e4</uci_move>",
        ground_truth=rm["ground_truth"],
        extra_info={"prompt_text": "Assistant:\n<reason>\n"},
        chess_reward_fn="gt_gated",
    )
    assert res["penalty_applied"] is True
    assert res["penalty_reason"] == "format_error"
    assert res["score"] == -1.0


def test_relaxed_format_allows_text_between_reason_and_move_tags(reward_mod):
    rm = _mk_reward_model()

    res = reward_mod.compute_score(
        data_source=rm,
        solution_str="<reason>test</reason>\nI choose from the list.\n<uci_move>e2e4</uci_move>",
        ground_truth=rm["ground_truth"],
        chess_reward_fn="gt_gated",
    )
    assert res["penalty_applied"] is False
    assert res["format_reward"] == 1.0
    assert res["score"] == 1.0

    res = reward_mod.compute_score(
        data_source=rm,
        solution_str="<reason>test</reason><uci_move> e2e4 </uci_move>",
        ground_truth=rm["ground_truth"],
        chess_reward_fn="gt_gated",
    )
    assert res["penalty_applied"] is True
    assert res["penalty_reason"] == "format_error"
    assert res["format_reward"] == 0.0
    assert res["score"] == -1.0


def test_open_reason_prompt_allows_response_to_start_inside_reasoning(reward_mod):
    rm = _mk_reward_model()
    prompt_info = {"prompt_text": "Assistant:\n<reason>\n"}

    res = reward_mod.compute_score(
        data_source=rm,
        solution_str="test</reason>\n<uci_move>e2e4</uci_move>",
        ground_truth=rm["ground_truth"],
        extra_info=prompt_info,
        chess_reward_fn="gt_gated",
    )
    assert res["penalty_applied"] is False
    assert res["format_reward"] == 1.0
    assert res["score"] == 1.0

    # The response-fragment regex is enabled only when the decoded prompt
    # actually ends with an open <reason>.
    res = reward_mod.compute_score(
        data_source=rm,
        solution_str="test</reason><uci_move>e2e4</uci_move>",
        ground_truth=rm["ground_truth"],
        extra_info={"prompt_text": "Assistant:"},
        chess_reward_fn="gt_gated",
    )
    assert res["penalty_applied"] is True
    assert res["penalty_reason"] == "format_error"
    assert res["format_reward"] == 0.0
    assert res["score"] == -1.0

    # The model still has to close </reason>; a bare move tag is not enough.
    res = reward_mod.compute_score(
        data_source=rm,
        solution_str="<uci_move>e2e4</uci_move>",
        ground_truth=rm["ground_truth"],
        extra_info=prompt_info,
        chess_reward_fn="gt_gated",
    )
    assert res["penalty_applied"] is True
    assert res["penalty_reason"] == "format_error"
    assert res["format_reward"] == 0.0
    assert res["score"] == -1.0


def test_legacy_think_contract_still_works_for_older_prompts(reward_mod):
    rm = _mk_reward_model()

    res = reward_mod.compute_score(
        data_source=rm,
        solution_str="<think>test</think><uci_move>e2e4</uci_move>",
        ground_truth=rm["ground_truth"],
        extra_info={"prompt_text": "Output <think>...</think><uci_move>...</uci_move>."},
        chess_reward_fn="gt_gated",
    )
    assert res["penalty_applied"] is False
    assert res["format_reward"] == 1.0
    assert res["score"] == 1.0


def test_gt_gated_hit_and_miss(reward_mod):
    rm = _mk_reward_model()

    hit = reward_mod.compute_score(
        data_source=rm,
        solution_str=_mk_solution("e2e4"),
        ground_truth=rm["ground_truth"],
        chess_reward_fn="gt_gated",
    )
    assert hit["penalty_applied"] is False
    assert hit["score"] == 1.0
    assert hit["reward_reason"] == "gt_gated:hit"

    miss = reward_mod.compute_score(
        data_source=rm,
        solution_str=_mk_solution("d2d4"),
        ground_truth=rm["ground_truth"],
        chess_reward_fn="gt_gated",
    )
    assert miss["penalty_applied"] is False
    assert miss["score"] == 0.0
    assert miss["reward_reason"] == "gt_gated:miss"


def test_gt_expected_threshold(reward_mod):
    rm = _mk_reward_model(include_expected=True)

    near = reward_mod.compute_score(
        data_source=rm,
        solution_str=_mk_solution("d2d4"),
        ground_truth=rm["ground_truth"],
        chess_reward_fn="gt_expected_threshold",
        gt_expected_score_diff_threshold=0.02,
    )
    assert near["penalty_applied"] is False
    assert near["score"] == pytest.approx(0.69, abs=1e-9)
    assert near["reward_reason"].startswith("gt_expected_threshold:hit(")

    far = reward_mod.compute_score(
        data_source=rm,
        solution_str=_mk_solution("a2a3"),
        ground_truth=rm["ground_truth"],
        chess_reward_fn="gt_expected_threshold",
        gt_expected_score_diff_threshold=0.02,
    )
    assert far["penalty_applied"] is False
    assert far["score"] == 0.0
    assert far["reward_reason"].startswith("gt_expected_threshold:miss(")


def test_gt_expected_threshold_missing_expected_map_is_safe(reward_mod):
    rm = _mk_reward_model(include_expected=False)

    res = reward_mod.compute_score(
        data_source=rm,
        solution_str=_mk_solution("e2e4"),
        ground_truth=rm["ground_truth"],
        chess_reward_fn="gt_expected_threshold",
        gt_expected_score_diff_threshold=0.1,
    )
    assert res["penalty_applied"] is False
    assert res["score"] == 0.0
    assert res["reward_reason"] == "gt_expected_threshold:missing_expected_map"
