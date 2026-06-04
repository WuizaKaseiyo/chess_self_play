# W&B Run ID Mapping (EMNLP Paper)

This file tracks the W&B runs used for the EMNLP paper only.

Follow-up experiments not intended for the EMNLP paper, including the small-legal logprob gain-gating runs,
are tracked in the follow-up experiment docs instead of this ledger.

- Qwen2.5-7B-Instruct
  - Ours:
    - fixed dataset: `s0anl08n`
    - online: `5cw6hm16`
  - Baseline:
    - n=8:
      - reward_fn=`expected_score_wdl_vs_best` (also used in our method): `mk76juq4`
      - reward_fn=`winrate_vs_best`: `iu768gtj`
      - reward_fn=`rank_among_moves`: `yu8phknt`
    - n=32:
      - reward_fn=`expected_score_wdl_vs_best` (also used in our method): `dg41tlmo`
      - reward_fn=`winrate_vs_best`: `nxhozx89`
      - reward_fn=`rank_among_moves`: `azs0jkjg`

- Qwen2.5-3B-Instruct
  - Ours:
    - fixed dataset: `h4rhtpg5`
    - online: `h6sqp0z4`
  - Baselines (all reward_fn=`expected_score_wdl_vs_best`):
    - n=8: `82fpo6l0`
    - n=32: `u2cuw56a`
  - Pass@k Baselines:
    - n=16, k=4: `f5guq4ti`

- Qwen3-4B-Instruct-2507
  - Ours:
    - fixed dataset: `TBD`
    - online: `TBD`
  - Baseline:
    - n=8:
      - reward_fn=`expected_score_wdl_vs_best`: `TBD`
