# Example submission (minimal; legacy prompt template)

This folder contains a minimal example submission wrapper around the official starter-kit interface:
you submit a Hugging Face model repo plus a Jinja prompt template file.

> [!IMPORTANT]
> The bundled submission prompt template (`competition/submission/player_agents/chess_rl_chessr1_prompt.jinja`)
> is **legacy / out-of-date relative to this repo’s current restricted-moves (“selection” / verbalized action
> masking) prompts**.
>
> - Current in-repo selection prompt: `recipe/chess/prompt_templates/select_prompt.jinja` (see `restricted_moves.md`)
> - This submission template is kept intact because the competition harness provides only `FEN` +
>   the full `legal_moves_uci_list` (no per-position candidate shortlist / `allowed_moves` list).

## What the evaluation harness calls

Per the starter kit, a submission is made by running `aicrowd submit-model` with:
- a Hugging Face model repo (`--hf-repo`, `--hf-repo-tag`), and
- a Jinja prompt template file (`--prompt_template_path`).

The evaluation system spins up a vLLM server for your HF model and formats each per-move prompt using
your Jinja template.

## Entrypoint

- `competition/submission/aicrowd_submit.sh`

This is a convenience wrapper around `aicrowd submit-model`.

## Prompt wiring (adapter)

The submission template consumes the starter-kit context variables:
- `{{ FEN }}` for the current position
- `{{ legal_moves_uci_list }}` for the legal move list (UCI)

If you want a selection-style submission prompt, the simplest compatible approach is to set:
`allowed_moves == legal_moves` inside the template (i.e., treat the full legal list as the candidate list),
matching how selection datasets are evaluated when `considered_moves_uci_list == legal_moves_uci_list`.
