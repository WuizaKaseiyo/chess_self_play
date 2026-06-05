# chess-agent

RL training for LLM chess agents, built on [verl-agent](https://github.com/langfengQ/verl-agent) / [veRL](https://github.com/volcengine/verl). This fork adds two text environments under `chess_game/`, with PPO launch scripts and HTCondor helpers for GPU clusters.

For the original framework (GiGPO, ALFWorld, WebShop, installation of base `verl-agent`, etc.), see **[README.verl-agent.md](./README.verl-agent.md)**.

---

## Environments

| | **Chess** (`chess/WhiteVsRandom`) | **Chesslesson** (`chesslesson/LichessLearn`) |
|---|---|---|
| Horizon | Multi-step (up to `max_steps` / `max_agent_plies`) | Single-turn (`max_steps=1`) |
| Task | Full game: model plays **White**, Black is random | ~170 verifiable puzzles (piece lessons, tactics, coordinates) |
| Board in prompt | FEN + ASCII unicode board each step | **FEN only** (no ASCII grid) |
| Reward | Shaped: −0.01/step, ±1 terminal, −0.1 illegal | Binary **0/1** (lichess-style judge) |
| Extra deps | `python-chess` | Self-contained under `chesslesson/` |
| Config | `verl/trainer/config/chess_agent.yaml` | `verl/trainer/config/chesslesson_agent.yaml` |

Both use Ray-parallel workers, `<think>` / `<action>` responses, and `env.env_name` in `make_envs()`.

---

## Chess — full game vs random Black

The policy controls **White**. After each White move, the environment plays a **random legal move** for Black.

**Prompt** (`agent_system/environments/prompts/chess.py`): role as White player, current unicode board + FEN + legal UCI hints; optional move history when `env.history_length > 0`.

**Rewards**

| Event | Reward |
|--------|--------|
| Legal move | −0.01 |
| Illegal / empty move | −0.1 (episode continues) |
| White wins | +1.0 |
| Black wins | −1.0 |
| Draw | 0.0 |

**Train**

```bash
source ~/envs/chess/bin/activate   # example venv
cd verl-agent
export PYTHONPATH=$PWD:$PYTHONPATH

bash recipe/hgpo/run_chess_train.sh
# python3 -m verl.trainer.main_chess_agent
```

```yaml
env:
  env_name: chess/WhiteVsRandom
  max_steps: 128
  history_length: 2
  chess:
    max_agent_plies: 100
```

---

## Chesslesson — single-turn puzzles

One question per episode. Tasks from `chess_game/chesslesson/instructions.jsonl` (tactics, piece rules, …) and `coordinates.jsonl` (optional, `include_coord`).

**Prompt** (`build_task_obs` in `chesslesson_envs.py`) — task-first, no lesson branding:

```text
Your task: <goal>
Constraints: <rules when present>
Side to move: White
Move budget: N move(s) or fewer
Target squares: e7, ...
Board (FEN): <position>

# Output format
Reason step by step inside <think></think>.
Then put your final answer inside <action></action>.
```

**Rewards:** 1.0 correct, 0.0 otherwise (bundled judge in `chesslesson/reward.py`).

**Train**

```bash
bash recipe/hgpo/run_chesslesson_train.sh

# HTCondor: 2× H100
bash recipe/hgpo/submit_chesslesson_train.sh 50
```

```yaml
env:
  env_name: chesslesson/LichessLearn
  max_steps: 1
  history_length: 0
  chesslesson:
    include_chess: true
    include_coord: true
```

**Smoke test**

```bash
python recipe/hgpo/test_env.py
bash recipe/hgpo/test_env.sh    # Condor wrapper
```

---

## Quick install

```bash
python3 -m venv ~/envs/chess
source ~/envs/chess/bin/activate
pip install -e .
pip install "chess>=1.10.0"    # only for chess/WhiteVsRandom
```

Prepare placeholder parquet (size/modality only; prompts come from the env):

```bash
python3 -m examples.data_preprocess.prepare --mode text --train_data_size 16 --val_data_size 16
```

---

## Repository layout

```
chess_game/           # env implementations + puzzle data
recipe/hgpo/          # train / eval / Condor scripts
verl/trainer/         # main_chess_agent.py, main_chesslesson_agent.py, yaml configs
agent_system/         # EnvironmentManager + prompt templates
```

---

## When to use which

- **Chess** — long-horizon RL on full games vs a weak opponent.
- **Chesslesson** — dense verifiable supervision for basics and tactics; fast parallel judging.

Switch environments by changing `env.env_name` and the matching launch script or Hydra config.

---

## Upstream

Based on [langfengQ/verl-agent](https://github.com/langfengQ/verl-agent). Sync or compare with `upstream` remote:

```bash
git fetch upstream
```
