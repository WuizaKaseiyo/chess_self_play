# ♟️ Global Chess Challenge 2025

The Global Chess Challenge uses chess as a **clean, rigorous testbed** for studying reasoning in language models. While classical engines like Stockfish use deep search and heuristics, LLMs often struggle with basic legality and tactical consistency. This challenge treats that gap as an opportunity to understand how **structured reasoning** can be learned and evaluated inside language models.

This challenge frames chess as a **text-only problem**. Models receive a symbolic description of the position and must decide what to play without access to boards, search procedures, or external tools.

---

## 💻 What is the Global Chess Challenge?

You will build a **text-only chess agent** that does two things for every position:

1. **Outputs exactly one legal move** (in UCI format).
2. **Outputs a one-sentence rationale** explaining the idea behind the move.

### Technical Constraints

* **Standalone Inference:** Submissions must behave as a standalone language model; move decisions must be made **solely via token-level prediction**.
* **No External Tools:** At inference time, you cannot use function calling, retrieval, heuristic search procedures, or embedded chess engines.
* **Training Freedom:** You **are** allowed to use open datasets, offline preprocessing, finetuning, and RL using Stockfish as a verifier/labeler **during training only**.

---

## 📤 Submission Format & Execution

Participants submit a **language model via a gated Hugging Face repository**. Organizers pull and run the model on their own controlled infrastructure (**AWS Trainium trn1.2xlarge instances**).

### Model Size Restriction

* Models must have a total parameter count of **strictly fewer than 8,000,000,000 (8B)**.
* Parameter count is determined from model weights at inference time (excluding optimizer state).

### Model Input

For every turn, the agent receives a prompt including:

* **Position:** As a FEN string (e.g., `r1bk3r/p2pBpNp/n4n2/1p1NP2P/6P1/3P4/P1P1K3/q5b1`)
* **Side to move:** (White / Black)
* **List of legal moves:** In UCI format (e.g., `e2e4`, `g1f3`)

### Model Output

The agent must return:

* **One UCI move:** Wrapped in `<uci_move>...</uci_move>` tags.
* **One-sentence rationale:** Wrapped in `<rationale>...</rationale>` tags (required but not scored).

> [!IMPORTANT]
> Evaluation is based **exclusively** on the UCI move inside the tags. If a valid move is not provided after three retries, the model is treated as having **resigned**.

---

## 📊 Evaluation & Metrics

### Round 1 & 2: Baseline Evaluation

Submissions play against **fixed Stockfish opponents** (50 games vs. Skill 0, Depth 1 and 50 games vs. Skill 0, Depth 5).

* **Primary Metric:** Average Centipawn Loss (ACPL). Lower is better. Measured using Stockfish Level 20 (Depth 20).
* **Secondary Metric:** Win Rate.
* **Eligibility:** Only models with an ACPL lower than the official baseline model advance to the Final Tournament.

### Final Tournament: Swiss-Style

Eligible submissions compete in a **Swiss-system tournament**.

* **Scoring:** Win = 1 pt, Draw = 0.5 pts, Loss = 0 pts.
* **Note:** ACPL is **not** used for ranking in the final tournament.

---

## 💡 Suggested Approaches

1. **Data-Centric Finetuning (SFT):** Train models to map text positions to high-quality moves and explanations using open corpora (Lichess) or offline Stockfish annotations.
2. **RLVR (Reinforcement Learning with Verifiable Rewards):** Use Stockfish as a verifier **during training** to generate rewards (legality + evaluation improvement) and optimize with PPO or GRPO.

---

## 🔑 Resources

* **Starter Kit:** [github.com/AIcrowd/global-chess-challenge-2025-starter-kit](https://github.com/AIcrowd/global-chess-challenge-2025-starter-kit)
* **Supported Backends:** Documentation for [Neuron and vLLM tuning](https://www.google.com/search?q=https://github.com/AIcrowd/global-chess-challenge-2025-starter-kit/blob/master/docs/neuron-and-vllm-tuning.md%23supported-model-types-backends).