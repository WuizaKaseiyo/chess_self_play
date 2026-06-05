from __future__ import annotations

from torch.utils.data import Dataset

import torch


class SelfPlayStubDataset(Dataset):
    """A minimal map-style dataset used to drive the PPO training loop in online play vs engine opponent mode.

    Note: the config key is `data.self_play` for back-compat, but this is **not** true self-play:
    the model plays against an engine opponent (Stockfish) and we stream positions online (both model-to-move
    and engine-to-move turns).

    When `data.self_play.enable=True`, the trainer ignores the dataloader payload and
    generates a fresh in-memory batch of chess positions each step.

    This dataset exists only to satisfy:
      - `create_rl_dataset(...)` / `StatefulDataLoader(...)` construction, and
      - checkpointable dataloader state.
    """

    def __init__(self, data_files, tokenizer, processor, config, max_samples: int = -1):
        _ = (data_files, tokenizer, processor, max_samples)
        self._len = int((config.get("self_play", {}) or {}).get("stub_length", 10_000_000) or 10_000_000)
        if self._len <= 0:
            self._len = 10_000_000

    def __len__(self) -> int:  # noqa: D401
        return int(self._len)

    def __getitem__(self, idx: int) -> dict:
        _ = idx
        # Keep it tensor-only so the default collate_fn stays cheap.
        return {"_self_play_stub": torch.zeros((1,), dtype=torch.long)}
