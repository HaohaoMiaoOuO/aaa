"""Inference adapter for a trained policy. Set MASK_ARENA_MODEL to its .pt file."""
import os

import numpy as np
import torch

from engine import DIRS
from rl_model import Policy


class Bot:
    def __init__(self, board, side, model_path=None, rng_seed=None):
        # Tiny network: avoid thread-pool overhead inside the 200 ms reply limit.
        torch.set_num_threads(1)
        self.board, self.side = board, side
        path = model_path or os.environ.get("MASK_ARENA_MODEL")
        if not path:
            raise ValueError("Set MASK_ARENA_MODEL to a trained .pt policy file")
        self.policy = Policy.load(path)
        self.rng = np.random.default_rng(rng_seed)

    def choose(self, state):
        probs = self.policy.probabilities(self.board, state, self.side).astype(np.float64)
        probs /= probs.sum()
        return DIRS[int(self.rng.choice(len(DIRS), p=probs))]
