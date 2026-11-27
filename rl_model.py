"""Compact PyTorch policy shared by training and local arena inference.

Actions use engine.DIRS order. Features are from the acting side's perspective,
while coordinates and directions remain in the board's absolute orientation.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

from engine import N, SIZE, bit

CHANNELS = 11
GRID = 9
LOCAL = 9
GLOBAL_FEATURES = 23
FEATURES = CHANNELS * (GRID * GRID + LOCAL * LOCAL) + GLOBAL_FEATURES
HIDDEN = 64
FORMAT_VERSION = 1


def _mask_array(mask):
    packed = np.frombuffer(mask.to_bytes((SIZE + 7) // 8, "little"), dtype=np.uint8)
    return np.unpackbits(packed, bitorder="little")[:SIZE].reshape(N, N).astype(np.float32)


@lru_cache(maxsize=64)
def _board_arrays(board, side):
    return tuple(_mask_array(mask) for mask in (
        board.defects, board.high, board.homes[side], board.homes[1-side], board.fast
    ))


def encode(board, state, side):
    """Encode a compressed observation of the public game state as float32.

    Eleven channels cover terrain, both homes, speed zones, both territories,
    trails and positions. The whole board is lossily pooled into 5x5 blocks; a separate
    9x9 window retains exact local detail. Global features include each heading,
    downtime, score, leading turns and exact positions and spawn locations.
    """
    other = 1 - side
    masks = (state.owned[side], state.owned[other], state.trail[side], state.trail[other],
             bit(state.pos[side]) if state.pos[side] >= 0 else 0,
             bit(state.pos[other]) if state.pos[other] >= 0 else 0)
    layers = np.stack((*_board_arrays(board, side), *(_mask_array(m) for m in masks)))
    pooled = np.pad(layers, ((0, 0), (0, 4), (0, 4))).reshape(CHANNELS, GRID, 5, GRID, 5).mean(axis=(2, 4))
    pos = state.pos[side]
    local = np.zeros((CHANNELS, LOCAL, LOCAL), dtype=np.float32)
    # Treat off-board cells as defects. A down player has no local field.
    if pos >= 0:
        x, y = pos % N, pos // N
        local[0].fill(1)
        x0, x1 = max(0, x-4), min(N, x+5)
        y0, y1 = max(0, y-4), min(N, y+5)
        local[:, y0-y+4:y1-y+4, x0-x+4:x1-x+4] = layers[:, y0:y1, x0:x1]
    heading = np.zeros(8, dtype=np.float32)
    heading[state.heading[side]] = 1
    heading[4 + state.heading[other]] = 1
    def coordinates(cell):
        return (-1.0, -1.0) if cell < 0 else (cell % N / (N-1), cell // N / (N-1))
    glob = np.asarray((*heading, state.down[side]/5, state.down[other]/5,
                       state.points[side]/(SIZE*3), state.points[other]/(SIZE*3),
                       state.leads[side]/600, state.leads[other]/600, state.turn/600,
                       *coordinates(state.pos[side]), *coordinates(state.pos[other]),
                       *coordinates(board.spawns[side]), *coordinates(board.spawns[other])), dtype=np.float32)
    return np.concatenate((pooled.ravel(), local.ravel(), glob))


class Policy(nn.Module):
    """A single-hidden-layer tanh policy with a masked action distribution."""
    def __init__(self, seed=0):
        super().__init__()
        self.layer1 = nn.Linear(FEATURES, HIDDEN)
        self.layer2 = nn.Linear(HIDDEN, 4)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            self.layer1.weight.copy_(torch.randn(self.layer1.weight.shape, generator=generator) / np.sqrt(FEATURES))
            self.layer1.bias.zero_()
            self.layer2.weight.copy_(torch.randn(self.layer2.weight.shape, generator=generator) / np.sqrt(HIDDEN))
            self.layer2.bias.zero_()

    @staticmethod
    def encode(board, state, side):
        return encode(board, state, side)

    def forward(self, features, heading):
        """Return (action probabilities, hidden activations) for one observation."""
        features = torch.as_tensor(features, dtype=torch.float32, device=self.layer1.weight.device)
        hidden = torch.tanh(self.layer1(features))
        logits = self.layer2(hidden)
        # The referee treats a reverse command as maintaining the heading.
        headings = torch.as_tensor(heading, device=logits.device, dtype=torch.long)
        reverse = torch.nn.functional.one_hot((headings + 2) % 4, num_classes=4).bool()
        logits = logits.masked_fill(reverse, float('-inf'))
        return torch.softmax(logits, dim=-1), hidden

    def probabilities(self, board, state, side):
        with torch.no_grad():
            return self.forward(encode(board, state, side), state.heading[side])[0].cpu().numpy()

    def save(self, path):
        """Write a versioned CPU state dict, loadable with weights_only=True."""
        weights = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        torch.save({'version': FORMAT_VERSION, 'state_dict': weights}, Path(path))

    @classmethod
    def load(cls, path):
        try:
            path = Path(path)
            if path.suffix.lower() == '.npz':
                # Read the previous NumPy-only checkpoints without pickle.
                with np.load(path, allow_pickle=False) as old:
                    if int(old['version']) != FORMAT_VERSION:
                        raise ValueError('Unsupported legacy policy format version')
                    data = {'version': FORMAT_VERSION, 'state_dict': {
                        'layer1.weight': torch.from_numpy(np.asarray(old['w1'], dtype=np.float32).T.copy()),
                        'layer1.bias': torch.from_numpy(np.asarray(old['b1'], dtype=np.float32).copy()),
                        'layer2.weight': torch.from_numpy(np.asarray(old['w2'], dtype=np.float32).T.copy()),
                        'layer2.bias': torch.from_numpy(np.asarray(old['b2'], dtype=np.float32).copy()),
                    }}
            else:
                data = torch.load(path, map_location='cpu', weights_only=True)
            if not isinstance(data, dict) or data.get('version') != FORMAT_VERSION:
                raise ValueError('Unsupported policy format version')
            weights = data['state_dict']
            policy = cls()
            if not isinstance(weights, dict) or set(weights) != set(policy.state_dict()):
                raise ValueError('Invalid policy weight keys')
            for key, reference in policy.state_dict().items():
                value = weights[key]
                if not torch.is_tensor(value) or value.shape != reference.shape or not torch.isfinite(value).all():
                    raise ValueError(f'Invalid policy weights: {key}')
            policy.load_state_dict(weights, strict=True)
            return policy
        except (KeyError, TypeError, RuntimeError, EOFError) as exc:
            raise ValueError(f'Invalid PyTorch checkpoint: {exc}') from exc
