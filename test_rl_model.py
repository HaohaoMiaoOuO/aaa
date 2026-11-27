"""Policy inference tests. No optimization or training is run here."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from bot_rl import Bot
from engine import DIRS, State, bit, make_board
from rl_model import FEATURES, Policy, encode


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.board = make_board(7)

    def setUp(self):
        self.state = State.initial(self.board)
        self.policy = Policy(seed=12)

    def test_observation_tracks_public_state(self):
        original = encode(self.board, self.state, 0)
        self.assertEqual(original.shape, (FEATURES,))
        self.assertEqual(original.dtype, np.float32)
        self.assertTrue(np.isfinite(original).all())
        # A local trail change must be visible at exact resolution.
        changed = self.state.copy()
        changed.trail[0] = bit(changed.pos[0] + 1)
        self.assertFalse(np.array_equal(original, encode(self.board, changed, 0)))
        # These are separate global inputs, rather than only score difference.
        for name, value in [('heading', 1), ('down', 3), ('points', 19), ('leads', 7)]:
            changed = self.state.copy()
            getattr(changed, name)[1] = value
            self.assertFalse(np.array_equal(original, encode(self.board, changed, 0)), name)
        changed = self.state.copy()
        changed.turn = 599
        self.assertFalse(np.array_equal(original, encode(self.board, changed, 0)))

    def test_only_reverse_is_masked_for_each_heading(self):
        for heading in range(4):
            self.state.heading[0] = heading
            probs = self.policy.probabilities(self.board, self.state, 0)
            self.assertEqual(probs[(heading + 2) % 4], 0)
            self.assertTrue(np.all(np.delete(probs, (heading + 2) % 4) > 0))
            self.assertAlmostEqual(float(probs.sum()), 1.0, places=6)

    def test_roundtrip_and_seeded_bot_sampling(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'policy.pt'
            self.policy.save(path)
            loaded = Policy.load(path)
            np.testing.assert_array_equal(self.policy.probabilities(self.board, self.state, 0),
                                          loaded.probabilities(self.board, self.state, 0))
            first = Bot(self.board, 0, model_path=path, rng_seed=17)
            second = Bot(self.board, 0, model_path=path, rng_seed=17)
            a = [first.choose(self.state) for _ in range(40)]
            b = [second.choose(self.state) for _ in range(40)]
            self.assertEqual(a, b)
            self.assertNotIn(DIRS[(self.state.heading[0] + 2) % 4], a)
            self.assertGreater(len(set(a)), 1)

    def test_bad_checkpoints_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'policy.pt'
            weights = self.policy.state_dict()
            for key, value in [('version', 999), ('layer1.weight', torch.zeros((2, 2))),
                               ('layer2.bias', torch.full((4,), float('nan')))]:
                bad = {'version': 1, 'state_dict': dict(weights)}
                if key == 'version':
                    bad[key] = value
                else:
                    bad['state_dict'][key] = value
                torch.save(bad, path)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    Policy.load(path)

    def test_legacy_numpy_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'legacy.npz'
            weights = self.policy.state_dict()
            np.savez(path, version=np.array(1),
                     w1=weights['layer1.weight'].numpy().T, b1=weights['layer1.bias'].numpy(),
                     w2=weights['layer2.weight'].numpy().T, b2=weights['layer2.bias'].numpy())
            loaded = Policy.load(path)
            np.testing.assert_allclose(self.policy.probabilities(self.board, self.state, 0),
                                       loaded.probabilities(self.board, self.state, 0), rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
