"""Training orchestration checks; no episodes or optimizer updates are run."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import train
from engine import State, make_board


class TrainingOrchestrationTests(unittest.TestCase):
    def test_reinforce_computes_gradients_without_updating_weights(self):
        policy = train.Policy(seed=7)
        board = make_board(8)
        state = State.initial(board)
        features = policy.encode(board, state, 0)
        samples = [(features, 0, 1), (features, 1, 2)]
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
        before = [p.detach().clone() for p in policy.parameters()]
        with patch.object(optimizer, 'step') as step:
            train.reinforce(policy, samples, [1.0, -0.5], 0.001, 1.0, 1.0, optimizer=optimizer)
        step.assert_called_once()
        for old, param in zip(before, policy.parameters()):
            self.assertTrue(torch.equal(old, param.detach()))
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in policy.parameters()), 0)

    def test_episode_shares_observation_but_not_mutable_state(self):
        test = self

        class FixedPolicy:
            observation = None

            def encode(self, board, state, side):
                self.observation = state.copy()
                return np.zeros(1)

            def forward(self, features, heading):
                return np.array([0.0, 1.0, 0.0, 0.0]), np.zeros(1)

        class FixedOpponent:
            calls = 0

            def choose(self, state):
                test.assertEqual(state, policy.observation)
                test.assertEqual(state.turn, self.calls)
                self.calls += 1
                # A bot's accidental mutation must not alter the actual match.
                state.points[0] = -10000
                return 'L'

        policy = FixedPolicy()
        opponent = FixedOpponent()
        samples, rewards, state, _ = train.episode(
            policy, opponent, make_board(1), 0, np.random.default_rng(1), 0.0)
        self.assertEqual(opponent.calls, 600)
        self.assertEqual(state.turn, 600)
        self.assertEqual(len(samples), 600)
        self.assertEqual(len(rewards), 600)
        self.assertNotEqual(state.points[0], -10000)

    def test_checkpoint_resume_and_protect_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / 'first'
            second = Path(tmp) / 'second'
            state = State.initial(make_board(8))
            fake_episode = ([], [], state, 0)
            with patch.object(train, 'make_board', return_value=object()), \
                    patch.object(train, 'episode', return_value=fake_episode) as play, \
                    patch.object(train, 'reinforce', return_value=0.0) as update, \
                    contextlib.redirect_stdout(io.StringIO()):
                train.main(['--episodes', '2', '--checkpoint-every', '1', '--output-dir', str(first),
                            '--train-seed-start', '10', '--train-seed-count', '5', '--bot2-prob', '0'])
                self.assertEqual(play.call_count, 2)
                self.assertEqual(update.call_count, 2)
                self.assertTrue((first / 'policy_0000001.pt').is_file())
                before = (first / 'latest.pt').read_bytes()
                metadata = json.loads((first / 'latest.json').read_text())
                self.assertEqual(metadata['train_seed_ranges'], [[10, 15]])
                with self.assertRaises(SystemExit):
                    train.main(['--episodes', '1', '--output-dir', str(first)])
                self.assertEqual((first / 'latest.pt').read_bytes(), before)
                train.main(['--episodes', '1', '--output-dir', str(second), '--bot2-prob', '0',
                            '--resume', str(first / 'latest.pt'), '--train-seed-start', '30',
                            '--train-seed-count', '4'])
                resumed = json.loads((second / 'latest.json').read_text())
                self.assertEqual(resumed['train_seed_ranges'], [[10, 15], [30, 34]])
                train.Policy.load(second / 'latest.pt')

    def test_invalid_parameters(self):
        invalid = [('--episodes', '0'), ('--train-seed-count', '0'), ('--seed', '-1'),
                   ('--gamma', 'nan'), ('--learning-rate', 'inf'), ('--bot2-prob', '1.1')]
        with contextlib.redirect_stderr(io.StringIO()):
            for flag, value in invalid:
                with self.subTest(flag=flag), self.assertRaises(SystemExit):
                    train.parse_args([flag, value])

    def test_normal_end_tiebreakers(self):
        state = State.initial(make_board(3))
        state.points = [10, 9]
        state.leads = [0, 50]
        self.assertEqual(train.game_result(state, 0), 1)
        state.points = [10, 10]
        self.assertEqual(train.game_result(state, 0), -1)
        state.leads = [50, 50]
        self.assertEqual(train.game_result(state, 0), 0)
        self.assertEqual(train.game_result(state, 1), 0)


if __name__ == '__main__':
    unittest.main()
