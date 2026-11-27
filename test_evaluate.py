import json
import sys
import tempfile
import unittest
from pathlib import Path

from evaluate import check_holdout, evaluate_opponent, main, model_spec, summarize
from arena import game
from engine import make_board
from rl_model import Policy
from unittest.mock import patch


class EvaluationTests(unittest.TestCase):
    def test_arena_passes_per_bot_options_and_counts_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'eval_good_fixture.py').write_text(
                "class Bot:\n def __init__(self, board, side, model_path):\n  assert model_path == 'candidate'\n"
                " def choose(self, state): return 'U'\n")
            (root / 'eval_bad_fixture.py').write_text(
                "class Bot:\n def __init__(self, board, side): pass\n def choose(self, state): return 'bad'\n")
            sys.path.insert(0, directory)
            try:
                result, _ = game(make_board(100), ['eval_good_fixture', 'eval_bad_fixture'],
                                 bot_options=[{'model_path': 'candidate'}, {}])
            finally:
                sys.path.remove(directory)
            self.assertEqual((result['winner'], result['turns'], result['requests']), (0, 9, 10))
            self.assertEqual(result['errors'], [0, 10])

    def test_swaps_identity_and_passes_separate_options(self):
        calls = []
        def fake_game(board, names, bot_options):
            calls.append((board, names, bot_options))
            # Side 0 always wins; candidate changes side in the second game.
            return (dict(kind='normal', winner=0, turns=10, requests=10,
                         score=[20, 9], leads=[7, 1], us=[100, 200],
                         peak_us=[30, 40], errors=[0, 1], closes=[2, 1], scraps=[0, 3]), [])
        candidate = dict(module='bot_rl', options={'model_path': 'candidate.pt'})
        opponent = dict(module='bot_rl', options={'model_path': 'history.pt'})
        report = evaluate_opponent(candidate, opponent, [123], fake_game, lambda seed: seed)
        self.assertEqual([[opts['model_path'] for opts in call[2]] for call in calls], [
            ['candidate.pt', 'history.pt'], ['history.pt', 'candidate.pt']])
        self.assertNotEqual(calls[0][2][0]['rng_seed'], calls[0][2][1]['rng_seed'])
        self.assertEqual(report['games'][0]['rng_seeds'],
                         [opts['rng_seed'] for opts in calls[0][2]])
        self.assertEqual([r['candidate_result'] for r in report['games']], ['win', 'loss'])
        self.assertEqual([r['candidate']['score'] for r in report['games']], [20, 9])
        self.assertEqual(report['summary']['win_rate'], 0.5)
        self.assertEqual(report['summary']['candidate']['mean_response_us'], 15)
        self.assertEqual(report['summary']['candidate']['peak_response_us'], 40)
        self.assertEqual(report['summary']['candidate']['errors'], 1)

    def test_double_loss_pending_and_empty_timing(self):
        participant = dict(requests=0, total_us=0, peak_us=0, errors=0, scraps=0)
        rows = [dict(candidate_result=x, candidate=participant, opponent=participant)
                for x in ('win', 'loss', 'double_loss', 'pending')]
        result = summarize(rows)
        self.assertEqual((result['wins'], result['losses'], result['double_losses'], result['pending']), (1, 2, 1, 1))
        self.assertEqual(result['win_rate'], .25)
        self.assertIsNone(result['candidate']['mean_response_us'])

    def test_holdout_ranges_and_missing_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory, 'model.pt')
            Policy().save(model)
            spec = model_spec(model)
            self.assertIn('unknown', check_holdout(spec, [9]))
            model.with_suffix('.json').write_text(json.dumps({'train_seed_ranges': [[10, 20], [30, 32]]}))
            self.assertIsNone(check_holdout(spec, [9, 20, 32]))
            with self.assertRaisesRegex(ValueError, 'overlap'):
                check_holdout(spec, [19])
            with self.assertRaisesRegex(ValueError, 'overlap'):
                check_holdout(spec, [31])
            model.with_suffix('.json').write_text(json.dumps({'train_seed_ranges': [[10, 10]]}))
            with self.assertRaisesRegex(ValueError, 'Invalid training metadata'):
                check_holdout(spec, [9])

    def test_bad_model_and_protected_output_rejected_before_games(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory, 'broken.pt')
            model.write_bytes(b'not a model')
            with patch('evaluate.evaluate_opponent') as play:
                with self.assertRaises(SystemExit):
                    main(['--model', str(model), '--seeds', '100', '--output', str(model)])
                with self.assertRaises(SystemExit):
                    main(['--model', str(model), '--seeds', '100', '--output', str(model.with_suffix('.json'))])
                with self.assertRaises(SystemExit):
                    main(['--model', str(model), '--seeds', '100', '--output', str(Path(directory, 'result.json'))])
                play.assert_not_called()
            self.assertEqual(model.read_bytes(), b'not a model')


if __name__ == '__main__':
    unittest.main()
