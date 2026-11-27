"""Evaluate a saved policy against Bot2 and optional historical policies.

This uses the timed, process-isolated arena referee. Each map is played twice,
with the candidate and opponent swapping sides. Pass holdout seeds explicitly.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from arena import game
from engine import make_board


def summarize(rows):
    """Summarize outcomes relative to the candidate, keeping unresolved games."""
    count = len(rows)
    wins = sum(row['candidate_result'] == 'win' for row in rows)
    losses = sum(row['candidate_result'] in ('loss', 'double_loss') for row in rows)
    doubles = sum(row['candidate_result'] == 'double_loss' for row in rows)
    pending = sum(row['candidate_result'] == 'pending' for row in rows)
    summary = dict(games=count, wins=wins, losses=losses, double_losses=doubles,
                   pending=pending, win_rate=wins / count if count else None)
    for who in ('candidate', 'opponent'):
        requests = sum(row[who]['requests'] for row in rows)
        total_us = sum(row[who]['total_us'] for row in rows)
        summary[who] = dict(
            requests=requests,
            mean_response_us=total_us / requests if requests else None,
            peak_response_us=max((row[who]['peak_us'] for row in rows), default=0),
            errors=sum(row[who]['errors'] for row in rows),
            scraps=sum(row[who]['scraps'] for row in rows),
        )
    return summary


def evaluate_opponent(candidate, opponent, seeds, run_game=game, board_factory=make_board,
                      rng_seed=0):
    """Play the candidate on both sides for every requested map seed."""
    rows = []
    for seed in seeds:
        board = board_factory(seed)
        for candidate_side in (0, 1):
            specs = [None, None]
            specs[candidate_side] = candidate
            specs[1 - candidate_side] = opponent
            options = []
            for side, spec in enumerate(specs):
                opts = dict(spec['options'])
                if 'model_path' in opts:
                    key = f'{rng_seed}:{seed}:{candidate_side}:{side}'.encode()
                    opts['rng_seed'] = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big')
                options.append(opts)
            result, _ = run_game(board, [s['module'] for s in specs],
                                 bot_options=options)
            winner = result['winner']
            outcome = ('double_loss' if result['kind'] == 'double_loss' else
                       'pending' if winner is None else
                       'win' if winner == candidate_side else 'loss')
            row = dict(seed=seed, candidate_side=candidate_side, kind=result['kind'],
                       turns=result['turns'], candidate_result=outcome,
                       rng_seeds=[opts.get('rng_seed') for opts in options])
            for who, i in (('candidate', candidate_side), ('opponent', 1-candidate_side)):
                row[who] = dict(score=result['score'][i], leads=result['leads'][i],
                                total_us=result['us'][i], peak_us=result['peak_us'][i],
                                errors=result['errors'][i], closes=result['closes'][i],
                                scraps=result['scraps'][i], requests=result['requests'])
            rows.append(row)
            print(f"seed {seed}, candidate side {candidate_side}: {outcome}; "
                  f"score {row['candidate']['score']}:{row['opponent']['score']}; "
                  f"errors {row['candidate']['errors']}:{row['opponent']['errors']}", flush=True)
    return dict(opponent=opponent, summary=summarize(rows), games=rows)


def model_spec(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'Model file does not exist: {path}')
    try:
        from rl_model import Policy
        Policy.load(path)
    except Exception as exc:
        raise ValueError(f'Invalid model {path}: {exc}') from exc
    return dict(module='bot_rl', options=dict(model_path=str(path)))


def check_holdout(spec, seeds):
    """Return a warning for unknown provenance; reject known overlap."""
    model = Path(spec['options']['model_path'])
    sidecar = model.with_suffix('.json')
    if not sidecar.is_file():
        return f'No training metadata for {model}; holdout independence is unknown.'
    try:
        metadata = json.loads(sidecar.read_text(encoding='utf-8'))
        ranges = metadata.get('train_seed_ranges')
        if ranges is None and 'train_seed_start' in metadata and 'train_seed_count' in metadata:
            start = metadata['train_seed_start']
            ranges = [[start, start + metadata['train_seed_count']]]
        if not ranges:
            return f'No training seed ranges in {sidecar}; holdout independence is unknown.'
        if any(not isinstance(r, list) or len(r) != 2 or
               any(type(v) is not int for v in r) or r[0] >= r[1] for r in ranges):
            raise ValueError('invalid train_seed_ranges')
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f'Invalid training metadata in {sidecar}: {exc}') from exc
    overlap = [s for s in seeds if any(start <= s < end for start, end in ranges)]
    if overlap:
        raise ValueError(f'Holdout seeds overlap training data for {model}: {overlap}')
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Candidate .pt model (legacy .npz also supported)')
    parser.add_argument('--opponent', action='append', default=[], metavar='MODULE',
                        help='Bot module to evaluate against (default: bot2)')
    parser.add_argument('--opponent-model', action='append', default=[], metavar='PATH',
                        help='Additional historical model; can be repeated')
    parser.add_argument('--seeds', type=int, nargs='+', required=True,
                        help='Explicit holdout map seeds, disjoint from training seeds')
    parser.add_argument('--rng-seed', type=int, default=0, help='Seed for policy sampling (default: 0)')
    parser.add_argument('--output', default='evaluation.json', help='JSON output path')
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Holdout seeds must be unique')
    try:
        output = Path(args.output).expanduser().resolve()
        models = [Path(p).expanduser().resolve() for p in [args.model, *args.opponent_model]]
        if output in {p for model in models for p in (model, model.with_suffix('.json'))}:
            raise ValueError('Output must not overwrite a model or its metadata sidecar')
        candidate = model_spec(args.model)
        opponents = [dict(module=x, options={}) for x in (args.opponent or ['bot2'])]
        opponents.extend(model_spec(p) for p in args.opponent_model)
        warnings = [warning for spec in [candidate] + [o for o in opponents if 'model_path' in o['options']]
                    if (warning := check_holdout(spec, args.seeds))]
    except ValueError as exc:
        parser.error(str(exc))
    for warning in warnings:
        print(f'Warning: {warning}', file=sys.stderr)
    report = dict(candidate=candidate, seeds=args.seeds, rng_seed=args.rng_seed,
                  seed_note='Caller must ensure these seeds are independent of training.',
                  warnings=warnings,
                  win_rate_denominator='All scheduled games, including double losses and pending games.',
                  evaluations=[])
    for opponent in opponents:
        print(f'Evaluating against {opponent}', flush=True)
        evaluation = evaluate_opponent(candidate, opponent, args.seeds, rng_seed=args.rng_seed)
        report['evaluations'].append(evaluation)
        s = evaluation['summary']
        print(f"{s['wins']}/{s['games']} wins ({s['win_rate']:.1%}); "
              f"{s['losses']} losses including {s['double_losses']} double losses; "
              f"{s['pending']} pending. Candidate mean/peak response: "
              f"{s['candidate']['mean_response_us']} / {s['candidate']['peak_response_us']} us", flush=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Saved {output.resolve()}')


if __name__ == '__main__':
    main()
