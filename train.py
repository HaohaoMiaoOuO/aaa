"""Train the lightweight policy with REINFORCE against a mixed opponent pool.

Training is deliberately separate from the timed arena: it uses the same turn
referee in-process, and does not try to model transport failures or latency.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from bot2 import Bot as Bot2
from engine import DIRS, State, make_board, step
from rl_model import Policy


def game_result(state, side):
    """Rules' first two normal-ending tiebreakers; a complete tie is neutral.

    Response time is platform-dependent and is not measured in training.
    """
    other = 1 - side
    ours = (state.points[side], state.leads[side])
    theirs = (state.points[other], state.leads[other])
    return int(ours > theirs) - int(ours < theirs)


def sample_action(policy, board, state, side, rng, record=False):
    features = policy.encode(board, state, side)
    with torch.no_grad():
        probs, _ = policy.forward(features, state.heading[side])
    if torch.is_tensor(probs):
        probs = probs.cpu().numpy()
    action = int(rng.choice(len(DIRS), p=probs / probs.sum()))
    item = (features, state.heading[side], action) if record and not state.down[side] else None
    return DIRS[action], item


def episode(policy, opponent, board, side, rng, score_weight):
    """Play 600 turns; both choices see separate copies of the same state."""
    state = State.initial(board)
    samples = []
    rewards = []
    for _ in range(600):
        command, item = sample_action(policy, board, state.copy(), side, rng, True)
        if isinstance(opponent, Policy):
            other_command, _ = sample_action(opponent, board, state.copy(), 1-side, rng)
        else:
            other_command = opponent.choose(state.copy())
        previous = state.points[side] - state.points[1-side]
        commands = [None, None]
        commands[side], commands[1-side] = command, other_command
        step(board, state, commands)
        current = state.points[side] - state.points[1-side]
        samples.append(item)
        rewards.append(score_weight * (current - previous))
    outcome = game_result(state, side)
    rewards[-1] += outcome
    return samples, rewards, state, outcome


def reinforce(policy, samples, rewards, learning_rate, gamma, max_grad_norm, baseline=0.0, optimizer=None):
    """One episodic REINFORCE update of the masked softmax policy.

    The baseline comes from earlier episodes, so the current outcome remains a
    learning signal. Down-state samples are excluded since commands are ignored.
    """
    returns = np.empty(len(rewards), dtype=np.float64)
    value = 0.0
    for t in range(len(rewards)-1, -1, -1):
        value = rewards[t] + gamma * value
        returns[t] = value
    active = [(sample, returns[t]) for t, sample in enumerate(samples) if sample is not None]
    if not active:
        return baseline
    device = next(policy.parameters()).device
    features = torch.as_tensor(np.stack([sample[0] for sample, _ in active]), device=device)
    headings = torch.tensor([sample[1] for sample, _ in active], dtype=torch.long, device=device)
    actions = torch.tensor([sample[2] for sample, _ in active], dtype=torch.long, device=device)
    advantages = torch.tensor([ret - baseline for _, ret in active], dtype=torch.float32, device=device)
    if optimizer is None:
        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)
    probs, _ = policy(features, headings)
    log_probs = torch.log(probs.gather(1, actions[:, None]).clamp_min(1e-12)).squeeze(1)
    loss = -(log_probs * advantages).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()
    return 0.95 * baseline + 0.05 * float(np.mean([ret for _, ret in active]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--output-dir', type=Path, default=Path('runs/experiment'))
    parser.add_argument('--resume', type=Path, help='start from an existing checkpoint and its JSON metadata')
    parser.add_argument('--train-seed-start', type=int, default=0)
    parser.add_argument('--train-seed-count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42, help='RNG seed for training, not map seed')
    parser.add_argument('--checkpoint-every', type=int, default=100)
    parser.add_argument('--history-size', type=int, default=8)
    parser.add_argument('--bot2-prob', type=float, default=0.4)
    parser.add_argument('--history-prob', type=float, default=0.3)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--score-weight', type=float, default=0.0,
                        help='score-difference shaping; nonzero can change the objective')
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--device', default='cpu', help='PyTorch device, e.g. cpu, cuda, cuda:0 or mps')
    parser.add_argument('--threads', type=int, default=1, help='PyTorch CPU threads (default: 1)')
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.train_seed_count < 1 or args.checkpoint_every < 1 or args.threads < 1:
        parser.error('episodes, train-seed-count, checkpoint-every and threads must be positive')
    numbers = (args.bot2_prob, args.history_prob, args.learning_rate, args.gamma,
               args.score_weight, args.max_grad_norm)
    if not all(math.isfinite(x) for x in numbers):
        parser.error('numeric parameters must be finite')
    if args.seed < 0:
        parser.error('seed must be non-negative')
    if args.history_size < 0 or min(args.bot2_prob, args.history_prob) < 0 or args.bot2_prob + args.history_prob > 1:
        parser.error('history-size and opponent probabilities are invalid')
    if args.learning_rate <= 0 or not 0 < args.gamma <= 1 or args.score_weight < 0 or args.max_grad_norm <= 0:
        parser.error('learning-rate/max-grad-norm must be positive, gamma in (0,1], score-weight non-negative')
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    try:
        device = torch.device(args.device)
        torch.empty(1, device=device)
    except (RuntimeError, ValueError, AssertionError) as exc:
        raise SystemExit(f'PyTorch device is unavailable: {exc}') from exc
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.glob('policy_*.pt')) or any(args.output_dir.glob('policy_*.json')) or any(
            (args.output_dir / name).exists() for name in ('latest.pt', 'latest.json', 'latest.npz')):
        raise SystemExit('Output directory already contains checkpoints; choose a new directory to preserve existing files')
    rng = np.random.default_rng(args.seed)
    map_rng = random.Random(args.seed)
    ranges = []
    if args.resume:
        metadata_path = args.resume.with_suffix('.json')
        if not metadata_path.is_file():
            raise SystemExit('Resume requires the checkpoint JSON metadata to track all training seeds')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        ranges = metadata['train_seed_ranges']
        policy = Policy.load(args.resume)
    else:
        policy = Policy(seed=args.seed)
    policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    ranges.append([args.train_seed_start, args.train_seed_start + args.train_seed_count])
    history = [copy.deepcopy(policy)] if args.resume and args.history_size else []
    baseline = 0.0
    for number in range(1, args.episodes + 1):
        board = make_board(args.train_seed_start + map_rng.randrange(args.train_seed_count))
        side = (number - 1) % 2
        pick = float(rng.random())
        if pick < args.bot2_prob:
            opponent = Bot2(board, 1-side)
            kind = 'bot2'
        elif pick < args.bot2_prob + args.history_prob and history:
            opponent = history[int(rng.integers(len(history)))]
            kind = 'history'
        else:
            # Freeze the current policy for the entire episode.
            opponent = copy.deepcopy(policy)
            kind = 'self'
        samples, rewards, state, outcome = episode(policy, opponent, board, side, rng, args.score_weight)
        baseline = reinforce(policy, samples, rewards, args.learning_rate, args.gamma,
                             args.max_grad_norm, baseline, optimizer)
        print(f'episode={number} side={side} opponent={kind} result={outcome:+d} '
              f'score={state.points[side]}:{state.points[1-side]}', flush=True)
        if number % args.checkpoint_every == 0 or number == args.episodes:
            path = args.output_dir / f'policy_{number:07d}.pt'
            policy.save(path)
            metadata = {'train_seed_ranges': ranges, 'episodes': number,
                        'seed': args.seed, 'resume': str(args.resume) if args.resume else None}
            path.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
            if args.history_size:
                history.append(copy.deepcopy(policy))
                history = history[-args.history_size:]
            policy.save(args.output_dir / 'latest.pt')
            (args.output_dir / 'latest.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
            print(f'checkpoint={path}', flush=True)


if __name__ == '__main__':
    main()
