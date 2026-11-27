"""Run python arena.py play, or python arena.py bench --seeds 1 2 3."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
import multiprocessing as mp
from queue import Queue
from threading import Thread
from pathlib import Path
from statistics import mean
from time import monotonic, perf_counter_ns

from engine import DIRS, State, make_board, step


def worker(pipe, name, options=None):
    try:
        factory = import_module(name).Bot
        pipe.send((-2, "ready"))
        policy = None
        while True:
            seq, data = pipe.recv()
            try:
                if seq == -1:
                    policy = factory(*data, **(options or {}))
                    reply = "ok"
                elif seq == -3:
                    pipe.send((seq,"ok"))
                    return
                else:
                    reply = policy.choose(data)
                    if type(reply) is not str or reply not in tuple(DIRS):
                        reply = None
            except Exception as exc:
                import sys
                print(f"{name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                reply = None
            pipe.send((seq, reply))
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass
    finally:
        pipe.close()


class Endpoint:
    def __init__(self, ctx, name, options=None):
        self.pipe, child = ctx.Pipe()
        self.proc = ctx.Process(target=worker, args=(child, name, options), daemon=True)
        self.start = monotonic()
        self.proc.start()
        child.close()
        # A stalled worker must never block the referee in an OS pipe write.
        self.outbox = Queue()
        self.writer = Thread(target=self._write, daemon=True)
        self.writer.start()

    def _write(self):
        try:
            while True:
                item = self.outbox.get()
                if item is None:
                    return
                self.pipe.send(item)
        except (OSError, EOFError):
            return

    def exchange(self, seq, payload, limit):
        start = perf_counter_ns()
        try:
            if seq != -2:
                self.outbox.put((seq, payload.copy() if isinstance(payload, State) else payload))
            while True:
                left = limit - (perf_counter_ns()-start)/1e9
                if left <= 0 or not self.pipe.poll(left):
                    return None, round(limit*1e6), "timeout"
                tag, reply = self.pipe.recv()
                elapsed = (perf_counter_ns()-start)//1000
                if elapsed > limit*1e6:
                    return None, round(limit*1e6), "timeout"
                if tag == seq:
                    return reply, elapsed, "ok" if reply is not None else "invalid"
        except (OSError, EOFError):
            return None, round(limit*1e6), "disconnect"

    def close(self):
        self.outbox.put(None)
        self.proc.join(.05)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(1)
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join()
        self.writer.join(1)
        self.pipe.close()


def frame(s, events, commands, errors):
    return dict(turn=s.turn, own=[hex(x) for x in s.owned], trail=[hex(x) for x in s.trail],
                p=s.pos[:], d=s.heading[:], down=s.down[:], score=s.points[:], leads=s.leads[:],
                ev=events, cmd=commands, err=errors[:])


def game(board, names, record=False, bot_options=None):
    state = State.initial(board)
    errors, micros, peaks, scraps, closes = ([0,0] for _ in range(5))
    frames = [frame(state, [], [], errors)] if record else []
    ctx = mp.get_context("spawn")
    endpoints = []
    bad = []
    requests = 0
    bot_options = bot_options if bot_options is not None else [{}, {}]
    if len(names) != 2 or len(bot_options) != 2:
        raise ValueError("Exactly two bot names and options are required")
    try:
        for name, options in zip(names, bot_options):
            endpoints.append(Endpoint(ctx, name, options))
        with ThreadPoolExecutor(max_workers=2) as pool:
            def both(seq, payloads, limits):
                fs = [pool.submit(ep.exchange, seq, arg, lim)
                      for ep, arg, lim in zip(endpoints, payloads, limits)]
                return [f.result() for f in fs]
            ready = both(-2, [None,None], [max(0,10-(monotonic()-e.start)) for e in endpoints])
            bad = [i for i in (0,1) if ready[i][0] != "ready"]
            opening = {i: pool.submit(ep.exchange, -1, (board, i), 2)
                       for i, ep in enumerate(endpoints) if i not in bad}
            for i, request in opening.items():
                reply, _, status = request.result()
                errors[i] += reply != "ok"
                if status == "disconnect" or not endpoints[i].proc.is_alive():
                    bad.append(i)
            while not bad and state.turn < 600:
                replies = both(state.turn+1, [state,state], [.2,.2])
                requests += 1
                commands = []
                for i,(reply,us,status) in enumerate(replies):
                    commands.append(reply)
                    errors[i] += status != "ok"
                    micros[i] += us
                    peaks[i] = max(peaks[i],us)
                    if errors[i] >= 10 or not endpoints[i].proc.is_alive() or status == "disconnect" and endpoints[i].proc.exitcode is not None:
                        bad.append(i)
                if bad:
                    break
                events = step(board, state, commands)
                for _, event, i, _ in events:
                    if event == "close":
                        closes[i] += 1
                    if event == "scrap":
                        scraps[i] += 1
                if record:
                    frames.append(frame(state, events, commands, errors))
            if all(e.proc.is_alive() for e in endpoints):
                both(-3,[None,None],[1,1])
    finally:
        for e in endpoints:
            e.close()
    if bad:
        kind = "double_loss" if len(bad)==2 else "technical"
        winner = None if len(bad)==2 else 1-bad[0]
    else:
        kind = "normal"
        keys = [(state.points[i],state.leads[i],-micros[i]) for i in (0,1)]
        winner = None if keys[0]==keys[1] else int(keys[1]>keys[0])
    return dict(names=names, kind=kind, winner=winner, turns=state.turn, score=state.points,
                leads=state.leads, us=micros, peak_us=peaks, errors=errors, closes=closes,
                scraps=scraps, requests=requests), frames


def show(r):
    w = r["winner"]
    winner = r["names"][w] if w is not None else ("双败" if r["kind"]=="double_loss" else "待裁定")
    print(f'{r["names"][0]} vs {r["names"][1]}  {r["score"]}  {r["turns"]} 回合  '
          f'{r["kind"]}  胜方 {winner}  错误 {r["errors"]}',flush=True)


def match_outcome(rows):
    # Rows already reordered by bot identity. Manual adjudications stay pending.
    if all(r["kind"]=="double_loss" for r in rows):
        return "双败"
    if any(r["kind"]=="normal" and r["win"] is None for r in rows):
        return "待裁定"
    normal = [r for r in rows if r["kind"]=="normal"]
    keys = [(sum(r["win"]==i for r in rows),
             sum(r[f"score{i+1}"] for r in normal),
             sum(r[f"leads{i+1}"] for r in normal),
             -sum(r[f"us{i+1}"] for r in normal)) for i in (0,1)]
    return "待裁定" if keys[0]==keys[1] else ("Bot1" if keys[0]>keys[1] else "Bot2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    play = commands.add_parser("play")
    play.add_argument("--seed", type=int, default=1)
    play.add_argument("--out", default="replay.html")
    play.add_argument("--open", action="store_true", help="Open the replay in your browser")
    play.add_argument("--swap", action="store_true")
    bench = commands.add_parser("bench")
    bench.add_argument("--seeds", type=int, nargs="+", default=[1,2,3])
    bench.add_argument("--csv", default="results.csv")
    for sub in (play,bench):
        sub.add_argument("--bot1",default="bot1")
        sub.add_argument("--bot2",default="bot2")
    args = parser.parse_args()
    names = [args.bot1,args.bot2]
    if args.mode == "play":
        from replay import write_replay
        b = make_board(args.seed)
        r,frames = game(b,names[::-1] if args.swap else names,True)
        show(r)
        write_replay(args.out,b,r,frames,args.seed)
        print(Path(args.out).resolve())
        if args.open:
            import webbrowser
            webbrowser.open(Path(args.out).resolve().as_uri())
        return
    rows = []
    for seed in args.seeds:
        board = make_board(seed)
        for swap in (0,1):
            r,_ = game(board,names[::-1] if swap else names)
            show(r)
            row = dict(seed=seed, swap=swap, kind=r["kind"], turns=r["turns"],
                       win=r["winner"]^swap if r["winner"] is not None else None)
            for field in ("score","leads","us","peak_us","errors","closes","scraps"):
                for i in (0,1):
                    row[field+str(i+1)] = r[field][i^swap]
            rows.append(row)
    with open(args.csv,"w",encoding="utf-8-sig",newline="") as file:
        writer = csv.DictWriter(file,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("整场:",match_outcome(rows))
    normal = [r for r in rows if r["kind"]=="normal"]
    for i, name in enumerate(names):
        wins = sum(r["win"]==i for r in rows)
        losses = sum(r["win"]==1-i or r["kind"]=="double_loss" for r in rows)
        diff = mean(r[f"score{i+1}"]-r[f"score{2-i}"] for r in normal) if normal else 0
        print(f"Bot{i+1} {name}: {wins} 胜 {losses} 负, 正常局平均分差 {diff:+.1f}")
    print(Path(args.csv).resolve())


if __name__ == "__main__":
    mp.freeze_support()
    main()
