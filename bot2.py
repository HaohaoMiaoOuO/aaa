"""Frozen baseline v1. Standard-library deterministic heuristics, no model weights."""
from functools import lru_cache
from time import perf_counter
from engine import DIRS, bit, cells, candidates, score, step
from planning import Geometry, directions, loops, frontage


class Bot:
    def __init__(self, board, side):
        self.b = board
        self.i = side
        self.g = Geometry(board)
        self.eta = lru_cache(maxsize=12)(self.g.arrival)
        self.route = ()
        self.expected = -1

    def choose(self, s):
        i, j = self.i, 1-self.i
        p, h = s.pos[i], s.heading[i]
        if p < 0:
            self.route = ()
            return "U"
        # Reserve time for immediate tactical checks, IPC, and scheduler jitter.
        deadline = perf_counter()+.065
        if p != self.expected:
            self.route = ()
        enemy = s.pos[j] if s.pos[j] >= 0 else self.b.spawns[j]
        eta = self.eta(enemy, s.heading[j] if s.pos[j]>=0 else 2*j, s.down[j])
        opts = directions(h)
        # Full-turn outcome against each distinct reply, including acceleration,
        # mutual cuts, closure, respawn contact and end-of-turn ownership.
        tactics = {}
        for d in opts:
            results = []
            for e in directions(s.heading[j]) if s.down[j] == 0 else (0,):
                t = s.copy()
                commands = [None, None]
                commands[i], commands[j] = DIRS[d], DIRS[e]
                step(self.b, t, commands)
                own_loss = t.down[i] == 5
                kill = not s.down[j] and t.down[j] == 5
                delta = t.points[i]-t.points[j] - (s.points[i]-s.points[j])
                results.append((own_loss, kill, delta))
            tactics[d] = results
        safest = min(sum(r[0] for r in tactics[d])/len(tactics[d]) for d in opts)
        viable = [d for d in opts if sum(r[0] for r in tactics[d])/len(tactics[d]) == safest]
        forced = [d for d in viable if all(r[1] for r in tactics[d])]
        desired = ()
        if s.turn == 599:
            def terminal(d):
                states = []
                for e in directions(s.heading[j]):
                    t = s.copy()
                    cmd = [None,None]
                    cmd[i], cmd[j] = DIRS[d], DIRS[e]
                    step(self.b, t, cmd)
                    states.append((t.points[i]-t.points[j], t.leads[i]-t.leads[j]))
                return min(states)
            desired = (max(opts, key=terminal),)
            viable = opts
        elif forced:
            desired = (max(forced, key=lambda d: min(r[2] for r in tactics[d])),)
        elif s.trail[i]:
            line_eta = min((eta[q] for q in cells(s.trail[i])), default=99)
            if (self.route and self.route[0] in viable and
                    line_eta > 2*len(self.route) and len(self.route) <= 600-s.turn):
                desired = self.route
            else:
                desired = self.g.navigate(s, i, s.owned[i], maxdepth=25) or self.route
        else:
            # Proactive pursuit only when enemy cannot already reconnect before our cut.
            if s.trail[j] and not s.down[j]:
                chase = self.g.navigate(s, i, s.trail[j], maxdepth=4)
                if chase and chase[0] in viable:
                    earliest_home = min((eta[q] for q in cells(s.owned[j])), default=99)
                    if 2*len(chase) < earliest_home and eta[p] > 2*len(chase):
                        desired = chase
            if not desired:
                desired = self.expansion(s, viable, eta, deadline)
        if not desired or desired[0] not in viable:
            def fallback(d):
                path = self.g.advance(p,d)
                q = path[-1] if path else p
                return (min(r[2] for r in tactics[d]), bool(bit(q) & s.owned[i]),
                        sum(r[1] for r in tactics[d]), eta[q], d == h)
            desired = (max(viable,key=fallback),)
        action = desired[0]
        path = self.g.advance(p, action)
        self.expected = path[-1] if path else -1
        self.route = desired[1:]
        return DIRS[action]

    def expansion(self, s, viable, eta, deadline):
        i, j = self.i, 1-self.i
        own, opp = s.owned[i], s.owned[j]
        best, plan = float("-inf"), ()
        seen = set()
        # Different outward lengths and widths allow small safe tabs and larger steals.
        for route, trail, pts, closure in loops(self.g, s, i, (2,1,4,6,9), (1,2,4,6,9)):
            if perf_counter() >= deadline:
                break
            if route[0] not in viable or trail in seen or len(route) > 600-s.turn:
                continue
            seen.add(trail)
            gain = candidates(self.b, i, own, trail)
            if not gain:
                continue
            swing = (score(self.b, i, own|gain)-s.points[i] +
                     s.points[j]-score(self.b, j, opp & ~gain))
            # Conservative: an enemy reaching a future line cell early may wait nearby.
            margin = min(eta[q] for q in pts) - closure
            risk = max(0, 3-margin)
            caution = 2.2 if s.points[i]>=s.points[j] and s.turn>530 else 1.25
            value = swing/(len(route)+2)**.72 - caution*risk + .025*min(margin, 12)
            if value > best:
                best, plan = value, route
        if plan and best > -2:
            return plan
        if self.route and self.route[0] in viable:
            return self.route
        fringe = frontage(s, i, self.b) & ~bit(s.pos[i])
        # Bias staging positions towards valuable/contested cells.
        from engine import neighbors
        valuable = fringe & neighbors((self.b.high | opp) & ~own & ~self.b.homes[j])
        for goal in (valuable, fringe):
            if goal:
                path = self.g.navigate(s, i, goal, inside=True)
                if path and path[0] in viable:
                    return path
        return plan

