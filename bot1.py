"""Replace Bot.choose with your own strategy. This starter closes small loops."""
from engine import DIRS, bit
from planning import Geometry, directions, loops, frontage


class Bot:
    def __init__(self, board, side):
        self.board, self.side = board, side
        self.g = Geometry(board)
        self.route = ()

    def choose(self, state):
        i = self.side
        p, h = state.pos[i], state.heading[i]
        if p < 0:
            self.route = ()
            return "U"
        if self.route:
            d, *remaining = self.route
            self.route = tuple(remaining)
            return DIRS[d]
        if state.trail[i]:
            route = self.g.navigate(state, i, state.owned[i], maxdepth=20)
        else:
            found = list(loops(self.g, state, i, (2, 3), (1, 2, 3)))
            if found:
                route = max(found, key=lambda v: len(v[2])/len(v[0]))[0]
            else:
                goal = frontage(state, i, self.board) & ~bit(p)
                route = self.g.navigate(state, i, goal, inside=True) if goal else ()
        if route:
            self.route = route[1:]
            return DIRS[route[0]]
        for d in directions(h):
            path = self.g.advance(p,d)
            if path and all(not bit(q) & state.trail[i] for q in path):
                return DIRS[d]
        return DIRS[h]

