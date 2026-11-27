"""Shared movement geometry. A route is a list of absolute direction indices."""
from collections import deque
from engine import N, SIZE, bit, cells, target, neighbors


class Geometry:
    def __init__(self, board):
        self.board = board
        self.walk = []
        for p in range(SIZE):
            options = []
            for d in range(4):
                path = []
                q = p
                for _ in range(2 if bit(p) & board.fast else 1):
                    q = target(q, d)
                    if q < 0 or bit(q) & board.defects:
                        break
                    path.append(q)
                options.append(tuple(path))
            self.walk.append(options)
        self.length = [2 if bit(p) & board.fast else 1 for p in range(SIZE)]

    def advance(self, p, d):
        path = self.walk[p][d]
        return path if len(path) == self.length[p] else ()

    def arrival(self, start, heading, delay=0, limit=15):
        """Earliest touch in half-turn ticks. Relax future trail and collision limits.
        May touch the first cell even if the forced second step hits a wall."""
        inf = 2*(limit+1)+1
        eta = [inf]*SIZE
        eta[start] = 2*delay
        queue = deque([(start, heading, delay)])
        seen = {(start, heading)}
        while queue:
            p, h, t = queue.popleft()
            if t >= limit:
                continue
            for d in directions(h):
                path = self.walk[p][d]
                for k, q in enumerate(path, 1):
                    eta[q] = min(eta[q], 2*t+k)
                if len(path) == self.length[p] and (path[-1], d) not in seen:
                    seen.add((path[-1], d))
                    queue.append((path[-1], d, t+1))
        return eta

    def navigate(self, s, i, goal, inside=False, maxdepth=50):
        """Shortest command route to a goal; checked again against live trails next turn."""
        p, h = s.pos[i], s.heading[i]
        queue = deque([(p, h, (), 0)])
        seen = {(p,h)}
        while queue:
            p, h, route, visited = queue.popleft()
            if len(route) >= maxdepth:
                continue
            for d in directions(h):
                path = self.advance(p, d)
                if not path:
                    continue
                extra = visited
                for q in path:
                    tile = bit(q)
                    if tile & (s.trail[i] | extra) or (inside and not tile & s.owned[i]):
                        break
                    if goal & tile and (not inside or q == path[-1]):
                        return route+(d,)
                    if not tile & s.owned[i]:
                        extra |= tile
                else:
                    key = path[-1], d
                    if key not in seen:
                        seen.add(key)
                        queue.append((path[-1], d, route+(d,), extra))
        return ()


def directions(h):
    return h, (h+1)%4, (h-1)%4


def loops(g, s, i, sizes, widths):
    """Trace three sides of a rectangle; the owned boundary supplies the fourth."""
    p, h = s.pos[i], s.heading[i]
    own = s.owned[i]
    for d in directions(h):
        first = g.advance(p, d)
        if not first or bit(first[0]) & own:
            continue
        for outward in sizes:
            for side in ((d+1)%4, (d-1)%4):
                for width in widths:
                    route = (d,)*outward + (side,)*width + ((d+2)%4,)*(outward+8)
                    q, trail, points = p, 0, []
                    for n, action in enumerate(route, 1):
                        path = g.advance(q, action)
                        if not path:
                            break
                        for sub, q in enumerate(path, 1):
                            tile = bit(q)
                            if trail & tile:
                                break
                            if own & tile:
                                if trail:
                                    yield route[:n], trail, tuple(points), 2*(n-1)+sub
                                break
                            trail |= tile
                            points.append(q)
                        else:
                            continue
                        break


def frontage(s, i, board):
    return s.owned[i] & neighbors(~(s.owned[i] | board.defects | board.homes[1-i]) & ((1<<SIZE)-1))

