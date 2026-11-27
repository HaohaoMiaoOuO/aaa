"""Local referee. Cells are y * 41 + x. Bit i in a mask describes cell i."""
from dataclasses import dataclass
import random

N = 41
SIZE = N * N
FULL = (1 << SIZE) - 1
TOP = (1 << N) - 1
BOTTOM = TOP << (SIZE - N)
LEFT = sum(1 << (y * N) for y in range(N))
RIGHT = LEFT << (N - 1)
RIM = TOP | BOTTOM | LEFT | RIGHT
DIRS = "URDL"
DELTAS = (-N, 1, N, -1)


def bit(i):
    return 1 << i


def cells(mask):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def neighbors(mask):
    return (((mask & ~RIGHT) << 1) | ((mask & ~LEFT) >> 1) |
            (mask << N) | (mask >> N)) & FULL


def flood(seed, allowed):
    area = seed & allowed
    front = area
    while front:
        front = neighbors(front) & allowed & ~area
        area |= front
    return area


def rectangle(x, y, w, h):
    row = (1 << w) - 1
    return sum(row << ((y + j) * N + x) for j in range(h))


def target(cell, direction):
    x, y = cell % N, cell // N
    if (direction == 0 and y == 0 or direction == 1 and x == N-1 or
            direction == 2 and y == N-1 or direction == 3 and x == 0):
        return -1
    return cell + DELTAS[direction]


@dataclass(frozen=True)
class Board:
    defects: int
    high: int
    homes: tuple
    spawns: tuple
    zones: tuple
    markers: tuple

    @property
    def fast(self):
        return self.zones[0] | self.zones[1] | self.zones[2]


def distances(start, allowed):
    layer = bit(start)
    seen = layer
    result = {}
    depth = 0
    while layer:
        result.update((p, depth) for p in cells(layer))
        layer = neighbors(layer) & allowed & ~seen
        seen |= layer
        depth += 1
    return result


def make_board(seed=1):
    rng = random.Random(seed)
    x, y = rng.randint(5, 10), rng.randint(25, 34)
    sp = (y*N+x, (N-1-y)*N+N-1-x)
    homes = tuple(rectangle(p%N-1, p//N-1, 3, 3) for p in sp)
    safe = sum(rectangle(p%N-4, p//N-4, 9, 9) for p in sp)
    while True:
        defects = high = 0
        for p in range((SIZE+1)//2):
            pair = bit(p) | bit(SIZE-1-p)
            u = rng.random()
            if u < .073 and not pair & safe:
                defects |= pair
            elif u < .22 and not pair & (homes[0] | homes[1]):
                high |= pair
        allowed = FULL ^ defects
        a, b = (distances(p, allowed) for p in sp)
        if len(a) != allowed.bit_count():
            continue
        equal = [p for p in a if a[p] == b[p]]
        if not equal:
            continue
        best = min(a[p] for p in equal)
        pool = [p for p in equal if a[p] <= best+2 and not bit(p) & safe]
        if len(pool) >= 3:
            break
    marks = tuple(rng.sample(sorted(pool), 3))
    zones = tuple((bit(p) | neighbors(bit(p)) | neighbors(neighbors(bit(p)))) & allowed
                  for p in marks)
    return Board(defects, high, homes, sp, zones, marks)


def score(board, side, owned):
    outside = owned & ~board.homes[side]
    adjacent = sum((outside & near).bit_count() for near in
                   ((owned & ~RIGHT) << 1, (owned & ~LEFT) >> 1, owned << N, owned >> N))
    edges = 4 * outside.bit_count() - adjacent
    return owned.bit_count() + 2*(owned & board.high).bit_count() - edges//3


def candidates(board, side, owned, trail):
    walls = owned | trail
    reachable = flood(RIM, FULL ^ walls)
    return (FULL ^ reachable) & ~(owned | board.defects | board.homes[1-side])


@dataclass
class State:
    owned: list
    trail: list
    pos: list
    heading: list
    down: list
    points: list
    leads: list
    turn: int = 0

    @classmethod
    def initial(cls, board):
        return cls(list(board.homes), [0, 0], list(board.spawns), [0, 2],
                   [0, 0], [9, 9], [0, 0])

    def copy(self):
        return State(*(list(a) for a in (self.owned, self.trail, self.pos, self.heading,
                                          self.down, self.points, self.leads)), self.turn)


def step(board, s, commands):
    """Resolve one accepted turn in place. Return an ordered event log."""
    events = []
    dead = set()
    def scrap(i, why, sub):
        if i not in dead:
            events.append((sub, "scrap", i, why))
            dead.add(i)

    steps = [0, 0]
    for i in (0, 1):
        if s.down[i]:
            continue
        cmd = commands[i]
        if cmd in tuple(DIRS):
            d = DIRS.index(cmd)
            if d != (s.heading[i]+2) % 4:
                s.heading[i] = d
        steps[i] = 2 if bit(s.pos[i]) & board.fast else 1
    for sub in range(1, max(steps)+1):
        going = [i for i in (0, 1) if steps[i] >= sub and i not in dead]
        dest = {}
        for i in going:
            p = target(s.pos[i], s.heading[i])
            if p < 0 or bit(p) & board.defects:
                scrap(i, "wall", sub)
            else:
                dest[i] = p
        if len(dest) == 2 and (dest[0] == dest[1] or
                (dest[0] == s.pos[1] and dest[1] == s.pos[0])):
            for i in going:
                if not bit(dest[i]) & s.owned[i]:
                    scrap(i, "collision", sub)
        moved = [i for i in dest if i not in dead]
        for i in moved:
            s.pos[i] = dest[i]
        for i in moved:
            for owner in (0, 1):
                if bit(dest[i]) & s.trail[owner]:
                    scrap(owner, "trail", sub)
        closing = []
        for i in moved:
            if i in dead:
                continue
            tile = bit(dest[i])
            if tile & s.owned[i]:
                if s.trail[i]:
                    closing.append(i)
            else:
                s.trail[i] |= tile
                s.trail[1-i] &= ~tile
        claims = [0, 0]
        for i in closing:
            claims[i] = candidates(board, i, s.owned[i], s.trail[i])
        for i in closing:
            gain = claims[i] & ~claims[1-i]
            s.owned[i] |= gain
            s.owned[1-i] &= ~gain
            s.trail[i] = 0
            events.append((sub, "close", i, gain.bit_count()))
    def remove(i):
        s.pos[i] = -1
        s.down[i] = 5
        s.trail[i] = 0
    for i in dead:
        remove(i)
    restarted = []
    for i in (0, 1):
        if i not in dead and s.down[i]:
            s.down[i] -= 1
            if not s.down[i]:
                s.pos[i] = board.spawns[i]
                s.heading[i] = 2*i
                restarted.append(i)
                events.append((3, "restart", i, 0))
    for i in restarted:
        if bit(s.pos[i]) & s.trail[1-i]:
            scrap(1-i, "restart", 3)
    for i in dead:
        remove(i)
    for i in (0, 1):
        if s.down[i] == 0 and not bit(s.pos[i]) & s.owned[i]:
            s.trail[i] |= bit(s.pos[i])
            s.trail[1-i] &= ~bit(s.pos[i])
        s.points[i] = score(board, i, s.owned[i])
    if s.points[0] != s.points[1]:
        s.leads[int(s.points[1] > s.points[0])] += 1
    s.turn += 1
    return events
