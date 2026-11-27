import json
from pathlib import Path
from engine import cells


def write_replay(path, b, result, frames, seed):
    board = dict(high=list(cells(b.high)), defects=list(cells(b.defects)),
                 homes=[list(cells(m)) for m in b.homes], marks=b.markers,
                 fast=list(cells(b.fast)), spawns=b.spawns)
    data = json.dumps(dict(board=board,result=result,frames=frames,seed=seed),
                      ensure_ascii=False,separators=(",",":")).replace("<","\\u003c")
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    Path(path).write_text(template.replace("__MATCH_DATA__",data), encoding="utf-8")

