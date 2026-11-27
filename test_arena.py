import sys
import tempfile
import unittest
from pathlib import Path
from engine import (Board, State, N, FULL, bit, rectangle, neighbors, cells, flood,
                    step, make_board, score, candidates, distances)
from arena import game

def xy(x,y):
    return y*N+x

def board(fast=0, defects=0, high=0):
    return Board(defects, high, (rectangle(3,3,3,3),rectangle(35,35,3,3)),
                 (xy(4,4),xy(36,36)),(fast,0,0),(0,1,2))

class RefereeTests(unittest.TestCase):
    def test_map_constraints(self):
        for seed in [0,11,29,71,411]:
            b = make_board(seed)
            for mask in (b.defects,b.high):
                self.assertEqual(mask,sum(bit(N*N-1-p) for p in cells(mask)))
            a,z = [distances(p,FULL^b.defects) for p in b.spawns]
            self.assertEqual(len(a),(FULL^b.defects).bit_count())
            best = min(a[p] for p in a if a[p]==z[p])
            self.assertEqual(len(set(b.markers)),3)
            for p, zone in zip(b.markers,b.zones):
                self.assertTrue(bit(p)&zone)
                self.assertFalse(zone&b.defects)
                self.assertEqual(a[p],z[p])
                self.assertLessEqual(a[p],best+2)
            for p in b.spawns:
                area=rectangle(p%N-4,p//N-4,9,9)
                self.assertFalse(area & (sum(bit(m) for m in b.markers)|b.defects))
    def test_scoring_examples(self):
        b=board(high=bit(xy(7,4)))
        self.assertEqual(score(b,0,b.homes[0]|rectangle(6,3,3,3)),17)
        self.assertEqual(score(b,0,b.homes[0]|rectangle(12,12,8,2)),19)
        self.assertEqual(score(b,0,b.homes[0]|rectangle(9,9,2,2)|rectangle(15,15,2,2)),12)
    def test_reverse_and_enter_boost(self):
        b=board(fast=bit(xy(4,3)))
        s=State.initial(b)
        step(b,s,["D","D"])
        self.assertEqual(s.pos[0],xy(4,3))
        step(b,s,["R","D"])
        self.assertEqual(s.pos[0],xy(6,3))
    def test_collision_and_swap(self):
        for swap in [False,True]:
            for owner in [None,0,1]:
                b=board()
                s=State.initial(b)
                s.pos=[xy(10,10),xy(11 if swap else 12,10)]
                s.heading=[1,3]
                if owner is not None:
                    dest=xy(10,10) if swap and owner==1 else xy(11,10)
                    s.owned[owner]|=bit(dest)
                step(b,s,["R","L"])
                self.assertEqual(s.down,[0 if owner==i else 5 for i in (0,1)])
    def test_mutual_cut(self):
        b=board()
        s=State.initial(b)
        s.pos=[xy(10,10),xy(20,20)]
        s.trail=[bit(xy(20,21)),bit(xy(11,10))]
        step(b,s,["R","D"])
        self.assertEqual(s.down,[5,5])
        self.assertEqual(s.trail,[0,0])
    def test_cut_before_closure(self):
        b=board()
        s=State.initial(b)
        s.pos=[xy(2,3),xy(1,4)]
        s.trail[0]=bit(s.pos[0])|bit(xy(2,4))
        step(b,s,["R","R"])
        self.assertEqual(s.down,[5,0])
        self.assertEqual(s.owned[0],b.homes[0])
    def test_mid_turn_close_then_depart(self):
        start,home=xy(9,9),xy(10,9)
        b=board(fast=bit(start))
        s=State.initial(b)
        s.pos[0]=start
        s.owned[0]|=bit(home)
        s.trail[0]=bit(start)
        ev=step(b,s,["R","D"])
        self.assertIn((1,"close",0,1),ev)
        self.assertEqual(s.trail[0],bit(xy(11,9)))
    def test_second_step_visits_stationary_enemy(self):
        start=xy(10,10)
        b=board(fast=bit(start))
        s=State.initial(b)
        s.pos=[start,xy(12,9)]
        ev=step(b,s,["R","D"])
        self.assertEqual(s.down,[0,5])
        self.assertIn((2,"scrap",1,"trail"),ev)
        self.assertEqual(s.pos[0],xy(12,10))
    def test_survivor_completes_second_step_after_enemy_wall_crash(self):
        start=xy(10,10)
        b=board(fast=bit(start))
        s=State.initial(b)
        s.pos=[start,xy(1,40)]
        s.trail[1]=bit(xy(12,10))
        ev=step(b,s,["R","D"])
        self.assertIn((1,"scrap",1,"wall"),ev)
        self.assertEqual(s.pos[0],xy(12,10))
        self.assertEqual(s.down,[0,5])
    def test_restart_timing_and_sweep(self):
        b=board()
        s=State.initial(b)
        s.pos[0]=xy(4,0)
        step(b,s,["U","L"])
        self.assertEqual(s.down[0],5)
        for n in range(4):
            step(b,s,["R","U"])
            self.assertEqual(s.down[0],4-n)
        s.trail[1]|=bit(b.spawns[0])
        step(b,s,["R","U"])
        self.assertEqual(s.pos[0],b.spawns[0])
        self.assertEqual(s.heading[0],0)
        self.assertEqual(s.down,[0,5])
    def test_defect_permeable_to_fill(self):
        r=rectangle(9,9,5,5)
        inside=rectangle(10,10,3,3)
        hole=bit(xy(11,9))
        b=board(defects=hole)
        wall=r & ~inside & ~hole
        self.assertFalse(candidates(b,0,b.homes[0],wall)&inside)
    def test_contested_closure_and_retag(self):
        b=board()
        s=State.initial(b)
        for i,(origin,size) in enumerate(((10,11),(13,5))):
            edge=rectangle(origin,origin,size,size)^rectangle(origin+1,origin+1,size-2,size-2)
            anchor=bit(xy(origin,origin))
            s.owned[i]|=anchor
            s.pos[i]=xy(origin-1,origin)
            s.trail[i]=(edge & ~anchor)|bit(s.pos[i])
        ev=step(b,s,["R","R"])
        self.assertEqual(sum(e[1]=="close" for e in ev),2)
        middle=bit(xy(15,15))
        self.assertFalse(middle & (s.owned[0]|s.owned[1]))
        self.assertTrue(s.owned[0]&bit(xy(13,13)))
        self.assertTrue(s.trail[1]&bit(xy(13,13)))

class ClockTests(unittest.TestCase):
    def run_case(self,body1,body2):
        with tempfile.TemporaryDirectory() as root:
            sys.path.insert(0,root)
            try:
                for name,body in (("fixture_a",body1),("fixture_b",body2)):
                    Path(root,name+".py").write_text("class Bot:\n def __init__(self,*a): pass\n def choose(self,s):\n  "+body+"\n",encoding="utf-8")
                return game(board(),["fixture_a","fixture_b"])[0]
            finally:
                sys.path.remove(root)
    def test_invalid_double_loss(self):
        r=self.run_case("return 'XX'","return 'RR'")
        self.assertEqual((r["kind"],r["turns"],r["errors"]),("double_loss",9,[10,10]))
    def test_hung_bot_cannot_block_referee(self):
        r=self.run_case("import time; time.sleep(60)","return 'D'")
        self.assertEqual((r["kind"],r["turns"],r["winner"]),("technical",9,1))
        self.assertEqual(r["us"][0],2000000)
    def test_exit_does_not_settle_turn(self):
        r=self.run_case("import os; os._exit(1)","return 'D'")
        self.assertEqual((r["winner"],r["turns"],r["score"]),(1,0,[9,9]))

if __name__=="__main__":
    unittest.main()

