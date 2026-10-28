"""Experiment 3: Unfamiliar challenges (navigation).

The brain is dropped into a grid world. It sees a 5x5 patch around itself
(walls, the goal if close, hazards, and cells it has already visited this
episode - a short-term memory) plus the rough direction of the goal. It gets
+1 for reaching the goal, small penalties for time and bumping into walls.
It is never told how to navigate.

It trains only on small open rooms. It is then tested, without warning, on
layouts it has never seen: bigger rooms, winding corridor mazes, deadly
hazard tiles (never present in training) and mirrored controls. For the last
two it is allowed to keep learning, so recovery can be measured."""
import json, os, random, time
from collections import deque
import numpy as np
from core import Net, Recorder, ROOT

RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right
ANAMES = ["up", "down", "left", "right"]
rng = random.Random(11)
nprng = np.random.default_rng(11)


def reachable(grid, s, g):
    n = len(grid)
    q, seen = deque([s]), {s}
    while q:
        r, c = q.popleft()
        if (r, c) == g:
            return True
        for dr, dc in ACTIONS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and grid[rr][cc] != 1 and (rr, cc) not in seen:
                seen.add((rr, cc)); q.append((rr, cc))
    return False


def make_rooms(n=9, wall_p=0.18, hazard_p=0.0, r=rng):
    while True:
        grid = [[1 if r.random() < wall_p else 0 for _ in range(n)] for _ in range(n)]
        free = [(i, j) for i in range(n) for j in range(n) if grid[i][j] == 0]
        s, g = r.sample(free, 2)
        if abs(s[0] - g[0]) + abs(s[1] - g[1]) < n // 2:
            continue
        for (i, j) in free:
            if (i, j) not in (s, g) and r.random() < hazard_p:
                grid[i][j] = 2
        if reachable(grid, s, g):
            return grid, s, g


def make_maze(n=11, r=rng):
    """Perfect corridor maze (one route), carved by depth-first search."""
    grid = [[1] * n for _ in range(n)]
    st = [(1, 1)]
    grid[1][1] = 0
    while st:
        cr, cc = st[-1]
        nb = [(cr + 2 * dr, cc + 2 * dc, dr, dc) for dr, dc in ACTIONS
              if 0 < cr + 2 * dr < n - 1 and 0 < cc + 2 * dc < n - 1 and grid[cr + 2 * dr][cc + 2 * dc] == 1]
        if not nb:
            st.pop(); continue
        nr, nc, dr, dc = r.choice(nb)
        grid[cr + dr][cc + dc] = 0; grid[nr][nc] = 0
        st.append((nr, nc))
    free = [(i, j) for i in range(n) for j in range(n) if grid[i][j] == 0]
    while True:
        s, g = r.sample(free, 2)
        if abs(s[0] - g[0]) + abs(s[1] - g[1]) >= n - 2:
            return grid, s, g


class World:
    def __init__(self, grid, start, goal, mirrored=False):
        self.grid, self.pos, self.goal = [row[:] for row in grid], start, goal
        self.n = len(grid)
        self.visited = {start: 1}
        self.steps = 0
        self.max_steps = 6 * self.n
        self.mirrored = mirrored

    def obs(self):
        x = np.zeros((4, 5, 5), np.float32)
        r0, c0 = self.pos
        for i in range(5):
            for j in range(5):
                r, c = r0 + i - 2, c0 + j - 2
                if not (0 <= r < self.n and 0 <= c < self.n) or self.grid[r][c] == 1:
                    x[0, i, j] = 1
                elif self.grid[r][c] == 2:
                    x[2, i, j] = 1
                if (r, c) == self.goal:
                    x[1, i, j] = 1
                if (r, c) in self.visited:
                    x[3, i, j] = min(1.0, self.visited[(r, c)] / 3)
        dr, dc = self.goal[0] - r0, self.goal[1] - c0
        comp = np.array([np.sign(dr), np.sign(dc), dr / self.n, dc / self.n], np.float32)
        return np.concatenate([x.ravel(), comp])

    def step(self, a):
        if self.mirrored and a in (2, 3):
            a = 5 - a
        dr, dc = ACTIONS[a]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        self.steps += 1
        rew, done, event = -0.01, False, None
        if not (0 <= r < self.n and 0 <= c < self.n) or self.grid[r][c] == 1:
            rew -= 0.03; event = "bump"
        else:
            self.pos = (r, c)
            self.visited[self.pos] = self.visited.get(self.pos, 0) + 1
            if self.pos == self.goal:
                rew, done, event = 1.0, True, "goal"
            elif self.grid[r][c] == 2:
                rew, done, event = -1.0, True, "hazard"
        if self.steps >= self.max_steps and not done:
            done, event = True, "timeout"
        return rew, done, event


class NavBrain:
    def __init__(self):
        self.net = Net([104, 64, 64, 4], seed=3, lr=5e-4)
        self.target = Net([104, 64, 64, 4], seed=3)
        self.target.load(self.net.state())
        self.mem = deque(maxlen=60_000)
        self.updates = 0

    def act(self, o, eps):
        q, acts = self.net.forward(o[None])
        explored = rng.random() < eps
        a = rng.randrange(4) if explored else int(np.argmax(q[0]))
        return a, explored, q[0], acts[1][0]

    def learn(self, bs=64, gamma=0.97):
        if len(self.mem) < 1000:
            return None
        ids = nprng.integers(0, len(self.mem), bs)
        batch = [self.mem[i] for i in ids]
        o = np.stack([b[0] for b in batch]); a = np.array([b[1] for b in batch])
        r = np.array([b[2] for b in batch], np.float32); o2 = np.stack([b[3] for b in batch])
        d = np.array([b[4] for b in batch], np.float32)
        a2 = np.argmax(self.net(o2), 1)
        tq = r + gamma * (1 - d) * self.target(o2)[np.arange(bs), a2]
        def g(out):
            grad = np.zeros_like(out)
            err = out[np.arange(bs), a] - tq
            err = np.clip(err, -1, 1)
            grad[np.arange(bs), a] = err / bs
            return float((err ** 2).mean()), grad
        loss = self.net.train(o, g)
        self.updates += 1
        if self.updates % 500 == 0:
            self.target.load(self.net.state())
        return loss


def episode(brain, world, eps, learn, rec=None, tag="", ep=0, trace=False):
    o = world.obs()
    total, steps, explored_n, stuck_events, loops = 0.0, [], 0, 0, 0
    event = None
    stuck = False
    while True:
        a, explored, q, hid = brain.act(o, eps)
        explored_n += explored
        prev = world.pos
        rew, done, event = world.step(a)
        o2 = world.obs()
        if learn:
            brain.mem.append((o, a, rew, o2, float(done and event != "timeout")))
            brain.learn()
        total += rew
        visits = world.visited.get(world.pos, 0)
        if visits == 4 and not stuck:
            stuck = True; stuck_events += 1
            if rec: rec.log("stuck", tag=tag, episode=ep, step=world.steps, cell=world.pos)
        if trace:
            steps.append(dict(step=world.steps, pos=prev, action=ANAMES[a], explored=explored,
                              q=np.round(q, 3).tolist(), hidden=np.round(hid, 3).tolist(), reward=round(rew, 3), event=event))
        o = o2
        if done:
            break
    recovered = stuck and event == "goal"
    out = dict(outcome=event, steps=world.steps, ret=round(total, 3), explored=explored_n, stuck=stuck_events,
               recovered=recovered, path_len_optimal=None)
    if rec:
        rec.log("episode_end", tag=tag, episode=ep, outcome=event, steps=world.steps, ret=out["ret"],
                exploratory_actions=explored_n, eps=round(eps, 3), stuck=stuck_events, recovered_after_stuck=recovered,
                memory_size=len(brain.mem))
    if trace:
        out["trace"] = steps
    return out


def shortest(grid, s, g):
    n = len(grid)
    q, d = deque([s]), {s: 0}
    while q:
        u = q.popleft()
        if u == g:
            return d[u]
        for dr, dc in ACTIONS:
            v = (u[0] + dr, u[1] + dc)
            if 0 <= v[0] < n and 0 <= v[1] < n and grid[v[0]][v[1]] == 0 and v not in d:
                d[v] = d[u] + 1; q.append(v)
    return None


def test_suite(brain, rec, tag, maker, n=60, eps=0.05, seed=0, mirrored=False, random_agent=False, learn=False):
    r = random.Random(seed)
    outs = []
    for i in range(n):
        grid, s, g = maker(r)
        w = World(grid, s, g, mirrored=mirrored)
        if random_agent:
            ev = None
            while True:
                _, done, ev = w.step(rng.randrange(4))
                if done: break
            o = dict(outcome=ev, steps=w.steps, stuck=0, recovered=False)
        else:
            o = episode(brain, w, eps, learn, rec=rec, tag=tag, ep=i)
        o["optimal"] = shortest(grid, s, g)
        outs.append(o)
    succ = [o for o in outs if o["outcome"] == "goal"]
    res = dict(tag=tag, episodes=n, success_rate=len(succ) / n,
               hazard_rate=sum(o["outcome"] == "hazard" for o in outs) / n,
               timeout_rate=sum(o["outcome"] == "timeout" for o in outs) / n,
               mean_steps_success=float(np.mean([o["steps"] for o in succ])) if succ else None,
               mean_optimal_success=float(np.mean([o["optimal"] for o in succ if o["optimal"]])) if succ else None,
               stuck_rate=sum(o["stuck"] > 0 for o in outs) / n,
               recovered_rate_given_stuck=(sum(o["recovered"] for o in outs) / max(1, sum(o["stuck"] > 0 for o in outs))))
    if rec:
        rec.log("test_result", **res)
    return res


ROOMS = lambda r: make_rooms(9, r=r)
BIGROOMS = lambda r: make_rooms(13, r=r)
MAZES = lambda r: make_maze(11, r=r)
HAZARDS = lambda r: make_rooms(9, wall_p=0.12, hazard_p=0.12, r=r)


def showcase(brain, rec, tag, maker, seed, eps=0.05, mirrored=False):
    r = random.Random(seed)
    grid, s, g = maker(r)
    w = World(grid, s, g, mirrored=mirrored)
    o = episode(brain, w, eps, False, rec=rec, tag="showcase-" + tag, trace=True)
    return dict(tag=tag, grid=grid, start=s, goal=g, optimal=shortest(grid, s, g), **o)


def main(train_eps=3000):
    rec = Recorder("unfamiliar")
    brain = NavBrain()
    results, shows = {}, []
    t0 = time.time()
    # before any learning
    results["before_rooms"] = test_suite(brain, rec, "before-learning/rooms", ROOMS, seed=100)
    results["random_rooms"] = test_suite(brain, None, "random-agent/rooms", ROOMS, seed=100, random_agent=True)
    shows.append(showcase(brain, rec, "before-learning/rooms", ROOMS, 7))
    curve = []
    for ep in range(1, train_eps + 1):
        eps = max(0.05, 1.0 - ep / (0.5 * train_eps))
        grid, s, g = make_rooms(9)
        o = episode(brain, World(grid, s, g), eps, True, rec=rec, tag="train", ep=ep)
        curve.append(dict(episode=ep, outcome=o["outcome"], steps=o["steps"], eps=eps, explored=o["explored"]))
        if ep % 250 == 0:
            sr = np.mean([c["outcome"] == "goal" for c in curve[-250:]])
            print(f"ep{ep} {time.time()-t0:.0f}s eps={eps:.2f} success(last250)={sr:.2f}", flush=True)
            results.setdefault("checkpoints", []).append(dict(train_episode=ep, **
                test_suite(brain, rec, f"checkpoint{ep}/rooms", ROOMS, n=40, seed=100)))
    brain.net.save(os.path.join(RES, "nav_net_trained.npz"))
    trained = brain.net.state()
    shows.append(showcase(brain, rec, "after-learning/rooms", ROOMS, 7))
    for name, maker in [("rooms", ROOMS), ("bigrooms", BIGROOMS), ("mazes", MAZES), ("hazards", HAZARDS)]:
        results[f"after_{name}"] = test_suite(brain, rec, f"after-learning/{name}", maker, seed=200)
        results[f"random_{name}"] = test_suite(brain, None, f"random-agent/{name}", maker, seed=200, random_agent=True)
        print(name, results[f"after_{name}"], flush=True)
    results["after_mirrored"] = test_suite(brain, rec, "after-learning/mirrored", ROOMS, seed=200, mirrored=True)
    shows.append(showcase(brain, rec, "unfamiliar/maze", MAZES, 5))
    shows.append(showcase(brain, rec, "unfamiliar/hazards", HAZARDS, 3))
    shows.append(showcase(brain, rec, "unfamiliar/mirrored", ROOMS, 7, mirrored=True))

    # keep learning inside the unfamiliar situations, measure recovery
    for name, maker, mir in [("hazards", HAZARDS, False), ("mirrored", ROOMS, True), ("mazes", MAZES, False)]:
        brain.net.load(trained); brain.target.load(trained)
        adapt = []
        for k in range(6):
            for ep in range(100):
                grid, s, g = maker(rng)
                episode(brain, World(grid, s, g, mirrored=mir), 0.1, True, rec=rec, tag=f"adapt-{name}", ep=k * 100 + ep)
            t = test_suite(brain, rec, f"adapt-{name}-{(k+1)*100}", maker, n=40, seed=300, mirrored=mir)
            adapt.append(dict(t, adapt_episodes=(k + 1) * 100))
        results[f"adapt_{name}"] = adapt
        # does adapting to the new situation damage the old skill?
        results[f"adapt_{name}_then_rooms"] = test_suite(brain, rec, f"adapt-{name}-then-rooms", ROOMS, seed=200)
        shows.append(showcase(brain, rec, f"adapted/{name}", maker, {"hazards": 3, "mirrored": 7, "mazes": 5}[name], mirrored=mir))
        print("adapt", name, [round(a["success_rate"], 2) for a in adapt], "rooms after:", results[f"adapt_{name}_then_rooms"]["success_rate"], flush=True)
    results["train_curve"] = curve
    with open(os.path.join(RES, "unfamiliar_summary.json"), "w") as f:
        json.dump(results, f)
    with open(os.path.join(RES, "unfamiliar_showcases.json"), "w") as f:
        json.dump(shows, f)
    rec.close()


if __name__ == "__main__":
    main()
