"""Experiment 1: Chess.

The brain sees a chess position only as which piece stands on which square.
It is told the rules (which moves are legal) and, at the end of each game,
whether it won, lost or drew. Nothing else: no piece values, no openings,
no tactics. It learns a position-evaluation network from its own games and
picks moves by evaluating the position each legal move would produce.

Everything measured here comes from games actually played; a separate
referee (not part of the brain) labels blunders and puzzle solutions."""
import json, os, random, time
import numpy as np
import chess
from core import Net, Recorder, ROOT

RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
MAX_PLIES = 160
rng = random.Random(7)
nprng = np.random.default_rng(7)


def encode(board, me):
    """Indices of active inputs (768 = 12 piece kinds x 64 squares), seen from `me`."""
    idx = []
    for sq, p in board.piece_map().items():
        s = sq if me == chess.WHITE else chess.square_mirror(sq)
        ch = (p.piece_type - 1) + (0 if p.color == me else 6)
        idx.append(ch * 64 + s)
    return idx


def dense(list_of_idx):
    x = np.zeros((len(list_of_idx), 768), np.float32)
    for r, idx in enumerate(list_of_idx):
        x[r, idx] = 1.0
    return x


def material(board, color):
    m = 0
    for p in board.piece_map().values():
        m += VAL[p.piece_type] * (1 if p.color == color else -1)
    return m


class ChessBrain:
    def __init__(self):
        self.net = Net([768, 64, 32, 1], seed=1, lr=1e-3)
        self.memory_x, self.memory_y = [], []  # replay memory of experienced positions
        self.mem_cap = 150_000

    def consider(self, board):
        """Evaluate every legal move by the position it leads to."""
        me = board.turn
        moves = list(board.legal_moves)
        feats = []
        for m in moves:
            board.push(m)
            feats.append(encode(board, me))
            board.pop()
        out, acts = self.net.forward(dense(feats))
        return moves, out[:, 0], feats, acts

    def choose(self, board, eps):
        t0 = time.perf_counter()
        moves, vals, feats, acts = self.consider(board)
        explored = rng.random() < eps
        i = rng.randrange(len(moves)) if explored else int(np.argmax(vals))
        ms = (time.perf_counter() - t0) * 1000
        order = np.argsort(-vals)[:3]
        alts = [(moves[j].uci(), round(float(vals[j]), 4)) for j in order]
        return dict(move=moves[i], value=float(vals[i]), feat=feats[i], explored=explored, ms=ms,
                    alts=alts, n_legal=len(moves), hidden=acts[1][i], hidden2=acts[2][i])

    def remember(self, feats, targets):
        self.memory_x.extend(feats)
        self.memory_y.extend(targets)
        if len(self.memory_x) > self.mem_cap:
            cut = len(self.memory_x) - self.mem_cap
            del self.memory_x[:cut], self.memory_y[:cut]

    def learn(self, batches=8, bs=256):
        if len(self.memory_x) < bs:
            return None
        losses = []
        for _ in range(batches):
            ids = nprng.integers(0, len(self.memory_x), bs)
            x = dense([self.memory_x[i] for i in ids])
            y = np.array([self.memory_y[i] for i in ids], np.float32)[:, None]
            def g(out):
                d = out - y
                return float((d ** 2).mean()), (2 * d / len(d)).astype(np.float32)
            losses.append(self.net.train(x, g))
        return float(np.mean(losses)), ids[:5].tolist()

    def hidden_of(self, board, me):
        _, acts = self.net.forward(dense([encode(board, me)]))
        return acts[1][0]


# ---------- opponents (fixed, not learning) ----------
def opp_random(board):
    return rng.choice(list(board.legal_moves))


def opp_greedy(board):
    """Captures the most valuable piece it can; otherwise moves randomly."""
    caps = [m for m in board.legal_moves if board.is_capture(m)]
    if not caps:
        return opp_random(board)
    def v(m):
        p = board.piece_at(m.to_square)
        return VAL[p.piece_type] if p else 1
    best = max(v(m) for m in caps)
    return rng.choice([m for m in caps if v(m) == best])


def opp_material(board):
    """Looks one reply ahead on material only (a classic simple engine heuristic)."""
    me = board.turn
    best, bestm = -1e9, []
    for m in board.legal_moves:
        board.push(m)
        if board.is_checkmate():
            board.pop(); return m
        score = material(board, me) - opponent_best_gain(board)
        board.pop()
        if score > best:
            best, bestm = score, [m]
        elif score == best:
            bestm.append(m)
    return rng.choice(bestm)


OPPONENTS = {"random": opp_random, "greedy-capture": opp_greedy, "material-lookahead": opp_material}


# ---------- referee (labels, not part of the brain) ----------
def opponent_best_gain(board):
    """Best immediate material the side to move can win with one capture."""
    best = 0
    for m in board.legal_moves:
        if not board.is_capture(m):
            continue
        tgt = board.piece_at(m.to_square)
        gain = VAL[tgt.piece_type] if tgt else 1
        mover = board.piece_at(m.from_square)
        board.push(m)
        if board.is_attacked_by(board.turn, m.to_square):
            gain -= VAL[mover.piece_type]
        board.pop()
        best = max(best, gain)
    return best


def mate_in_one_moves(board):
    out = []
    for m in board.legal_moves:
        board.push(m)
        if board.is_checkmate():
            out.append(m)
        board.pop()
    return out


def is_mate_in_two_move(board, m):
    board.push(m)
    ok = True
    if board.is_game_over():
        ok = board.is_checkmate()
    else:
        for r in list(board.legal_moves):
            board.push(r)
            if not mate_in_one_moves(board):
                ok = False
            board.pop()
            if not ok:
                break
    board.pop()
    return ok


# ---------- games ----------
def play(brain, opponent, brain_color, eps, start=None, record=False, rec=None, gid=None, phase="train"):
    board = start.copy() if start is not None else chess.Board()
    feats = {chess.WHITE: [], chess.BLACK: []}
    trace = []
    blunders = missed_mates = 0
    selfplay = opponent == "self"
    while not board.is_game_over() and board.ply() < MAX_PLIES:
        mover = board.turn
        if selfplay or mover == brain_color:
            mates = mate_in_one_moves(board) if record else []
            mat_before = material(board, mover)
            c = brain.choose(board, eps)
            m = c["move"]
            san = board.san(m)
            fen_before = board.fen()
            board.push(m)
            feats[mover].append(c["feat"])
            step = dict(ply=board.ply(), fen_before=fen_before, uci=m.uci(), san=san, explored=c["explored"],
                        value=round(c["value"], 4), alts=c["alts"], n_legal=c["n_legal"], ms=round(c["ms"], 2))
            if record:
                gain = opponent_best_gain(board) if not board.is_game_over() else 0
                blunder = gain >= 3
                missed = bool(mates) and not board.is_checkmate()
                blunders += blunder
                missed_mates += missed
                step.update(blunder=blunder, hangs=gain, missed_mate=missed, fen_after=board.fen(),
                            material=material(board, brain_color),
                            hidden=np.round(c["hidden"], 3).tolist(), hidden2=np.round(c["hidden2"], 3).tolist())
                trace.append(step)
            if rec is not None:
                rec.log("move", phase=phase, game=gid, ply=step["ply"], uci=step["uci"], explored=step["explored"],
                        value=step["value"], alts=step["alts"], n_legal=step["n_legal"], ms=step["ms"],
                        **({"blunder": step["blunder"], "missed_mate": step["missed_mate"]} if record else {}))
        else:
            m = OPPONENTS[opponent](board)
            if record:
                trace.append(dict(ply=board.ply() + 1, uci=m.uci(), san=board.san(m), opponent=True))
            board.push(m)
            if record:
                trace[-1].update(fen_after=board.fen(), material=material(board, brain_color))
    res = board.result(claim_draw=False) if board.is_game_over() else "1/2-1/2"
    ending = (board.outcome().termination.name.lower() if board.outcome() else "move-limit")
    z = {"1-0": 1, "0-1": -1}.get(res, 0)
    outcome = {}
    for col, sgn in ((chess.WHITE, 1), (chess.BLACK, -1)):
        outcome[col] = z * sgn
    return dict(feats=feats, outcome=outcome, result=res, ending=ending, plies=board.ply(),
                final_material=material(board, brain_color), trace=trace,
                blunders=blunders, missed_mates=missed_mates, final_fen=board.fen())


def targets_for(feats, z, gamma=0.985):
    n = len(feats)
    return [z * gamma ** (n - 1 - i) for i in range(n)]


# ---------- puzzles ----------
def make_puzzles(seed=99, budget_s=240):
    """Positions from games the brain never played (different seed, random/greedy players)."""
    r = random.Random(seed)
    m1, hang, m2 = [], [], []
    t0 = time.time()
    while time.time() - t0 < budget_s and (len(m1) < 100 or len(hang) < 100 or len(m2) < 30):
        b = chess.Board()
        while not b.is_game_over() and b.ply() < 200:
            mv = list(b.legal_moves)
            caps = [m for m in mv if b.is_capture(m)]
            b.push(r.choice(caps) if caps and r.random() < 0.5 else r.choice(mv))
            if b.is_game_over() or b.ply() < 10 or r.random() > 0.25:
                continue
            mates = mate_in_one_moves(b)
            if mates:
                if len(m1) < 100:
                    m1.append(dict(fen=b.fen(), solutions=[m.uci() for m in mates]))
                continue
            qcaps = []
            for m in b.legal_moves:
                p = b.piece_at(m.to_square)
                if p and p.piece_type == chess.QUEEN and p.color != b.turn and not b.is_attacked_by(not b.turn, m.to_square):
                    qcaps.append(m.uci())
            if qcaps and len(hang) < 100:
                hang.append(dict(fen=b.fen(), solutions=qcaps))
            elif len(m2) < 30 and material(b, b.turn) >= 6 and b.legal_moves.count() <= 40:
                sols = [m.uci() for m in list(b.legal_moves) if b.gives_check(m) and is_mate_in_two_move(b, m)]
                if sols:
                    m2.append(dict(fen=b.fen(), solutions=sols))
    return {"mate_in_1": m1, "win_free_queen": hang, "mate_in_2": m2}


def solve_rate(brain, puzzles):
    out = {}
    for k, ps in puzzles.items():
        ok = 0
        for p in ps:
            b = chess.Board(p["fen"])
            c = brain.choose(b, 0.0)
            ok += c["move"].uci() in p["solutions"]
        base = np.mean([len(p["solutions"]) / chess.Board(p["fen"]).legal_moves.count() for p in ps]) if ps else 0
        out[k] = dict(solved=ok, total=len(ps), rate=ok / max(1, len(ps)), chance_rate=float(base))
    return out


def evaluate(brain, rec, tag, n_games=40, puzzles=None, probes=None, opponents=("random", "greedy-capture"),
             chess960=False, showcase=False):
    res = {"tag": tag}
    for opp in opponents:
        W = D = L = 0
        mats, bl, mm, moves, ms, endings = [], 0, 0, 0, [], {}
        best_trace = None
        for g in range(n_games):
            col = chess.WHITE if g % 2 == 0 else chess.BLACK
            start = None
            if chess960:
                start = chess.Board.from_chess960_pos(rng.randrange(960))
                start.chess960 = True
            r = play(brain, opp, col, 0.0, start=start, record=True, rec=rec, gid=f"{tag}-{opp}-{g}", phase="eval")
            sc = r["outcome"][col]
            W += sc == 1; D += sc == 0; L += sc == -1
            mats.append(r["final_material"]); bl += r["blunders"]; mm += r["missed_mates"]
            nb = sum(1 for s in r["trace"] if not s.get("opponent"))
            moves += nb
            ms += [s["ms"] for s in r["trace"] if not s.get("opponent")]
            endings[r["ending"]] = endings.get(r["ending"], 0) + 1
            rec.log("game_end", phase="eval", tag=tag, opponent=opp, game=g, color="white" if col else "black",
                    result=r["result"], score=sc, ending=r["ending"], plies=r["plies"],
                    final_material=r["final_material"], blunders=r["blunders"], missed_mates=r["missed_mates"])
            if showcase:
                r2 = dict(r); r2.pop("feats"); r2["color"] = "white" if col else "black"; r2["opponent"] = opp; r2["score"] = sc
                os.makedirs(os.path.join(RES, "chess_games"), exist_ok=True)
                with open(os.path.join(RES, "chess_games", f"{tag}_{opp}_{g}.json"), "w") as f:
                    json.dump(r2, f)
        key = opp + ("-chess960" if chess960 else "")
        res[key] = dict(win=W, draw=D, loss=L, games=n_games, score=(W + 0.5 * D) / n_games,
                        mean_final_material=float(np.mean(mats)), blunders_per_100_moves=100 * bl / max(1, moves),
                        missed_mates=mm, mean_decision_ms=float(np.mean(ms)), endings=endings)
    if puzzles:
        res["puzzles"] = solve_rate(brain, puzzles)
    if probes is not None:
        res["probe_hidden"] = np.round(np.array([brain.hidden_of(chess.Board(f), chess.Board(f).turn) for f in probes]), 3).tolist()
    rec.log("evaluation", **{k: v for k, v in res.items() if k != "probe_hidden"})
    return res


def main(n_train_games=3000, eval_every=500):
    rec = Recorder("chess")
    brain = ChessBrain()
    t0 = time.time()
    puzzles = make_puzzles()
    rec.log("puzzles_created", **{k: len(v) for k, v in puzzles.items()})
    print("puzzles", {k: len(v) for k, v in puzzles.items()}, f"{time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(RES, "chess_puzzles.json"), "w") as f:
        json.dump(puzzles, f)
    probes = [p["fen"] for p in puzzles["win_free_queen"][:24]]

    evals = [evaluate(brain, rec, "game0000", puzzles=puzzles, probes=probes, showcase=True,
                      opponents=("random", "greedy-capture", "material-lookahead"))]
    evals[-1]["chess960"] = evaluate(brain, rec, "game0000-960", n_games=20, chess960=True, opponents=("random",))
    print("eval0", json.dumps({k: v for k, v in evals[-1].items() if k not in ("probe_hidden", "chess960")})[:600], flush=True)
    curve = []
    for g in range(1, n_train_games + 1):
        eps = max(0.05, 0.3 * (1 - g / (0.6 * n_train_games)))
        u = rng.random()
        opp = "self" if u < 0.4 else ("random" if u < 0.8 else "greedy-capture")
        col = rng.choice([chess.WHITE, chess.BLACK])
        r = play(brain, opp, col, eps, rec=rec, gid=g, phase="train")
        for c in (chess.WHITE, chess.BLACK):
            if r["feats"][c]:
                brain.remember(r["feats"][c], targets_for(r["feats"][c], r["outcome"][c]))
        lr = brain.learn()
        loss = lr[0] if lr else None
        rec.log("game_end", phase="train", game=g, opponent=opp, color="white" if col else "black", result=r["result"],
                score=r["outcome"][col], ending=r["ending"], plies=r["plies"], eps=round(eps, 3), loss=loss,
                memory_size=len(brain.memory_x), memory_samples_drawn=8 * 256 if lr else 0)
        curve.append(dict(game=g, opponent=opp, score=r["outcome"][col], loss=loss, eps=eps, plies=r["plies"]))
        if g % 100 == 0:
            recent = [c["score"] for c in curve[-100:] if c["opponent"] == "random"]
            print(f"g{g} {time.time()-t0:.0f}s eps={eps:.2f} loss={loss:.4f} vs-random-train={np.mean(recent):.2f}", flush=True)
        if g % eval_every == 0:
            final = g == n_train_games
            e = evaluate(brain, rec, f"game{g:04d}", puzzles=puzzles, probes=probes, showcase=final,
                         opponents=("random", "greedy-capture", "material-lookahead") if final else ("random", "greedy-capture"))
            if final:
                e["chess960"] = evaluate(brain, rec, f"game{g:04d}-960", n_games=20, chess960=True, opponents=("random",))
            evals.append(e)
            brain.net.save(os.path.join(RES, f"chess_net_{g:04d}.npz"))
            print("eval", g, json.dumps({k: v for k, v in e.items() if k not in ("probe_hidden", "chess960")})[:700], flush=True)
    with open(os.path.join(RES, "chess_summary.json"), "w") as f:
        json.dump(dict(evals=evals, train_curve=curve, n_train_games=n_train_games, probes=probes), f)
    rec.close()


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000, int(sys.argv[2]) if len(sys.argv) > 2 else 500)
