"""Renders the ~60 s research film from recorded experiment data only.
Every board, activation map, program, test result and chart below is drawn
from files in ../results and ../logs; nothing is staged or invented."""
import json, math, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import chess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES, LOGS = os.path.join(ROOT, "results"), os.path.join(ROOT, "logs")
sys.path.insert(0, HERE)
import audio

W, H, FPS, DUR = 1920, 1080, 30, 60.0
BG = (9, 12, 17)
INK = (226, 230, 236)
DIM = (128, 140, 156)
FAINT = (52, 62, 76)
CYAN = (90, 200, 250)
AMBER = (245, 177, 76)
RED = (255, 107, 107)
GREEN = (123, 216, 143)
VIOLET = (170, 140, 255)
WATERMARK = "Amaresh | Student and Open Researcher"
F = "C:/Windows/Fonts/"


def font(name, size):
    return ImageFont.truetype(F + name, size)


FT = {k: font(*v) for k, v in {
    "title": ("segoeuil.ttf", 92), "h1": ("segoeuisl.ttf", 54), "h2": ("segoeuisl.ttf", 38), "body": ("segoeui.ttf", 30),
    "small": ("segoeui.ttf", 25), "tiny": ("segoeui.ttf", 21), "mono": ("consola.ttf", 27), "monos": ("consola.ttf", 22),
    "label": ("segoeuib.ttf", 21), "num": ("segoeuil.ttf", 64), "piece": ("seguisym.ttf", 70), "wm": ("segoeui.ttf", 27),
}.items()}

SECTIONS = [(0, 7, "OPENING"), (7, 17, "BRAIN ACTIVITY"), (17, 29, "EXPERIMENT 1 · CHESS"),
            (29, 41, "EXPERIMENT 2 · PROGRAMMING"), (41, 53, "LEARNING & GENERALIZATION"), (53, 60, "SUMMARY")]
CUES = []          # (time, sound, gain) collected while rendering
MANIFEST = []      # film time -> source data


def lerp(a, b, t):
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def ease(x):
    x = min(1, max(0, x))
    return x * x * (3 - 2 * x)


class Frame:
    def __init__(self, t):
        self.t = t
        self.img = Image.new("RGB", (W, H), BG)
        self.d = ImageDraw.Draw(self.img)
        self.A = 1.0

    def c(self, col, a=1.0):
        return lerp(BG, col, max(0, min(1, a * self.A)))

    def text(self, xy, s, f="body", col=INK, a=1.0, anchor="la", **kw):
        self.d.text(xy, s, font=FT[f], fill=self.c(col, a), anchor=anchor, **kw)

    def rect(self, box, col, a=1.0, fill=True, r=0, width=2):
        if fill:
            self.d.rounded_rectangle(box, r, fill=self.c(col, a))
        else:
            self.d.rounded_rectangle(box, r, outline=self.c(col, a), width=width)

    def line(self, pts, col, a=1.0, width=2):
        self.d.line(pts, fill=self.c(col, a), width=width)


def background(fr):
    for x in range(0, W, 96):
        fr.d.line([(x, 0), (x, H - 70)], fill=lerp(BG, FAINT, 0.28))
    for y in range(0, H - 70, 96):
        fr.d.line([(0, y), (W, y)], fill=lerp(BG, FAINT, 0.28))


def chrome(fr, sec_name, source):
    """Persistent frame elements: section label, data-source line, watermark."""
    d = fr.d
    d.rectangle([0, H - 66, W, H], fill=(6, 8, 12))
    d.line([(0, H - 66), (W, H - 66)], fill=(40, 48, 60), width=1)
    d.text((W // 2, H - 33), WATERMARK, font=FT["wm"], fill=(214, 220, 228), anchor="mm")
    d.text((40, H - 33), sec_name, font=FT["tiny"], fill=(120, 132, 148), anchor="lm")
    if source:
        d.text((W - 40, H - 33), source, font=FT["tiny"], fill=(120, 132, 148), anchor="rm")
    d.text((W - 40, 30), "RECORDED DATA · NOT STAGED", font=FT["tiny"], fill=(100, 112, 128), anchor="ra")
    d.text((40, 30), f"{fr.t:05.2f}s", font=FT["tiny"], fill=(100, 112, 128), anchor="la")


def heat_rgb(v):
    """Diverging map: -1 cool blue, 0 near-background, +1 amber."""
    v = np.clip(v, -1, 1)[..., None]
    neg, mid, pos = np.array([70, 140, 230.]), np.array([22, 28, 36.]), np.array([245, 177, 76.])
    return np.where(v < 0, mid + (neg - mid) * (-v), mid + (pos - mid) * v).astype(np.uint8)


def paste_heat(fr, arr, box, alpha=1.0):
    x0, y0, x1, y1 = box
    rgb = heat_rgb(np.asarray(arr, float))
    im = Image.fromarray(rgb).resize((x1 - x0, y1 - y0), Image.NEAREST)
    if alpha * fr.A < 1:
        bgim = Image.new("RGB", im.size, BG)
        im = Image.blend(bgim, im, alpha * fr.A)
    fr.img.paste(im, (x0, y0))


# ---------------- data ----------------
def load():
    D = {}
    D["chess"] = json.load(open(os.path.join(RES, "chess_summary.json")))
    D["code"] = json.load(open(os.path.join(RES, "code_summary.json")))
    D["nav"] = json.load(open(os.path.join(RES, "unfamiliar_summary.json")))
    D["navshow"] = {s["tag"]: s for s in json.load(open(os.path.join(RES, "unfamiliar_showcases.json")))}
    ev = D["chess"]["evals"]
    final_tag = ev[-1]["tag"]
    gdir = os.path.join(RES, "chess_games")
    games = [json.load(open(os.path.join(gdir, f))) | {"file": f} for f in sorted(os.listdir(gdir)) if f.startswith(final_tag + "_")]
    D["games_final"] = games
    D["games_before"] = [json.load(open(os.path.join(gdir, f))) | {"file": f} for f in sorted(os.listdir(gdir)) if f.startswith("game0000_")]
    D["code_events"] = [json.loads(l) for l in open(os.path.join(LOGS, "code_events.jsonl"))]
    D["chess_events_index"] = index_chess_events()
    return D


def index_chess_events():
    """Map (game id, ply) -> (seq, t) in the chess event log, for on-screen sync references."""
    idx = {}
    with open(os.path.join(LOGS, "chess_events.jsonl")) as f:
        for l in f:
            if '"phase": "eval"' in l and '"kind": "move"' in l:
                e = json.loads(l)
                idx[(e["game"], e["ply"])] = (e["seq"], e["t"])
    return idx


def pick_showcase_game(games):
    """Prefer a checkmate win; otherwise the game where the brain finished furthest ahead."""
    wins = [g for g in games if g["score"] == 1 and g["ending"] == "checkmate"]
    pool = wins or games
    return max(pool, key=lambda g: (g["score"], g["final_material"]))


# ---------------- chess drawing ----------------
GLYPH = {"k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟"}


def draw_board(fr, fen, x0, y0, size, hl=None, hl_col=CYAN, flip=False, a=1.0):
    b = chess.Board(fen)
    sq = size // 8
    for r in range(8):
        for f_ in range(8):
            light = (r + f_) % 2 == 0
            col = (84, 96, 112) if light else (52, 62, 76)
            s = chess.square(f_, 7 - r) if not flip else chess.square(7 - f_, r)
            x, y = x0 + f_ * sq, y0 + r * sq
            if hl and s in hl:
                col = lerp(col, hl_col, 0.45)
            fr.d.rectangle([x, y, x + sq - 1, y + sq - 1], fill=fr.c(col, a))
            p = b.piece_at(s)
            if p:
                g = GLYPH[p.symbol().lower()]
                fill = (240, 236, 226) if p.color else (16, 18, 22)
                stroke = (16, 18, 22) if p.color else (200, 205, 214)
                fr.d.text((x + sq / 2, y + sq / 2 + 3), g, font=FT["piece"], anchor="mm",
                          fill=fr.c(fill, a), stroke_width=2, stroke_fill=fr.c(stroke, a))
    fr.rect([x0 - 2, y0 - 2, x0 + 8 * sq + 1, y0 + 8 * sq + 1], FAINT, a, fill=False, width=2)


# ---------------- sections ----------------
def title_block(fr, kicker, title, a=1.0, y=110):
    fr.text((120, y), kicker, "label", CYAN, a)
    fr.text((120, y + 30), title, "h1", INK, a)


def s_opening(fr, t, D):
    g = D["show_game"]
    hid = np.array([s["hidden"] for s in g["trace"] if not s.get("opponent")])
    # faint field of real hidden activations from the showcase game, drifting slowly
    cols = min(len(hid), 60)
    off = int(t * 2) % max(1, len(hid) - cols + 1)
    paste_heat(fr, hid[off:off + cols].T, (0, 70, W, H - 300), alpha=0.22 * ease(t / 2))
    fr.text((W // 2, 360), "A Digital Brain", "title", INK, ease((t - 0.4) / 1.2), anchor="mm")
    fr.text((W // 2, 450), "An ongoing home research experiment in learning, memory and generalization",
            "h2", DIM, ease((t - 1.0) / 1.0), anchor="mm")
    fr.text((W // 2, 540), "Inspired by Einstein's curiosity and mathematical imagination.",
            "body", AMBER, ease((t - 1.8) / 1.0), anchor="mm")
    fr.text((W // 2, 640), "Three experiments  ·  chess  ·  programming challenges  ·  unfamiliar worlds",
            "body", INK, ease((t - 2.6) / 0.9), anchor="mm")
    fr.text((W // 2, 700), "The experiments measure what it actually learns.",
            "small", DIM, ease((t - 3.2) / 0.9), anchor="mm")
    fr.text((W // 2, H - 250), "background: real hidden-unit activations recorded during one of its chess games",
            "tiny", DIM, 0.8 * ease((t - 2) / 1.5), anchor="mm")
    return "logs/chess_events.jsonl · results/chess_games"


def s_activity(fr, t, D):
    g = D["show_game"]
    mine = [s for s in g["trace"] if not s.get("opponent")]
    title_block(fr, "OBSERVABLE INTERNAL ACTIVITY", "What changes inside while it decides", ease(t / 0.8))
    if t < 5.6:
        # live: board + 64 hidden units for each of its decisions, in sync
        n = len(mine)
        start = max(0, n - 22)
        k = min(n - 1, start + int(t / 5.6 * (n - start)))
        s = mine[k]
        a = ease(t / 0.6)
        draw_board(fr, s["fen_after"], 120, 250, 560, hl={chess.parse_square(s["uci"][:2]), chess.parse_square(s["uci"][2:4])}, a=a)
        fr.text((120, 830), f"its move {s['san']}   ·   own evaluation {s['value']:+.2f}   ·   {s['ms']:.1f} ms",
                "small", INK, a)
        hv = np.array(s["hidden"])
        cx0, cy0, gap = 790, 270, 62
        fr.text((790, 220), "64 hidden units · evaluation network, layer 1", "small", DIM, a)
        for i, v in enumerate(hv):
            x, y = cx0 + (i % 8) * gap, cy0 + 20 + (i // 8) * gap
            rad = 10 + 16 * abs(v)
            col = tuple(int(c) for c in heat_rgb(np.array([v]))[0])
            fr.d.ellipse([x + 26 - rad, y + 26 - rad, x + 26 + rad, y + 26 + rad], fill=fr.c(col, a))
        hist = np.array([m["hidden"] for m in mine[:k + 1]])[-40:]
        fr.text((1360, 220), "same units, last 40 decisions (time →)", "small", DIM, a)
        paste_heat(fr, hist.T, (1360, 290, 1360 + 12 * len(hist), 290 + 7 * 64), a)
        fr.text((790, 840), "Directly measured: activations recorded at every decision.", "small", GREEN, a)
        fr.text((790, 878), "Not measured, not claimed: thoughts, feelings, awareness.", "small", RED, a)
        idx = D["chess_events_index"].get((g["gid"], s["ply"]))
        if int(t * FPS) % 1 == 0 and k != D.setdefault("_lastk", -1):
            D["_lastk"] = k
            CUES.append((fr.t, "tick", 0.12))
        return f"logs/chess_events.jsonl · seq {idx[0]} · t={idx[1]:.2f}s" if idx else "results/chess_games"
    # before vs after learning on identical positions
    t2 = t - 5.6
    a = ease(t2 / 0.6)
    ev = D["chess"]["evals"]
    before, after = np.array(ev[0]["probe_hidden"]), np.array(ev[-1]["probe_hidden"])
    fr.text((120, 245), "The same 64 units on the same 24 unseen puzzle positions,", "body", INK, a)
    fr.text((120, 285), f"before learning (left) and after {D['chess']['n_train_games']:,} practice games (right)", "body", INK, a)
    paste_heat(fr, before.T, (120, 350, 120 + 24 * 30, 350 + 64 * 7), a)
    paste_heat(fr, after.T, (1000, 350, 1000 + 24 * 30, 350 + 64 * 7), a)
    fr.text((120, 815), "before: random starting weights", "small", DIM, a)
    fr.text((1000, 815), "after: weights shaped only by wins, losses and draws", "small", DIM, a)
    ch = float(np.mean(np.abs(after - before)))
    fr.text((120, 870), f"Mean absolute change per unit: {ch:.2f}. A change in activity does not show understanding.",
            "small", AMBER, ease((t2 - 1.0) / 0.8))
    return "results/chess_summary.json · probe activations"


def s_chess(fr, t, D):
    g = D["show_game"]
    ev = D["chess"]["evals"]
    e0, e1 = ev[0], ev[-1]
    title_block(fr, "EXPERIMENT 1 · CHESS", "Told only the rules and, at the end, win / loss / draw", ease(t / 0.8))
    if t < 7.2:
        plies = g["trace"][-16:]
        k = min(len(plies) - 1, int(t / 6.6 * len(plies)))
        s = plies[k]
        a = ease(t / 0.5)
        hl = {chess.parse_square(s["uci"][:2]), chess.parse_square(s["uci"][2:4])}
        draw_board(fr, s["fen_after"], 120, 240, 600, hl=hl, hl_col=AMBER if s.get("opponent") else CYAN,
                   flip=g["color"] == "black", a=a)
        if k != D.setdefault("_lastc", -1):
            D["_lastc"] = k
            CUES.append((fr.t, "tick", 0.14))
        x = 790
        fr.text((x, 240), f"Final evaluation game {D['show_game_no']} · brain plays {g['color']} vs {g['opponent']} opponent", "body", INK, a)
        mover = "opponent" if s.get("opponent") else "brain"
        fr.text((x, 300), f"ply {s['ply']} · {mover} plays {s['san']}", "h2", AMBER if mover == "opponent" else CYAN, a)
        mine = [p for p in plies[:k + 1] if not p.get("opponent")]
        s = mine[-1] if mine else s
        if not s.get("opponent"):
            fr.text((x, 370), f"its decision at ply {s['ply']} ({s['san']}): {s['n_legal']} legal moves compared. Its own scores:", "small", DIM, a)
            for i, (u, v) in enumerate(s["alts"]):
                bx = x + 40 + int(260 * (v + 1) / 2)
                fr.text((x, 410 + i * 42), u, "mono", INK if i == 0 else DIM, a)
                fr.rect([x + 110, 418 + i * 42, x + 110 + max(4, int(300 * (v + 1) / 2)), 438 + i * 42], CYAN if i == 0 else FAINT, a)
                fr.text((x + 430, 410 + i * 42), f"{v:+.3f}", "mono", INK if i == 0 else DIM, a)
            fr.text((x, 545), f"decision time {s['ms']:.1f} ms · exploration off in evaluation", "small", DIM, a)
            flag = "referee: left a piece hanging (blunder)" if s.get("blunder") else ("referee: missed a mate in one" if s.get("missed_mate") else "referee: no blunder flagged")
            fr.text((x, 585), flag, "small", RED if (s.get("blunder") or s.get("missed_mate")) else GREEN, a)
        # material curve for the whole game
        mats = [p["material"] for p in g["trace"]]
        mx = max(10, max(abs(m) for m in mats))
        gx0, gy0, gw, gh = x, 650, 900, 180
        fr.line([(gx0, gy0 + gh / 2), (gx0 + gw, gy0 + gh / 2)], FAINT, a)
        n = len(mats)
        upto = len(g["trace"]) - len(plies) + k + 1
        pts = [(gx0 + gw * i / max(1, n - 1), gy0 + gh / 2 - gh / 2 * m / mx) for i, m in enumerate(mats[:upto])]
        if len(pts) > 1:
            fr.line(pts, CYAN, a, 3)
        fr.text((gx0, gy0 + gh + 8), "material balance from the brain's side across the whole game (referee count)", "tiny", DIM, a)
        if k == len(plies) - 1:
            res = {1: "WIN", 0: "DRAW", -1: "LOSS"}[g["score"]]
            fr.text((120, 870), f"Result: {res} by {g['ending'].replace('_', ' ')} after {g['plies']} plies", "h2",
                    GREEN if g["score"] == 1 else AMBER, ease((t - 6.0) / 0.4))
            if "_chessres" not in D:
                D["_chessres"] = 1
                CUES.append((fr.t, "pass", 0.25))
        idx = D["chess_events_index"].get((g["gid"], s["ply"]))
        return f"logs/chess_events.jsonl · seq {idx[0]} · t={idx[1]:.2f}s" if idx else f"results/chess_games/{g['file']}"
    # results: before vs after
    t2 = t - 7.2
    a = ease(t2 / 0.5)
    fr.text((120, 245), f"Evaluation games and unseen puzzles, before learning and after {D['chess']['n_train_games']:,} games",
            "body", INK, a)
    rows = [("score vs random mover", e0["random"]["score"], e1["random"]["score"], "{:.0%}"),
            ("score vs greedy capturer", e0["greedy-capture"]["score"], e1["greedy-capture"]["score"], "{:.0%}"),
            ("score vs 1-move material engine", e0["material-lookahead"]["score"], e1["material-lookahead"]["score"], "{:.0%}"),
            ("win the free queen (100 puzzles)", e0["puzzles"]["win_free_queen"]["rate"], e1["puzzles"]["win_free_queen"]["rate"], "{:.0%}"),
            ("mate in one (100 puzzles)", e0["puzzles"]["mate_in_1"]["rate"], e1["puzzles"]["mate_in_1"]["rate"], "{:.0%}"),
            ("mate in two (30 puzzles)", e0["puzzles"]["mate_in_2"]["rate"], e1["puzzles"]["mate_in_2"]["rate"], "{:.0%}")]
    for i, (lab, b, af, fmt) in enumerate(rows):
        y = 320 + i * 82
        ai = ease((t2 - 0.2 - i * 0.18) / 0.5)
        fr.text((120, y), lab, "body", INK, ai)
        fr.rect([700, y + 4, 700 + int(520 * b), y + 22], DIM, ai)
        fr.rect([700, y + 30, 700 + int(520 * af * ai), y + 48], CYAN, ai)
        fr.text((1240, y), "before " + fmt.format(b), "small", DIM, ai)
        fr.text((1240, y + 28), "after  " + fmt.format(af), "small", CYAN, ai)
    bb, ba = e0["random"]["blunders_per_100_moves"], e1["random"]["blunders_per_100_moves"]
    mb, ma = e0["random"]["mean_final_material"], e1["random"]["mean_final_material"]
    fr.text((120, 830), f"vs random: blunders per 100 moves {bb:.0f} → {ba:.0f} · final material {mb:+.1f} → {ma:+.1f} · score = (wins + ½ draws) / games",
            "small", AMBER, ease((t2 - 1.6) / 0.5))
    cr = {k: v["chance_rate"] for k, v in e1["puzzles"].items()}
    fr.text((120, 870), f"Puzzle score = its single top choice. Random-guess level: free queen {cr['win_free_queen']:.0%}, mate in one {cr['mate_in_1']:.0%}, mate in two {cr['mate_in_2']:.0%}.",
            "small", DIM, ease((t2 - 1.9) / 0.5))
    return "results/chess_summary.json · evaluations"


def s_code(fr, t, D):
    title_block(fr, "EXPERIMENT 2 · PROGRAMMING CHALLENGES", "It sees only examples and test feedback", ease(t / 0.8))
    C = D["code"]
    evs = [e for e in D["code_events"] if e.get("phase") == "experienced-unseen" and e.get("problem") == "distinctletters"]
    prob = C["problems"]["distinctletters"]
    if t < 8.0:
        a = ease(t / 0.5)
        fr.rect([120, 240, 760, 860], (16, 21, 28), a, r=14)
        fr.text((150, 262), "UNSEEN PROBLEM (title shown for you, not to it)", "label", DIM, a)
        fr.text((150, 295), prob["title"], "h2", INK, a)
        fr.text((150, 365), "What the brain is given (3 examples):", "small", DIM, a)
        for i, (x, y) in enumerate(prob["visible"]):
            fr.text((150, 410 + i * 46), f'f("{x}")  →  {y}', "mono", CYAN, a)
        fr.text((150, 575), f"30 hidden tests, revealed only as pass/fail", "small", DIM, a)
        fr.text((150, 610), "plus one failing case per wrong submission.", "small", DIM, a)
        fr.text((150, 680), "Programs are chains of building blocks", "small", DIM, a)
        fr.text((150, 715), "from a fixed vocabulary of 74 blocks.", "small", DIM, a)
        fr.text((150, 760), "It is not writing free-form code, and no", "small", AMBER, a)
        fr.text((150, 795), "language model is involved.", "small", AMBER, a)
        # the attempt log as recorded
        lines = []
        for e in evs:
            if e["kind"] == "attempt_start":
                mem = ", ".join(m[0] for m in e["memory_retrieved"]) or "none"
                lines.append((f"memory recalls similar solved problems: {mem}", DIM, None, e))
            elif e["kind"] == "candidate_failed":
                lines.append((f'× try #{e["checked"]:<3} {" → ".join(e["program"]):<22} f("{e["case_input"]}") gave {e["got"]}, need {e["expected"]}', DIM, None, e))
            elif e["kind"] == "submission":
                ok = "failed_case" not in e
                lines.append((f'» SUBMIT  {" → ".join(e["program"])}', INK, None, e))
                if ok:
                    lines.append((f'   hidden tests {e["hidden_passed"]}/{e["hidden_total"]}  PASS, ACCEPTED  (after {e["checked_so_far"]} candidates)', GREEN, "pass", e))
                else:
                    fc = e["failed_case"]
                    lines.append((f'   hidden tests {e["hidden_passed"]}/{e["hidden_total"]}  WRONG ANSWER on "{fc["input"]}": got {fc["got"]}, need {fc["expected"]}', RED, "fail", e))
                    lines.append(("   → failing case added to its examples; search continues", AMBER, None, e))
        shown = min(len(lines), int(1 + t / 7.2 * len(lines)))
        fr.text((820, 245), "Recorded attempt log", "label", DIM, a)
        y = 285
        for i, (s, col, snd, e) in enumerate(lines[:shown]):
            fr.text((820, y), s, "monos", col, a)
            y += 36
            key = ("_codecue", i)
            if snd and key not in D:
                D[key] = 1
                CUES.append((fr.t, snd, 0.28 if snd == "pass" else 0.4))
        last = lines[shown - 1][3]
        fr.text((820, 860), f'searched with a budget of {C["budget"]} candidates · whole attempt took {evs[-1]["seconds"]*1000:.0f} ms', "small", DIM, a)
        return f"logs/code_events.jsonl · seq {last['seq']} · t={last['t']:.3f}s"
    # practice problems: same problems with fresh tests, before vs after practice
    t2 = t - 8.0
    a = ease(t2 / 0.5)
    nt, et = C["naive_train"], C["exp_train"]
    fr.text((120, 245), "Practice problems re-tested with fresh examples and hidden tests", "body", INK, a)
    fr.text((120, 285), "candidates it had to check before an accepted solution (log scale)", "small", DIM, a)
    pick = ["kadane", "gauss", "triangular", "counteven", "sumeven", "factorial", "maxeven"]
    byid = {r["problem"]: r for r in nt}, {r["problem"]: r for r in et}
    for i, pid in enumerate(pick):
        y = 340 + i * 66
        ai = ease((t2 - 0.15 * i) / 0.4)
        n, e = byid[0][pid], byid[1][pid]
        fr.text((120, y), n["title"], "small", INK, ai)
        def bar(val, solved, yy, col):
            w = int(380 * math.log10(val + 1) / math.log10(C["budget"] + 1))
            fr.rect([560, yy, 560 + w, yy + 18], col if solved else RED, ai)
            fr.text((560 + w + 12, yy - 4), f"{val}" + ("" if solved else " (budget ran out)"), "tiny", col if solved else RED, ai)
        bar(n["checked"], n["solved"], y + 2, DIM)
        bar(e["checked"], e["solved"], y + 26, CYAN)
    fr.text((1250, 340), "grey: no experience", "small", DIM, a)
    fr.text((1250, 375), "cyan: after 4 practice passes", "small", CYAN, a)
    med = lambda rs: int(np.median([r["checked"] for r in rs]))
    fr.text((1250, 450), f"all 35 practice problems", "small", INK, a)
    fr.text((1250, 485), f"solved {sum(r['solved'] for r in nt)} → {sum(r['solved'] for r in et)}", "small", INK, a)
    fr.text((1250, 520), f"median candidates {med(nt)} → {med(et)}", "small", INK, a)
    fr.text((1250, 600), "Still unsolved after practice:", "small", AMBER, a)
    uns = [r["title"] for r in et if not r["solved"]]
    for i, u in enumerate(uns[:3]):
        fr.text((1250, 635 + 34 * i), u, "small", AMBER, a)
    return "results/code_summary.json"


def draw_grid_world(fr, show, x0, y0, cell, upto, a):
    grid = show["grid"]
    n = len(grid)
    for r in range(n):
        for c in range(n):
            v = grid[r][c]
            col = (44, 52, 64) if v == 1 else ((120, 40, 46) if v == 2 else (20, 25, 32))
            fr.d.rectangle([x0 + c * cell, y0 + r * cell, x0 + (c + 1) * cell - 2, y0 + (r + 1) * cell - 2], fill=fr.c(col, a))
    gr, gc = show["goal"]
    fr.d.ellipse([x0 + gc * cell + 6, y0 + gr * cell + 6, x0 + (gc + 1) * cell - 8, y0 + (gr + 1) * cell - 8], fill=fr.c(GREEN, a))
    tr = show["trace"]
    pos = [tuple(s["pos"]) for s in tr[:upto]]
    if upto >= len(tr):
        pass
    # path
    pts = [(x0 + p[1] * cell + cell / 2, y0 + p[0] * cell + cell / 2) for p in pos]
    if upto > 0 and upto <= len(tr):
        # end position after the last shown step
        last = tr[upto - 1]
        dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[last["action"]]
        nxt = (last["pos"][0] + dr, last["pos"][1] + dc)
        if 0 <= nxt[0] < n and 0 <= nxt[1] < n and grid[nxt[0]][nxt[1]] != 1:
            pts.append((x0 + nxt[1] * cell + cell / 2, y0 + nxt[0] * cell + cell / 2))
    if len(pts) > 1:
        fr.line(pts, CYAN, 0.55 * a, 4)
    if pts:
        px, py = pts[-1]
        fr.d.ellipse([px - cell * 0.3, py - cell * 0.3, px + cell * 0.3, py + cell * 0.3], fill=fr.c(CYAN, a))


def s_general(fr, t, D):
    N = D["nav"]
    title_block(fr, "LEARNING & GENERALIZATION", "Dropped into worlds it never trained on", ease(t / 0.8))
    if t < 6.2:
        a = ease(t / 0.5)
        shows = [("after-learning/rooms", "familiar kind of room (new layout)"), ("unfamiliar/maze", "unseen: winding corridor maze")]
        for j, (tag, lab) in enumerate(shows):
            sh = D["navshow"][tag]
            n = len(sh["grid"])
            cell = 500 // n
            x0 = 120 + j * 560
            upto = min(len(sh["trace"]), int(t / 5.6 * len(sh["trace"])) + 1)
            draw_grid_world(fr, sh, x0, 300, cell, upto, a)
            fr.text((x0, 250), lab, "small", INK, a)
            done = upto >= len(sh["trace"])
            if done:
                out = sh["outcome"]
                txt = {"goal": f"reached goal in {sh['steps']} steps (shortest {sh['optimal']})",
                       "timeout": f"stuck in a loop; gave up after {sh['steps']} steps"}.get(out, out)
                fr.text((x0, 820), txt, "small", GREEN if out == "goal" else RED, a)
                key = ("_navcue", tag)
                if key not in D:
                    D[key] = 1
                    CUES.append((fr.t, "pass" if out == "goal" else "fail", 0.25 if out == "goal" else 0.4))
        # success rates on unseen test sets
        x = 1290
        fr.text((x, 250), "success over 60 unseen worlds each", "small", DIM, a)
        rows = [("new rooms (familiar kind)", "rooms"), ("bigger rooms 13×13", "bigrooms"), ("corridor mazes", "mazes"),
                ("deadly hazard tiles", "hazards"), ("mirrored controls", "mirrored")]
        for i, (lab, k) in enumerate(rows):
            y = 300 + i * 96
            ai = ease((t - 0.4 - 0.2 * i) / 0.4)
            v = N[f"after_{k}"]["success_rate"]
            rnd = N.get(f"random_{k}", {}).get("success_rate")
            fr.text((x, y), lab, "small", INK, ai)
            fr.rect([x, y + 38, x + int(480 * v), y + 58], CYAN if v > 0.6 else (AMBER if v > 0.3 else RED), ai)
            fr.text((x + 490, y + 32), f"{v:.0%}", "small", INK, ai)
            if rnd is not None:
                fr.line([(x + int(480 * rnd), y + 32), (x + int(480 * rnd), y + 64)], INK, ai, 2)
        fr.text((x, 790), "white tick = random-action baseline", "tiny", DIM, a)
        fr.text((x, 820), f"before any learning: {N['before_rooms']['success_rate']:.0%} on rooms", "tiny", DIM, a)
        return "results/unfamiliar_showcases.json · logs/unfamiliar_events.jsonl"
    # keep learning in the new world: recovery and forgetting
    t2 = t - 6.2
    a = ease(t2 / 0.5)
    fr.text((120, 245), "It kept learning inside each unfamiliar world, then was re-tested on its old one", "body", INK, a)
    gx0, gy0, gw, gh = 140, 320, 760, 370
    fr.line([(gx0, gy0 + gh), (gx0 + gw, gy0 + gh)], FAINT, a)
    fr.line([(gx0, gy0), (gx0, gy0 + gh)], FAINT, a)
    for yv in (0, 0.5, 1):
        fr.text((gx0 - 14, gy0 + gh - gh * yv), f"{yv:.0%}", "tiny", DIM, a, anchor="rm")
    fr.text((gx0 + gw / 2, gy0 + gh + 22), "episodes of continued learning (0 to 600)", "tiny", DIM, a, anchor="ma")
    series = [("mirrored", CYAN, "mirrored controls"), ("hazards", AMBER, "hazard tiles"), ("mazes", VIOLET, "corridor mazes")]
    prog = ease((t2 - 0.3) / 1.6)
    for i, (k, col, lab) in enumerate(series):
        ys = [N[f"after_{k}"]["success_rate"]] + [p["success_rate"] for p in N[f"adapt_{k}"]]
        m = 1 + int(prog * (len(ys) - 1))
        pts = [(gx0 + gw * j / (len(ys) - 1), gy0 + gh - gh * v) for j, v in enumerate(ys[:m])]
        if len(pts) > 1:
            fr.line(pts, col, a, 4)
        fr.text((gx0 + gw + 20, gy0 + 10 + 40 * i), f"{lab}: {ys[0]:.0%} → {ys[-1]:.0%}", "small", col, a)
    x = 1060
    fr.text((x, 520), "Old skill afterwards (familiar rooms, was "
            f"{N['after_rooms']['success_rate']:.0%}):", "small", INK, ease((t2 - 1.3) / 0.5))
    for i, (k, col, lab) in enumerate(series):
        v = N[f"adapt_{k}_then_rooms"]["success_rate"]
        fr.text((x, 560 + 38 * i), f"after adapting to {lab}: {v:.0%}", "small", RED if v < 0.6 else GREEN, ease((t2 - 1.5 - 0.2 * i) / 0.5))
    C = D["code"]
    nu, eu = C["naive_unseen"], C["exp_unseen"]
    gained = [e["title"] for n_, e in zip(nu, eu) if e["solved"] and not n_["solved"]]
    lost = [e["title"] for n_, e in zip(nu, eu) if n_["solved"] and not e["solved"]]
    ap = ease((t2 - 2.3) / 0.5)
    fr.text((120, 790), f"Unseen programming problems: {sum(r['solved'] for r in nu)}/26 without experience → {sum(r['solved'] for r in eu)}/26 with it  ·  "
            f"newly solved: {', '.join(gained) or 'none'}", "tiny", INK, ap)
    fr.text((120, 820), f"lost to habits learned in practice: {', '.join(lost) or 'none'}", "tiny", INK, ap)
    c960 = D["chess"]["evals"]
    c0, c1 = c960[0].get("chess960", {}).get("random-chess960"), c960[-1].get("chess960", {}).get("random-chess960")
    if c0 and c1:
        fr.text((120, 850), f"Chess from 20 unseen Chess960 start positions vs random mover: score {c0['score']:.0%} → {c1['score']:.0%}  ·  "
                f"final material {c0['mean_final_material']:+.1f} → {c1['mean_final_material']:+.1f}", "tiny", INK, ap)
    fr.text((120, 892), "It adapted, but learning the new world overwrote part of the old skill.", "small", AMBER, ease((t2 - 2.9) / 0.5))
    return "results/unfamiliar_summary.json · results/code_summary.json"


def s_closing(fr, t, D):
    a = ease(t / 0.8)
    S = D["summary_lines"]
    fr.text((W // 2, 170), "What the experiment shows so far", "h1", INK, a, anchor="mm")
    heads = [("TESTED", CYAN), ("CHANGED WITH EXPERIENCE", GREEN), ("STILL UNSOLVED", AMBER)]
    for i, ((h, col), body) in enumerate(zip(heads, S)):
        ai = ease((t - 0.6 - 0.6 * i) / 0.7)
        y = 270 + i * 170
        fr.text((260, y), h, "label", col, ai)
        for j, line in enumerate(body):
            fr.text((260, y + 34 + 36 * j), line, "small", INK, ai)
    ai = ease((t - 3.6) / 0.8)
    fr.text((W // 2, 830), "A Digital Brain · independent home research, 2026 · all figures from recorded runs", "small", DIM, ai, anchor="mm")
    fr.text((W // 2, 900), "Amaresh  ·  Student and Open Researcher", "h2", INK, ai, anchor="mm")
    return "results/*.json · logs/*.jsonl"


FUNCS = [s_opening, s_activity, s_chess, s_code, s_general, s_closing]


def build_summary(D):
    ev = D["chess"]["evals"]
    e0, e1 = ev[0], ev[-1]
    C, N = D["code"], D["nav"]
    nu, eu = C["naive_unseen"], C["exp_unseen"]
    return [
        [f"{D['chess']['n_train_games']:,} practice chess games; 61 programming problems (26 never seen); 3,000 practice rooms, then 5 kinds of unseen worlds.",
         "Every result is compared with the same brain before learning, or with simple baselines."],
        [f"Chess: free-queen puzzles {e0['puzzles']['win_free_queen']['rate']:.0%} → {e1['puzzles']['win_free_queen']['rate']:.0%} (guessing ≈ {e1['puzzles']['win_free_queen']['chance_rate']:.0%}); "
         f"material vs random {e0['random']['mean_final_material']:+.1f} → {e1['random']['mean_final_material']:+.1f}.  "
         f"Programs: median search {int(np.median([r['checked'] for r in C['naive_train']]))} → {int(np.median([r['checked'] for r in C['exp_train']]))} candidates.",
         f"Navigation: {N['before_rooms']['success_rate']:.0%} → {N['after_rooms']['success_rate']:.0%} in new rooms; mirrored controls "
         f"{N['after_mirrored']['success_rate']:.0%} → {N['adapt_mirrored'][-1]['success_rate']:.0%} by continued learning."],
        [f"Mate in two: {e1['puzzles']['mate_in_2']['rate']:.0%}. Corridor mazes: {N['after_mazes']['success_rate']:.0%}. "
         f"Unseen programs: {sum(r['solved'] for r in eu)}/26 vs {sum(r['solved'] for r in nu)}/26 without experience.",
         "Habits learned in practice sometimes hurt on new problems, and adapting erased old skills."],
    ]


def main(preview_times=None):
    D = load()
    games = D["games_final"]
    g = pick_showcase_game(games)
    D["show_game"] = g
    D["show_game_no"] = int(g["file"].rsplit("_", 1)[1].split(".")[0]) + 1
    tag = D["chess"]["evals"][-1]["tag"]
    D["show_game"]["gid"] = f"{tag}-{g['opponent']}-{D['show_game_no'] - 1}"
    D["summary_lines"] = build_summary(D)
    out_dir = os.path.join(ROOT, "film_output")
    os.makedirs(out_dir, exist_ok=True)

    def render(tsec):
        fr = Frame(tsec)
        background(fr)
        for i, (s0, s1, name) in enumerate(SECTIONS):
            if s0 <= tsec < s1 or (i == len(SECTIONS) - 1 and tsec >= s1 - 1e-9):
                lt = tsec - s0
                fr.A = min(ease(lt / 0.45) if i else 1.0, ease((s1 - tsec) / 0.4) if i < len(SECTIONS) - 1 else 1.0)
                src = FUNCS[i](fr, lt, D)
                fr.A = 1.0
                chrome(fr, f"{i+1}/6 · {name}", src)
                return fr.img, name, src

    if preview_times:
        for ts in preview_times:
            img, _, _ = render(ts)
            img.save(os.path.join(out_dir, f"preview_{ts:05.2f}.png"))
        return
    video = os.path.join(out_dir, "video_only.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
                           "-pix_fmt", "yuv420p", video], stdin=subprocess.PIPE)
    n = int(DUR * FPS)
    last = None
    for k in range(n):
        tsec = k / FPS
        img, name, src = render(tsec)
        ff.stdin.write(img.tobytes())
        if (name, src) != last:
            MANIFEST.append(dict(film_time=round(tsec, 3), section=name, source=src))
            last = (name, src)
        if k == n - 1:
            img.save(os.path.join(out_dir, "final_frame.png"))
        if k % 150 == 0:
            print(f"frame {k}/{n}", flush=True)
    ff.stdin.close(); ff.wait()
    for s0, _, _ in SECTIONS[1:]:
        CUES.append((s0 - 0.7, "whoosh", 0.18))
    wav = audio.build_track(CUES, DUR, os.path.join(out_dir, "soundtrack.wav"))
    final = os.path.join(out_dir, "digital_brain_research_film.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart", final], check=True)
    with open(os.path.join(out_dir, "film_manifest.json"), "w") as f:
        json.dump(dict(sections=SECTIONS, segments=MANIFEST, audio_cues=CUES,
                       note="Each segment lists the recorded data it was rendered from; seq/t refer to the event logs."), f, indent=1)
    print("done", final)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main([float(x) for x in sys.argv[1:]])
    else:
        main()
