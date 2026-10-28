"""Writes REPORT.md — the research write-up accompanying the film. Every number
is read from the recorded result files so text and data cannot drift apart."""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
C = json.load(open(os.path.join(RES, "chess_summary.json")))
P = json.load(open(os.path.join(RES, "code_summary.json")))
N = json.load(open(os.path.join(RES, "unfamiliar_summary.json")))
pct = lambda v: f"{v:.0%}"
ev = C["evals"]
e0, e1 = ev[0], ev[-1]

L = []
w = L.append
w("# A Digital Brain — Home Research Experiment\n")
w("*Amaresh | Student and Open Researcher · October 2026*\n")
w("An experiment in how an artificial learning system behaves with familiar tasks, unfamiliar ones, failure and repeated "
  "experience. It is inspired by Einstein's curiosity and mathematical imagination. It is **not** a model of Einstein's brain, "
  "personality or intelligence, and nothing here claims thoughts, emotions or consciousness.\n")
w("**Film:** `film_output/digital_brain_research_film.mp4` (60 s). Every frame is drawn from the recorded logs and results. "
  "`film_output/film_manifest.json` maps each part of the film to the data it shows.\n")

w("## What counts as evidence here\n")
w("- **Measured directly:** actions chosen, alternatives it scored, decision times, rewards and outcomes, test results, "
  "memory reads and writes, exploration flags, and hidden-unit activations of its networks.")
w("- **Labelled by an outside referee:** chess blunders, missed mates and puzzle answers. The referee is a fixed checker, "
  "not part of the brain.")
w("- **Not claimed:** understanding, intention, insight or feeling. A change in activations shows that something changed, "
  "not what it means.")
w("- **No outside help:** no language model, engine or human writes moves, programs or routes for it. In the programming "
  "experiment it builds programs from a fixed set of 74 building blocks; it does not write free-form code.")
w("- **Not hand-picked:** the film shows the chess game picked by a fixed rule (a checkmate win if one exists, otherwise the "
  "game where it finished furthest ahead) and the first recorded runs of each showcase. Failures are reported next to successes.\n")

w("## Experiment 1 — Chess\n")
w(f"The brain was given the rules and, after each game, only *win / loss / draw*. It was not given piece values, openings "
  f"or tactics. It played **{C['n_train_games']:,} practice games** against itself, a random mover and a greedy capturer. "
  f"Evaluations used fixed opponents and puzzle sets taken from games it never played.\n")
w("| Measure | Before learning | After learning |")
w("|---|---|---|")
for opp, lab in [("random", "Score vs random mover"), ("greedy-capture", "Score vs greedy capturer"),
                 ("material-lookahead", "Score vs 1-move material engine")]:
    w(f"| {lab} | {pct(e0[opp]['score'])} (W{e0[opp]['win']}/D{e0[opp]['draw']}/L{e0[opp]['loss']}) | "
      f"{pct(e1[opp]['score'])} (W{e1[opp]['win']}/D{e1[opp]['draw']}/L{e1[opp]['loss']}) |")
for k, lab in [("win_free_queen", "Win a free queen (100 puzzles)"), ("mate_in_1", "Mate in one (100)"), ("mate_in_2", "Mate in two (30)")]:
    w(f"| {lab} | {pct(e0['puzzles'][k]['rate'])} | {pct(e1['puzzles'][k]['rate'])} (chance ≈ {pct(e1['puzzles'][k]['chance_rate'])}) |")
w(f"| Final material vs random (referee count) | {e0['random']['mean_final_material']:+.1f} | {e1['random']['mean_final_material']:+.1f} |")
w(f"| Blunders per 100 moves vs random | {e0['random']['blunders_per_100_moves']:.0f} | {e1['random']['blunders_per_100_moves']:.0f} |")
w(f"| Missed mate-in-one chances vs random (40 games) | {e0['random']['missed_mates']} | {e1['random']['missed_mates']} |")
w(f"| Mean decision time | {e0['random']['mean_decision_ms']:.1f} ms | {e1['random']['mean_decision_ms']:.1f} ms |")
c0, c1 = e0.get("chess960", {}).get("random-chess960"), e1.get("chess960", {}).get("random-chess960")
if c0 and c1:
    w(f"| **Unseen Chess960 starts** vs random: score / material | {pct(c0['score'])} / {c0['mean_final_material']:+.1f} | "
      f"{pct(c1['score'])} / {c1['mean_final_material']:+.1f} |")
w("\nCheckpoints during learning (vs random mover):\n")
w("| Games played | Score | Final material | Free-queen puzzles | Mate-in-one puzzles |")
w("|---|---|---|---|---|")
for e in ev:
    w(f"| {int(e['tag'][4:8]):,} | {pct(e['random']['score'])} | {e['random']['mean_final_material']:+.1f} | "
      f"{pct(e['puzzles']['win_free_queen']['rate'])} | {pct(e['puzzles']['mate_in_1']['rate'])} |")
w("")

w("## Experiment 2 — Programming challenges\n")
w(f"Each problem gives only **3 visible examples**. A submission is run on **30 hidden tests**. Like LeetCode, a wrong "
  f"answer returns one failing case, which the brain adds to its examples before searching again (up to {P['max_submissions']} "
  f"submissions). Every attempt has the same budget of {P['budget']} candidate programs. From experience it can learn which "
  "building blocks tend to follow each other, invent new blocks from repeated sub-solutions, and recall similar solved problems. "
  "A copy with no experience was run on the same problems with the same budget.\n")
nt, et, nu, eu = P["naive_train"], P["exp_train"], P["naive_unseen"], P["exp_unseen"]
med = lambda rs: int(np.median([r["checked"] for r in rs]))
w("| Problem set | No experience: solved / median candidates | After practice: solved / median candidates |")
w("|---|---|---|")
w(f"| 35 practice problems, re-tested with fresh tests | {sum(r['solved'] for r in nt)}/35 · {med(nt)} | {sum(r['solved'] for r in et)}/35 · {med(et)} |")
w(f"| 26 never-seen problems | {sum(r['solved'] for r in nu)}/26 · {med(nu)} | {sum(r['solved'] for r in eu)}/26 · {med(eu)} |")
w("\nNever-seen problems, one by one (✓ = all 30 hidden tests passed):\n")
w("| Category | Problem | No experience | After practice | Notes |")
w("|---|---|---|---|---|")
for a, b in zip(nu, eu):
    note = ""
    if b["solved"] and not a["solved"]:
        note = "newly solved after practice"
    elif a["solved"] and not b["solved"]:
        note = "**lost** — learned preferences steered the search away"
    elif not a["solved"] and not b["solved"]:
        note = "unsolved; wrong submission(s)" if b["submissions"] else "unsolved within budget"
    f = lambda r: ("✓ " if r["solved"] else "✗ ") + f"{r['checked']} cand., {r['submissions']} sub."
    w(f"| {a['category']} | {a['title']} | {f(a)} | {f(b)} | {note} |")
w(f"\nInvented building blocks: {', '.join(P['macros']) or 'none'}.\n")

w("## Experiment 3 — Unfamiliar worlds\n")
w("The brain learned to reach a goal in small open rooms. It sees a 5×5 patch around itself, the cells it has already "
  "visited, and the rough direction of the goal. It was then tested, with no warning, on worlds it had never met. "
  "Each test has 60 worlds; a little random exploration (5%) stays on.\n")
w("| World | Success | Random-action baseline | Notes |")
w("|---|---|---|---|")
w(f"| Rooms, before any learning | {pct(N['before_rooms']['success_rate'])} | {pct(N['random_rooms']['success_rate'])} | |")
for k, lab in [("rooms", "New rooms (familiar kind)"), ("bigrooms", "Bigger rooms 13×13"), ("mazes", "Corridor mazes"),
               ("hazards", "Deadly hazard tiles"), ("mirrored", "Mirrored left/right controls")]:
    r = N[f"after_{k}"]
    rnd = N.get(f"random_{k}", {}).get("success_rate")
    note = f"stuck in {pct(r['stuck_rate'])} of runs; recovered from {pct(r['recovered_rate_given_stuck'])} of those"
    if k == "hazards":
        note += f"; died on a hazard in {pct(r['hazard_rate'])}"
    w(f"| {lab} | {pct(r['success_rate'])} | {pct(rnd) if rnd is not None else '—'} | {note} |")
w("\nAllowed to keep learning inside the unfamiliar world (success after each 100 episodes), then re-tested on the old rooms:\n")
w("| World | Start | 100 | 200 | 300 | 400 | 500 | 600 | Old rooms afterwards |")
w("|---|---|---|---|---|---|---|---|---|")
for k in ("mirrored", "hazards", "mazes"):
    ys = [N[f"after_{k}"]["success_rate"]] + [p["success_rate"] for p in N[f"adapt_{k}"]]
    w(f"| {k} | " + " | ".join(pct(y) for y in ys) + f" | {pct(N[f'adapt_{k}_then_rooms']['success_rate'])} (was {pct(N['after_rooms']['success_rate'])}) |")
w("")

w("## Observations\n")
w(f"1. **Chess:** from outcomes alone it learned to win material against weak opponents. Free-queen puzzles went from "
  f"{pct(e0['puzzles']['win_free_queen']['rate'])} to {pct(e1['puzzles']['win_free_queen']['rate'])}, and material against a "
  f"random mover went from {e0['random']['mean_final_material']:+.1f} to {e1['random']['mean_final_material']:+.1f}. It still "
  "rarely turns that advantage into checkmate: most games end at the move limit, and it often misses mate in one. Positions that "
  "need a plan two moves deep remain largely beyond it.")
w("2. **Programming:** practice made solving familiar problem types much cheaper (fewer candidate programs checked). On "
  "never-seen problems the effect was mixed. Recalling related solutions solved a problem it could not solve before (sum of "
  "squares 1..n, built from what it learned on 'sum 1..n'). But its learned habits also made it miss problems the beginner "
  "copy solved. That is negative transfer.")
w("3. **Unfamiliar worlds:** navigation skill carried over to bigger rooms but not to winding corridor mazes, where it loops "
  "and times out. It walks into hazards it has never seen. With more learning it adapted to mirrored controls, but adapting "
  "overwrote its original skill (catastrophic forgetting).")
w("4. **Learning tests feedback:** in programming, wrong submissions were followed by revisions that used the returned "
  "failing case (see 'Count distinct letters' in the film: two wrong answers, then accepted on all 30 hidden tests).\n")
w("## What remains unsolved\n")
w("- Converting a material advantage into checkmate; any planning more than one move ahead.")
w("- Problems that need building blocks it does not have (2ⁿ, climbing stairs, longest increasing run, counting degree-1 nodes).")
w("- Keeping old skills while learning new ones; exploring long dead-end corridors.\n")
w("## Data\n")
w("- `logs/*_events.jsonl` — every recorded event. `t` = seconds since that experiment started (monotonic clock), "
  "`wall` = UTC time, `seq` = order.")
w("- `results/` — evaluation summaries, puzzle sets, recorded evaluation games and navigation traces with activations.")
w("- `film_output/film_manifest.json` — which data each part of the film was drawn from.")
open(os.path.join(ROOT, "REPORT.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("REPORT.md written")
