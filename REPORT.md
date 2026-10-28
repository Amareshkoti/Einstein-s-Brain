# A Digital Brain — Home Research Experiment

*Amaresh | Student and Open Researcher · October 2026*

An experiment in how an artificial learning system behaves with familiar tasks, unfamiliar ones, failure and repeated experience. It is inspired by Einstein's curiosity and mathematical imagination. It is **not** a model of Einstein's brain, personality or intelligence, and nothing here claims thoughts, emotions or consciousness.

**Film:** [film/digital_brain_research_film.mp4](film/digital_brain_research_film.mp4) (60 s). Every frame is drawn from the recorded logs and results.

## What counts as evidence here

- **Measured directly:** actions chosen, alternatives it scored, decision times, rewards and outcomes, test results, memory reads and writes, exploration flags, and hidden-unit activations of its networks.
- **Labelled by an outside referee:** chess blunders, missed mates and puzzle answers. The referee is a fixed checker, not part of the brain.
- **Not claimed:** understanding, intention, insight or feeling. A change in activations shows that something changed, not what it means.
- **No outside help:** no language model, engine or human writes moves, programs or routes for it. In the programming experiment it builds programs from a fixed set of 74 building blocks; it does not write free-form code.
- **Not hand-picked:** the film shows the chess game picked by a fixed rule (a checkmate win if one exists, otherwise the game where it finished furthest ahead) and the first recorded runs of each showcase. Failures are reported next to successes.

## Experiment 1 — Chess

The brain was given the rules and, after each game, only *win / loss / draw*. It was not given piece values, openings or tactics. It played **8,000 practice games** against itself, a random mover and a greedy capturer. Evaluations used fixed opponents and puzzle sets taken from games it never played.

| Measure | Before learning | After learning |
|---|---|---|
| Score vs random mover | 46% (W1/D35/L4) | 61% (W9/D31/L0) |
| Score vs greedy capturer | 49% (W0/D39/L1) | 48% (W0/D38/L2) |
| Score vs 1-move material engine | 4% (W0/D3/L37) | 2% (W0/D2/L38) |
| Win a free queen (100 puzzles) | 22% | 44% (chance ≈ 25%) |
| Mate in one (100) | 4% | 2% (chance ≈ 4%) |
| Mate in two (30) | 3% | 7% (chance ≈ 4%) |
| Final material vs random (referee count) | +3.0 | +16.7 |
| Blunders per 100 moves vs random | 47 | 40 |
| Missed mate-in-one chances vs random (40 games) | 22 | 174 |
| Mean decision time | 1.1 ms | 1.2 ms |
| **Unseen Chess960 starts** vs random: score / material | 55% / +7.7 | 55% / +9.7 |

Checkpoints during learning (vs random mover):

| Games played | Score | Final material | Free-queen puzzles | Mate-in-one puzzles |
|---|---|---|---|---|
| 0 | 46% | +3.0 | 22% | 4% |
| 1,000 | 55% | +15.2 | 39% | 5% |
| 2,000 | 52% | +13.5 | 35% | 8% |
| 3,000 | 59% | +15.8 | 35% | 5% |
| 4,000 | 56% | +11.1 | 34% | 3% |
| 5,000 | 56% | +14.3 | 44% | 8% |
| 6,000 | 60% | +12.9 | 41% | 2% |
| 7,000 | 62% | +15.6 | 45% | 2% |
| 8,000 | 61% | +16.7 | 44% | 2% |

## Experiment 2 — Programming challenges

Each problem gives only **3 visible examples**. A submission is run on **30 hidden tests**. Like LeetCode, a wrong answer returns one failing case, which the brain adds to its examples before searching again (up to 4 submissions). Every attempt has the same budget of 250 candidate programs. From experience it can learn which building blocks tend to follow each other, invent new blocks from repeated sub-solutions, and recall similar solved problems. A copy with no experience was run on the same problems with the same budget.

| Problem set | No experience: solved / median candidates | After practice: solved / median candidates |
|---|---|---|
| 35 practice problems, re-tested with fresh tests | 32/35 · 13 | 34/35 · 4 |
| 26 never-seen problems | 16/26 · 73 | 14/26 · 208 |

Never-seen problems, one by one (✓ = all 30 hidden tests passed):

| Category | Problem | No experience | After practice | Notes |
|---|---|---|---|---|
| arrays | Count positive numbers | ✓ 77 cand., 1 sub. | ✗ 250 cand., 1 sub. | **lost** — learned preferences steered the search away |
| arrays | Sum of absolute values | ✓ 56 cand., 1 sub. | ✓ 232 cand., 1 sub. |  |
| arrays | Second smallest (with repeats) | ✗ 250 cand., 0 sub. | ✗ 250 cand., 0 sub. | unsolved within budget |
| arrays | Largest step between neighbours | ✓ 88 cand., 1 sub. | ✗ 250 cand., 0 sub. | **lost** — learned preferences steered the search away |
| arrays | Is the array sorted? | ✓ 1 cand., 1 sub. | ✓ 1 cand., 1 sub. |  |
| arrays | Contains a duplicate? | ✓ 9 cand., 2 sub. | ✓ 9 cand., 2 sub. |  |
| arrays | Count distinct even values | ✗ 250 cand., 1 sub. | ✗ 250 cand., 1 sub. | unsolved; wrong submission(s) |
| arrays | Third largest (with repeats) | ✗ 250 cand., 0 sub. | ✗ 250 cand., 0 sub. | unsolved within budget |
| strings | Uppercase and reverse | ✓ 8 cand., 1 sub. | ✓ 8 cand., 1 sub. |  |
| strings | Length of shortest word | ✓ 20 cand., 1 sub. | ✓ 53 cand., 1 sub. |  |
| strings | Count distinct letters | ✓ 94 cand., 3 sub. | ✓ 120 cand., 3 sub. |  |
| strings | Count consonants (letters only) | ✗ 250 cand., 1 sub. | ✗ 250 cand., 1 sub. | unsolved; wrong submission(s) |
| recursion | Sum of squares 1..n | ✗ 250 cand., 0 sub. | ✓ 34 cand., 1 sub. | newly solved after practice |
| recursion | 2 to the power n | ✗ 250 cand., 0 sub. | ✗ 250 cand., 0 sub. | unsolved within budget |
| dynamic programming | Minimum subarray sum | ✗ 250 cand., 0 sub. | ✗ 250 cand., 0 sub. | unsolved within budget |
| dynamic programming | Maximum prefix sum | ✓ 69 cand., 1 sub. | ✓ 29 cand., 1 sub. |  |
| dynamic programming | Longest increasing run | ✗ 250 cand., 0 sub. | ✗ 250 cand., 1 sub. | unsolved; wrong submission(s) |
| dynamic programming | Climbing stairs (ways) | ✗ 250 cand., 0 sub. | ✗ 250 cand., 0 sub. | unsolved within budget |
| graphs | Number of connected components | ✓ 19 cand., 1 sub. | ✓ 66 cand., 1 sub. |  |
| graphs | Largest connected component | ✓ 17 cand., 1 sub. | ✓ 52 cand., 1 sub. |  |
| graphs | Farthest node from node 0 | ✓ 22 cand., 1 sub. | ✓ 118 cand., 2 sub. |  |
| graphs | Count nodes of degree 1 | ✗ 250 cand., 1 sub. | ✗ 250 cand., 1 sub. | unsolved; wrong submission(s) |
| trees | Largest value in a tree | ✓ 16 cand., 1 sub. | ✓ 22 cand., 1 sub. |  |
| trees | Count leaves | ✓ 38 cand., 2 sub. | ✓ 184 cand., 2 sub. |  |
| trees | Largest level sum | ✓ 50 cand., 2 sub. | ✗ 250 cand., 1 sub. | **lost** — learned preferences steered the search away |
| trees | Is a binary search tree? | ✓ 1 cand., 1 sub. | ✓ 1 cand., 1 sub. |  |

Invented building blocks: range+inc.

## Experiment 3 — Unfamiliar worlds

The brain learned to reach a goal in small open rooms. It sees a 5×5 patch around itself, the cells it has already visited, and the rough direction of the goal. It was then tested, with no warning, on worlds it had never met. Each test has 60 worlds; a little random exploration (5%) stays on.

| World | Success | Random-action baseline | Notes |
|---|---|---|---|
| Rooms, before any learning | 3% | 8% | |
| New rooms (familiar kind) | 97% | 8% | stuck in 5% of runs; recovered from 67% of those |
| Bigger rooms 13×13 | 95% | 7% | stuck in 8% of runs; recovered from 60% of those |
| Corridor mazes | 18% | 0% | stuck in 77% of runs; recovered from 4% of those |
| Deadly hazard tiles | 35% | 2% | stuck in 2% of runs; recovered from 0% of those; died on a hazard in 65% |
| Mirrored left/right controls | 2% | — | stuck in 10% of runs; recovered from 0% of those |

Allowed to keep learning inside the unfamiliar world (success after each 100 episodes), then re-tested on the old rooms:

| World | Start | 100 | 200 | 300 | 400 | 500 | 600 | Old rooms afterwards |
|---|---|---|---|---|---|---|---|---|
| mirrored | 2% | 10% | 12% | 12% | 50% | 60% | 72% | 48% (was 97%) |
| hazards | 35% | 28% | 10% | 25% | 30% | 45% | 42% | 97% (was 97%) |
| mazes | 18% | 10% | 0% | 5% | 22% | 25% | 25% | 3% (was 97%) |

## Observations

1. **Chess:** from outcomes alone it learned to win material against weak opponents. Free-queen puzzles went from 22% to 44%, and material against a random mover went from +3.0 to +16.7. It still rarely turns that advantage into checkmate: most games end at the move limit, and it often misses mate in one. Positions that need a plan two moves deep remain largely beyond it.
2. **Programming:** practice made solving familiar problem types much cheaper (fewer candidate programs checked). On never-seen problems the effect was mixed. Recalling related solutions solved a problem it could not solve before (sum of squares 1..n, built from what it learned on 'sum 1..n'). But its learned habits also made it miss problems the beginner copy solved. That is negative transfer.
3. **Unfamiliar worlds:** navigation skill carried over to bigger rooms but not to winding corridor mazes, where it loops and times out. It walks into hazards it has never seen. With more learning it adapted to mirrored controls, but adapting overwrote its original skill (catastrophic forgetting).
4. **Learning tests feedback:** in programming, wrong submissions were followed by revisions that used the returned failing case (see 'Count distinct letters' in the film: two wrong answers, then accepted on all 30 hidden tests).

## What remains unsolved

- Converting a material advantage into checkmate; any planning more than one move ahead.
- Problems that need building blocks it does not have (2ⁿ, climbing stairs, longest increasing run, counting degree-1 nodes).
- Keeping old skills while learning new ones; exploring long dead-end corridors.

## Data

- [evidence/](evidence/) has the headline metrics and short excerpts of the timestamped event logs. In each event, `t` is seconds since that experiment started (monotonic clock), `wall` is UTC time and `seq` is the order.
- The full event logs (about 240 MB), the trained networks and the experiment code are kept private.
