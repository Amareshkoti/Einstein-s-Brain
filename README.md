# Einstein's Brain — a home research experiment

**Amaresh | Student and Open Researcher**

A small artificial learning system, started from zero and studied the way you would study a living learner. It was given three kinds of challenge: chess, programming problems and unfamiliar worlds. I recorded what it did, what changed with experience, where it failed and what it never managed to learn.

The project is inspired by Einstein's curiosity and his habit of learning through experiments. The system is not a copy of Einstein's intelligence. It starts knowing nothing, and the experiment measures what it actually learns.

▶ **Watch the 60-second research film:** [film/digital_brain_research_film.mp4](film/digital_brain_research_film.mp4)
📄 **Full results:** [REPORT.md](REPORT.md)

![Opening of the film](figures/film_005_0s.png)

---

## The research question

Not "can it look intelligent?" but:

- What does it learn when nobody gives it the solution?
- Does its behaviour measurably change with experience?
- Do skills carry over to challenges it has never seen?
- What happens when it fails, gets stuck, or the rules change?

Every result is compared with the same brain before learning, or with simple baselines (random play, a beginner copy with no experience).

## How it was trained

**Chess.** The brain was given the rules and, after each game, only *win, loss or draw*. No piece values, no openings, no tactics. It played 8,000 practice games against itself, a random player and a greedy capturing player. It kept a memory of positions it had experienced and slowly learned to predict, from each position, how likely it was to win. On its turn it scores every legal move by the position that move leads to and plays the best one. Exploration (trying random moves) starts at 30% and fades to 5%.

**Programming challenges.** Like LeetCode without the problem statement: for each problem it sees only 3 input → output examples. A submission is run against 30 hidden tests, and a wrong answer returns one failing case, which it adds to its examples before trying again. It builds programs by chaining 74 basic building blocks, with a fixed budget of 250 candidate programs per attempt. Practice taught it which blocks tend to work together. It invented a new block from a pattern it kept reusing, and it remembers solved problems so it can recall similar ones later. It practised on 35 problems (arrays, strings, recursion, dynamic programming, graphs, trees) and was then tested on 26 problems it had never seen.

**Unfamiliar worlds.** The brain learned to reach a goal in small rooms, rewarded for arriving and lightly penalised for wasted steps and bumping into walls. It sees a small window around itself plus a short-term memory of where it has been. After 3,000 practice rooms it was dropped, without warning, into worlds it had never seen: bigger rooms, winding corridor mazes, deadly hazard tiles and mirrored controls. It was then allowed to keep learning there, and finally re-tested on its old rooms.

**Recording.** Every decision, every alternative it scored, every reward, memory update, test result and hidden-unit activation was logged with synchronised timestamps. The film is drawn entirely from those recordings. A separate referee (not part of the brain) labels chess blunders and puzzle answers.

## What it did

| | Before learning | After learning |
|---|---|---|
| Chess: win a free queen (100 unseen puzzles) | 22% | **44%** (random guess ≈ 25%) |
| Chess: material balance vs a random player | +3.0 | **+16.7** |
| Chess: score vs a random player | 46% | **61%** |
| Programming: median attempts on practice problems | 13 | **4** |
| Navigation: success in new rooms | 3% | **97%** |
| Navigation: mirrored controls, after continued learning | 2% | **72%** |

![Chess decision](figures/film_022_5s.png)
![Programming attempt log](figures/film_036_5s.png)

## What it did *not* learn

- **Chess:** it wins material but rarely converts it to checkmate. It misses mate in one, it has never beaten a player that looks one capture ahead, and mate-in-two puzzles stay near chance (7%).
- **Unseen programming problems:** 14 of 26 after practice versus 16 of 26 for the beginner copy. Recalling a related problem let it solve a new one ("sum of squares 1..n"). But habits learned in practice also made it miss three problems the beginner solved (negative transfer).
- **Unfamiliar worlds:** only 18% of corridor mazes, where it loops and times out. It walks straight into hazards it has never seen. Adapting to a new world erased its old skill: rooms dropped from 97% to 48% after learning mirrored controls, and to 3% after learning mazes.

![Learning and forgetting](figures/film_051_5s.png)

## Evidence in this repository

- [film/](film/): the 60-second research film.
- [figures/](figures/): stills from the film.
- [evidence/](evidence/): headline metrics and short excerpts of the timestamped event logs (a chess evaluation game, two programming attempts, navigation tests).
- [REPORT.md](REPORT.md): complete result tables, including every unseen programming problem and every failure.

The experiment code, trained networks and full logs are not published.

## Rights

© 2026 Amaresh. All rights reserved. See [LICENSE](LICENSE). You may view and share links to this repository. Copying, reproducing or reusing its content, results or methods without written permission is not allowed.
