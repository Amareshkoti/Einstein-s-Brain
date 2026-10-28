"""Experiment 2: Programming challenges.

The brain never sees a problem statement or a reference solution. Like a
LeetCode submitter it only gets a few visible input/output examples, can
submit a program, and is told pass / fail plus one failing hidden case.

It writes programs by composing building blocks from a fixed vocabulary
(listed in VOCAB). What it can learn:
  * which blocks tend to follow which (a learned preference over blocks),
  * reusable multi-block strategies (abstractions it invents by noticing a
    sub-pipeline that solved several problems), and
  * a memory of solved problems it retrieves for similar-looking new ones.
A "naive" copy with none of that experience is run on exactly the same
problems with exactly the same search budget, so changes can be measured."""
import heapq, json, math, os, random, time
from collections import Counter, defaultdict, deque
import numpy as np
from core import Recorder, ROOT

RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
BUDGET = 250           # candidate programs that may be checked per attempt
MAX_DEPTH = 6
MAX_SUBMISSIONS = 4


# ---------------- vocabulary of building blocks ----------------
def _scan(xs, f):
    out, acc = [], None
    for x in xs:
        acc = x if acc is None else f(acc, x)
        out.append(acc)
    return out


def _dedup_adj(xs):
    out = []
    for x in xs:
        if not out or out[-1] != x:
            out.append(x)
    return out


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
        if abs(p) > 10 ** 12:
            raise OverflowError
    return p


def _adj(g):
    n, edges = g
    a = [[] for _ in range(n)]
    for u, v in edges:
        a[u].append(v); a[v].append(u)
    return a


def _comps(g):
    a, seen, sizes = _adj(g), set(), []
    for s in range(g[0]):
        if s in seen:
            continue
        q, c = [s], 0
        seen.add(s)
        while q:
            u = q.pop(); c += 1
            for v in a[u]:
                if v not in seen:
                    seen.add(v); q.append(v)
        sizes.append(c)
    return sizes


def _dists0(g):
    a, d, q = _adj(g), {0: 0}, deque([0])
    while q:
        u = q.popleft()
        for v in a[u]:
            if v not in d:
                d[v] = d[u] + 1; q.append(v)
    return [d[k] for k in sorted(d)]


def _inorder(t):
    return [] if t is None else _inorder(t[1]) + [t[0]] + _inorder(t[2])


def _preorder(t):
    return [] if t is None else [t[0]] + _preorder(t[1]) + _preorder(t[2])


def _leaves(t):
    if t is None:
        return []
    if t[1] is None and t[2] is None:
        return [t[0]]
    return _leaves(t[1]) + _leaves(t[2])


def _height(t):
    return 0 if t is None else 1 + max(_height(t[1]), _height(t[2]))


def _levelsums(t):
    out, lvl = [], [t] if t else []
    while lvl:
        out.append(sum(n[0] for n in lvl))
        lvl = [c for n in lvl for c in (n[1], n[2]) if c]
    return out


def _range(n):
    if n > 60 or n < 0:
        raise ValueError
    return list(range(n))


V = "aeiou"
# name: (input type, output type, function, human-readable label)
VOCAB = {
    "sort": ("L", "L", sorted, "sort"), "rev": ("L", "L", lambda x: x[::-1], "reverse"),
    "uniq": ("L", "L", lambda x: list(dict.fromkeys(x)), "remove duplicates"),
    "inc": ("L", "L", lambda x: [a + 1 for a in x], "add 1 to each"), "dbl": ("L", "L", lambda x: [2 * a for a in x], "double each"),
    "sq": ("L", "L", lambda x: [a * a for a in x], "square each"), "neg": ("L", "L", lambda x: [-a for a in x], "negate each"),
    "absl": ("L", "L", lambda x: [abs(a) for a in x], "absolute value of each"),
    "evens": ("L", "L", lambda x: [a for a in x if a % 2 == 0], "keep even"), "odds": ("L", "L", lambda x: [a for a in x if a % 2], "keep odd"),
    "pos": ("L", "L", lambda x: [a for a in x if a > 0], "keep positive"),
    "cumsum": ("L", "L", lambda x: _scan(x, lambda a, b: a + b), "running sum"),
    "cummax": ("L", "L", lambda x: _scan(x, max), "running max"), "cummin": ("L", "L", lambda x: _scan(x, min), "running min"),
    "bestend": ("L", "L", lambda x: _scan(x, lambda acc, b: max(b, acc + b)), "best sum ending here"),
    "diffs": ("L", "L", lambda x: [b - a for a, b in zip(x, x[1:])], "adjacent differences"),
    "tail": ("L", "L", lambda x: x[1:], "drop first"), "init": ("L", "L", lambda x: x[:-1], "drop last"),
    "dedupadj": ("L", "L", _dedup_adj, "merge equal neighbours"),
    "sum": ("L", "I", sum, "sum"), "max": ("L", "I", max, "maximum"), "min": ("L", "I", min, "minimum"),
    "len": ("L", "I", len, "count"), "head": ("L", "I", lambda x: x[0], "first"), "last": ("L", "I", lambda x: x[-1], "last"),
    "prod": ("L", "I", _prod, "product"),
    "inc1": ("I", "I", lambda n: n + 1, "+1"), "dec1": ("I", "I", lambda n: n - 1, "-1"), "dbl1": ("I", "I", lambda n: 2 * n, "x2"),
    "sq1": ("I", "I", lambda n: n * n, "square"), "half": ("I", "I", lambda n: n // 2, "halve"), "neg1": ("I", "I", lambda n: -n, "negate"),
    "range": ("I", "L", _range, "0..n-1"),
    "srev": ("S", "S", lambda s: s[::-1], "reverse text"), "upper": ("S", "S", str.upper, "uppercase"),
    "ssort": ("S", "S", lambda s: "".join(sorted(s)), "sort letters"),
    "vowels": ("S", "S", lambda s: "".join(c for c in s if c in V), "keep vowels"),
    "nospace": ("S", "S", lambda s: s.replace(" ", ""), "remove spaces"),
    "sdedup": ("S", "S", lambda s: "".join(_dedup_adj(list(s))), "merge repeated letters"),
    "wordlens": ("S", "L", lambda s: [len(w) for w in s.split()], "word lengths"),
    "slen": ("S", "I", len, "text length"), "nwords": ("S", "I", lambda s: len(s.split()), "word count"),
    "degrees": ("G", "L", lambda g: [len(a) for a in _adj(g)], "node degrees"),
    "compsizes": ("G", "L", _comps, "component sizes"), "dists0": ("G", "L", _dists0, "distances from node 0"),
    "nnodes": ("G", "I", lambda g: g[0], "node count"), "nedges": ("G", "I", lambda g: len(g[1]), "edge count"),
    "inorder": ("T", "L", _inorder, "in-order values"), "preorder": ("T", "L", _preorder, "pre-order values"),
    "leaves": ("T", "L", _leaves, "leaf values"), "levelsums": ("T", "L", _levelsums, "sum per level"),
    "height": ("T", "I", _height, "height"), "tsize": ("T", "I", lambda t: len(_inorder(t)), "node count"),
    "not": ("B", "B", lambda b: not b, "not"),
}
# "unchanged by f?" tests, one per same-type block
for _k, (_i, _o, _f, _lab) in list(VOCAB.items()):
    if _i == _o and _i in "LS" and _k not in ("inc", "dbl", "sq", "neg", "upper"):
        VOCAB["fixed_" + _k] = (_i, "B", (lambda f: lambda x: f(x) == x)(_f), f"unchanged by '{_lab}'?")
BASE_VOCAB = dict(VOCAB)


def run_prog(prog, x, vocab):
    for op in prog:
        x = vocab[op][2](x)
    return x


# ---------------- problems (reference solutions never shown to the brain) ----------------
R = random.Random(2024)


def rl(r, lo=-9, hi=9, n=(1, 8)):
    return [r.randint(lo, hi) for _ in range(r.randint(*n))]


def rnn(r, lo=0, hi=9, n=(1, 8)):
    return rl(r, lo, hi, n)


def rword(r):
    words = ["apple", "tree", "sky", "moon", "river", "stone", "light", "atom", "wave", "field", "orbit", "eon", "aurora", "quiet"]
    return " ".join(r.choice(words) for _ in range(r.randint(1, 4)))


def rletters(r):
    return "".join(r.choice("abcdeioustrlmn") for _ in range(r.randint(1, 9)))


def rpal(r):
    if r.random() < 0.5:
        h = rletters(r)[:4]
        return h + (r.choice("xyz") if r.random() < 0.5 else "") + h[::-1]
    return rletters(r)


def rgraph(r):
    n = r.randint(1, 8)
    edges = set()
    for _ in range(r.randint(0, n + 2)):
        u, v = r.randrange(n), r.randrange(n)
        if u != v:
            edges.add((min(u, v), max(u, v)))
    return (n, sorted(edges))


def rtree(r, depth=0, vals=None):
    if depth > 3 or (depth > 0 and r.random() < 0.3):
        return None
    return (r.randint(-5, 9), rtree(r, depth + 1), rtree(r, depth + 1))


def rbst(r):
    if r.random() < 0.5:
        vals = r.sample(range(1, 30), r.randint(1, 8))
        t = None
        def ins(t, v):
            if t is None:
                return (v, None, None)
            return (t[0], ins(t[1], v), t[2]) if v < t[0] else (t[0], t[1], ins(t[2], v))
        for v in vals:
            t = ins(t, v)
        return t
    return rtree(r) or (1, None, None)


def rsorted_or_not(r):
    x = rl(r, n=(2, 7))
    return sorted(x) if r.random() < 0.5 else x


def rmaybe_dups(r):
    x = r.sample(range(-9, 10), r.randint(2, 7))
    if r.random() < 0.5:
        x.append(r.choice(x)); r.shuffle(x)
    return x


def second_largest(x):
    return sorted(x)[-2]


def kadane(x):
    return max(_scan(x, lambda a, b: max(b, a + b)))


def min_sub(x):
    best = cur = x[0]
    for a in x[1:]:
        cur = min(a, cur + a); best = min(best, cur)
    return best


def longest_inc_run(x):
    best = cur = 1
    for a, b in zip(x, x[1:]):
        cur = cur + 1 if b > a else 1
        best = max(best, cur)
    return best


def stairs(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def tree_leaves_count(t):
    return len(_leaves(t))


ne = lambda r: rl(r, n=(1, 8))
ne2 = lambda r: rl(r, n=(2, 8))
small = lambda r: r.randint(0, 9)
# (id, title, category, in type, out type, input generator, reference solution)
TRAIN = [
    ("sum", "Sum of array", "arrays", "L", "I", ne, sum),
    ("max", "Maximum element", "arrays", "L", "I", ne, max),
    ("min", "Minimum element", "arrays", "L", "I", ne, min),
    ("length", "Length of array", "arrays", "L", "I", lambda r: rl(r, n=(0, 8)), len),
    ("sort", "Sort ascending", "arrays", "L", "L", ne, sorted),
    ("sortdesc", "Sort descending", "arrays", "L", "L", ne, lambda x: sorted(x, reverse=True)),
    ("reverse", "Reverse array", "arrays", "L", "L", ne, lambda x: x[::-1]),
    ("counteven", "Count even numbers", "arrays", "L", "I", ne, lambda x: sum(a % 2 == 0 for a in x)),
    ("sumeven", "Sum of even numbers", "arrays", "L", "I", ne, lambda x: sum(a for a in x if a % 2 == 0)),
    ("sumsq", "Sum of squares", "arrays", "L", "I", ne, lambda x: sum(a * a for a in x)),
    ("distinct", "Count distinct values", "arrays", "L", "I", ne, lambda x: len(set(x))),
    ("sortuniq", "Sorted distinct values", "arrays", "L", "L", ne, lambda x: sorted(set(x))),
    ("secondlargest", "Second largest (with repeats)", "arrays", "L", "I", ne2, second_largest),
    ("maxeven", "Largest even number + 1", "arrays", "L", "I", lambda r: rl(r, n=(1, 8)) + [r.choice([-8, 0, 4])], lambda x: max(a for a in x if a % 2 == 0) + 1),
    ("srev", "Reverse a string", "strings", "S", "S", rword, lambda s: s[::-1]),
    ("supper", "Uppercase a string", "strings", "S", "S", rword, str.upper),
    ("slen", "Length of string", "strings", "S", "I", rword, len),
    ("words", "Count words", "strings", "S", "I", rword, lambda s: len(s.split())),
    ("vowelcount", "Count vowels", "strings", "S", "I", rword, lambda s: sum(c in V for c in s)),
    ("longestword", "Length of longest word", "strings", "S", "I", rword, lambda s: max(len(w) for w in s.split())),
    ("palindrome", "Is palindrome", "strings", "S", "B", rpal, lambda s: s == s[::-1]),
    ("letters", "Letters without spaces, sorted", "strings", "S", "S", rword, lambda s: "".join(sorted(s.replace(" ", "")))),
    ("gauss", "Sum 1..n", "recursion", "I", "I", small, lambda n: n * (n + 1) // 2),
    ("factorial", "Factorial n!", "recursion", "I", "I", small, lambda n: math.factorial(n)),
    ("triangular", "Triangular numbers up to n", "recursion", "I", "L", small, lambda n: _scan(list(range(1, n + 1)), lambda a, b: a + b)),
    ("squares", "Squares below n", "recursion", "I", "L", small, lambda n: [i * i for i in range(n)]),
    ("prefix", "Prefix sums", "dynamic programming", "L", "L", ne, lambda x: _scan(x, lambda a, b: a + b)),
    ("runmax", "Running maximum", "dynamic programming", "L", "L", ne, lambda x: _scan(x, max)),
    ("kadane", "Maximum subarray sum", "dynamic programming", "L", "I", ne, kadane),
    ("edges", "Number of edges", "graphs", "G", "I", rgraph, lambda g: len(g[1])),
    ("maxdeg", "Maximum degree", "graphs", "G", "I", rgraph, lambda g: max(len(a) for a in _adj(g))),
    ("treesize", "Size of a binary tree", "trees", "T", "I", lambda r: rtree(r) or (1, None, None), lambda t: len(_inorder(t))),
    ("treeheight", "Height of a binary tree", "trees", "T", "I", lambda r: rtree(r) or (1, None, None), _height),
    ("inorder", "In-order traversal", "trees", "T", "L", lambda r: rtree(r) or (1, None, None), _inorder),
    ("treesum", "Sum of tree values", "trees", "T", "I", lambda r: rtree(r) or (1, None, None), lambda t: sum(_inorder(t))),
]
UNSEEN = [
    ("countpos", "Count positive numbers", "arrays", "L", "I", ne, lambda x: sum(a > 0 for a in x)),
    ("sumabs", "Sum of absolute values", "arrays", "L", "I", ne, lambda x: sum(abs(a) for a in x)),
    ("secondsmallest", "Second smallest (with repeats)", "arrays", "L", "I", ne2, lambda x: sorted(x)[1]),
    ("maxjump", "Largest step between neighbours", "arrays", "L", "I", ne2, lambda x: max(b - a for a, b in zip(x, x[1:]))),
    ("issorted", "Is the array sorted?", "arrays", "L", "B", rsorted_or_not, lambda x: x == sorted(x)),
    ("hasdups", "Contains a duplicate?", "arrays", "L", "B", rmaybe_dups, lambda x: len(set(x)) != len(x)),
    ("distincteven", "Count distinct even values", "arrays", "L", "I", ne, lambda x: len({a for a in x if a % 2 == 0})),
    ("thirdlargest", "Third largest (with repeats)", "arrays", "L", "I", lambda r: rl(r, n=(3, 8)), lambda x: sorted(x)[-3]),
    ("upperrev", "Uppercase and reverse", "strings", "S", "S", rword, lambda s: s.upper()[::-1]),
    ("shortestword", "Length of shortest word", "strings", "S", "I", rword, lambda s: min(len(w) for w in s.split())),
    ("distinctletters", "Count distinct letters", "strings", "S", "I", rletters, lambda s: len(set(s))),
    ("consonants", "Count consonants (letters only)", "strings", "S", "I", rword, lambda s: sum(c.isalpha() and c not in V for c in s)),
    ("sumsq1n", "Sum of squares 1..n", "recursion", "I", "I", small, lambda n: sum(i * i for i in range(1, n + 1))),
    ("pow2", "2 to the power n", "recursion", "I", "I", small, lambda n: 2 ** n),
    ("minsub", "Minimum subarray sum", "dynamic programming", "L", "I", ne, min_sub),
    ("maxprefix", "Maximum prefix sum", "dynamic programming", "L", "I", ne, lambda x: max(_scan(x, lambda a, b: a + b))),
    ("incrun", "Longest increasing run", "dynamic programming", "L", "I", ne, longest_inc_run),
    ("stairs", "Climbing stairs (ways)", "dynamic programming", "I", "I", small, stairs),
    ("components", "Number of connected components", "graphs", "G", "I", rgraph, lambda g: len(_comps(g))),
    ("biggestcomp", "Largest connected component", "graphs", "G", "I", rgraph, lambda g: max(_comps(g))),
    ("farthest", "Farthest node from node 0", "graphs", "G", "I", rgraph, lambda g: max(_dists0(g))),
    ("leafnodes", "Count nodes of degree 1", "graphs", "G", "I", rgraph, lambda g: sum(len(a) == 1 for a in _adj(g))),
    ("treemax", "Largest value in a tree", "trees", "T", "I", lambda r: rtree(r) or (1, None, None), lambda t: max(_inorder(t))),
    ("leafcount", "Count leaves", "trees", "T", "I", lambda r: rtree(r) or (1, None, None), tree_leaves_count),
    ("maxlevel", "Largest level sum", "trees", "T", "I", lambda r: rtree(r) or (1, None, None), lambda t: max(_levelsums(t))),
    ("isbst", "Is a binary search tree?", "trees", "T", "B", rbst, lambda t: _inorder(t) == sorted(set(_inorder(t)))),
]


def build(spec, seed, n_visible=3, n_hidden=30):
    pid, title, cat, ti, to, gen, ref = spec
    r = random.Random(f"{pid}-{seed}")
    def case():
        while True:
            x = gen(r)
            try:
                return (x, ref(x))
            except Exception:
                continue
    vis = [case() for _ in range(n_visible)]
    hid = [case() for _ in range(n_hidden)]
    return dict(id=pid, title=title, category=cat, tin=ti, tout=to, visible=vis, hidden=hid)


# ---------------- the learner ----------------
def signature(p):
    """Coarse features of the examples alone (what a solver can 'see')."""
    xs, ys = [e[0] for e in p["visible"]], [e[1] for e in p["visible"]]
    f = [p["tin"] == t for t in "LISGT"] + [p["tout"] == t for t in "LISBT"]
    if p["tout"] == "L" and p["tin"] == "L":
        f += [np.mean([len(y) == len(x) for x, y in zip(xs, ys)]), np.mean([sorted(y) == y for y in ys]),
              np.mean([set(y) <= set(x) for x, y in zip(xs, ys)])]
    else:
        f += [0, 0, 0]
    if p["tout"] == "I" and p["tin"] == "L":
        f += [np.mean([y in x for x, y in zip(xs, ys)]), np.mean([abs(y) > max(map(abs, x), default=0) for x, y in zip(xs, ys)])]
    else:
        f += [0, 0]
    return np.array(f, float)


class CodeBrain:
    def __init__(self, learning=True):
        self.vocab = dict(BASE_VOCAB)
        self.counts = defaultdict(Counter)    # learned preference: context -> next block
        self.solved = []                      # memory: (problem id, signature, program)
        self.macros = {}
        self.learning = learning

    # learned preference over next block
    def logp(self, ctx, op, ty, boost):
        valid = [k for k, v in self.vocab.items() if v[0] == ty]
        c = self.counts[ctx]
        a = 1.0
        tot = sum(c[k] + boost.get(k, 0) + a for k in valid)
        return math.log((c[op] + boost.get(op, 0) + a) / tot)

    def retrieve(self, p):
        if not self.solved:
            return [], {}
        s = signature(p)
        sims = []
        for pid, sig, prog in self.solved:
            sim = float(s @ sig / (np.linalg.norm(s) * np.linalg.norm(sig) + 1e-9))
            sims.append((sim, pid, prog))
        sims.sort(reverse=True)
        top = [t for t in sims[:3] if t[0] > 0.8]
        boost = Counter()
        for sim, pid, prog in top:
            for op in prog:
                boost[op] += 2.0 * sim
        return [(pid, round(sim, 3), list(prog)) for sim, pid, prog in top], boost

    def attempt(self, p, rec, phase, budget=BUDGET):
        t0 = time.perf_counter()
        memory_hits, boost = self.retrieve(p) if self.learning else ([], {})
        rec.log("attempt_start", phase=phase, problem=p["id"], category=p["category"], visible=len(p["visible"]),
                memory_retrieved=memory_hits)
        examples = list(p["visible"])
        rejected = set()
        submissions, failures, n_checked, n_runtime_err = [], [], 0, 0
        first_fail_logged = 0
        while len(submissions) < MAX_SUBMISSIONS:
            found = None
            heap = [(0.0, 0, (), p["tin"])]
            seen_outputs = set()
            tie = 0
            pops = 0
            xs = [e[0] for e in examples]
            while heap and n_checked < budget and pops < 40 * budget:
                cost, _, prog, ty = heapq.heappop(heap)
                pops += 1
                if prog:
                    try:
                        outs = [run_prog(prog, x, self.vocab) for x in xs]
                    except Exception:
                        n_runtime_err += 1
                        continue
                    key = (ty, repr(outs))
                    if key in seen_outputs:
                        continue            # behaves identically to something already tried
                    seen_outputs.add(key)
                    if ty == p["tout"] and prog not in rejected:
                        n_checked += 1
                        bad = [(x, y, o) for (x, y), o in zip(examples, outs) if o != y]
                        if not bad:
                            found = prog
                            break
                        if first_fail_logged < 6:
                            first_fail_logged += 1
                            x, y, o = bad[0]
                            rec.log("candidate_failed", phase=phase, problem=p["id"], program=list(prog), checked=n_checked,
                                    case_input=x, expected=y, got=o)
                ctx = prog[-1] if prog else "START:" + p["tin"]
                if len(self._expand(prog)) < MAX_DEPTH:
                    for op, (ti, to, _, _) in self.vocab.items():
                        if ti == ty:
                            tie += 1
                            heapq.heappush(heap, (cost - self.logp(ctx, op, ty, boost), tie, prog + (op,), to))
            if not found:
                break
            # submit against hidden tests
            hidden_fail = None
            passed = 0
            for x, y in p["hidden"]:
                try:
                    o = run_prog(found, x, self.vocab)
                except Exception as e:
                    o = f"error: {type(e).__name__}"
                if o == y:
                    passed += 1
                elif hidden_fail is None:
                    hidden_fail = (x, y, o)
            sub = dict(program=list(found), expanded=self._expand(found), hidden_passed=passed,
                       hidden_total=len(p["hidden"]), checked_so_far=n_checked,
                       t_attempt=round(time.perf_counter() - t0, 4))
            if hidden_fail:
                sub["failed_case"] = dict(input=hidden_fail[0], expected=hidden_fail[1], got=hidden_fail[2])
            submissions.append(sub)
            rec.log("submission", phase=phase, problem=p["id"], **sub)
            if not hidden_fail:
                break
            rejected.add(found)
            examples.append((hidden_fail[0], hidden_fail[1]))   # learn from the failing test
        ok = bool(submissions) and "failed_case" not in submissions[-1]
        dt = time.perf_counter() - t0
        out = dict(problem=p["id"], title=p["title"], category=p["category"], solved=ok, checked=n_checked,
                   submissions=len(submissions), seconds=round(dt, 4), runtime_errors=n_runtime_err,
                   program=submissions[-1]["program"] if submissions else None,
                   expanded=submissions[-1]["expanded"] if submissions else None,
                   hidden_pass_rate=(submissions[-1]["hidden_passed"] / submissions[-1]["hidden_total"]) if submissions else 0.0,
                   memory_retrieved=memory_hits, attempts=submissions)
        rec.log("attempt_end", phase=phase, **{k: v for k, v in out.items() if k != "attempts"})
        if ok and self.learning:
            self.learn_from(p, submissions[-1]["program"], rec, phase)
        return out

    def _expand(self, prog):
        out = []
        for op in prog:
            out += self.macros[op] if op in self.macros else [op]
        return out

    def learn_from(self, p, prog, rec, phase):
        ctx = "START:" + p["tin"]
        for op in prog:
            self.counts[ctx][op] += 2
            ctx = op
        if p["id"] not in [s[0] for s in self.solved]:
            self.solved.append((p["id"], signature(p), tuple(prog)))
        rec.log("memory_update", phase=phase, problem=p["id"], stored_program=list(prog), memory_size=len(self.solved))

    def invent_abstractions(self, rec, phase):
        """Sub-pipelines used by >=2 different solved problems become new single blocks."""
        sub = Counter()
        for pid, _, prog in self.solved:
            full = tuple(self._expand(prog))
            seen = set()
            for n in (2, 3):
                for i in range(len(full) - n + 1):
                    seen.add(full[i:i + n])
            sub.update(seen)
        made = []
        for s, c in sub.most_common():
            if c < 2:
                break
            name = "+".join(s)
            if name in self.vocab or any(s == tuple(m) for m in self.macros.values()):
                continue
            ti, to = BASE_VOCAB[s[0]][0], BASE_VOCAB[s[-1]][1]
            fs = [BASE_VOCAB[o][2] for o in s]
            def f(x, fs=fs):
                for g in fs:
                    x = g(x)
                return x
            self.vocab[name] = (ti, to, f, " → ".join(BASE_VOCAB[o][3] for o in s))
            self.macros[name] = list(s)
            made.append(dict(name=name, used_by=c))
            if len(made) >= 6:
                break
        rec.log("abstractions_invented", phase=phase, new=made, vocabulary_size=len(self.vocab))
        return made


def main():
    rec = Recorder("code")
    train = [build(s, 0) for s in TRAIN]
    unseen = [build(s, 0) for s in UNSEEN]
    train_retest = [build(s, 1) for s in TRAIN]   # same problems, fresh examples and hidden tests
    rec.log("problem_sets", train=[p["id"] for p in train], unseen=[p["id"] for p in unseen])
    results = {"budget": BUDGET, "max_submissions": MAX_SUBMISSIONS}

    naive = CodeBrain(learning=False)
    results["naive_unseen"] = [naive.attempt(p, rec, "naive-unseen") for p in unseen]
    results["naive_train"] = [naive.attempt(p, rec, "naive-train") for p in train_retest]
    print("naive unseen solved", sum(r["solved"] for r in results["naive_unseen"]), "/", len(unseen), flush=True)
    print("naive train solved", sum(r["solved"] for r in results["naive_train"]), "/", len(train), flush=True)

    brain = CodeBrain(learning=True)
    results["training_passes"] = []
    order = list(train)
    for ps in range(4):
        random.Random(ps).shuffle(order)
        rs = [brain.attempt(p, rec, f"train-pass{ps+1}") for p in order]
        made = brain.invent_abstractions(rec, f"train-pass{ps+1}")
        results["training_passes"].append(dict(pass_=ps + 1, results=rs, abstractions=made))
        print(f"pass {ps+1}: solved {sum(r['solved'] for r in rs)}/{len(rs)}  checked {sum(r['checked'] for r in rs)}  new blocks {[m['name'] for m in made]}", flush=True)

    brain.learning_frozen = True
    learned = brain.learning
    brain.learning = True   # memory retrieval stays on, but no updates from test problems:
    brain.learn_from = lambda *a, **k: None
    results["exp_unseen"] = [brain.attempt(p, rec, "experienced-unseen") for p in unseen]
    results["exp_train"] = [brain.attempt(p, rec, "experienced-train") for p in train_retest]
    print("experienced unseen solved", sum(r["solved"] for r in results["exp_unseen"]), "/", len(unseen), flush=True)
    print("experienced train solved", sum(r["solved"] for r in results["exp_train"]), "/", len(train), flush=True)
    results["macros"] = brain.macros
    results["top_preferences"] = {k: v.most_common(4) for k, v in brain.counts.items() if k.startswith("START") or sum(v.values()) >= 6}
    results["problems"] = {p["id"]: dict(title=p["title"], category=p["category"], tin=p["tin"], tout=p["tout"],
                                         visible=p["visible"], hidden_count=len(p["hidden"]))
                           for p in unseen + train}
    with open(os.path.join(RES, "code_summary.json"), "w") as f:
        json.dump(results, f, default=str)
    rec.close()


if __name__ == "__main__":
    main()
