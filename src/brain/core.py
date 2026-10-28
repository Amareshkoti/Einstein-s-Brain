"""Shared pieces of the digital brain: a small neural network whose activations
can be read directly, and an event recorder with synchronized timestamps."""
import json, os, time, datetime
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


class Recorder:
    """Append-only JSONL event log. Every event carries `t` (seconds since the
    experiment started, monotonic clock) and `wall` (UTC wall-clock time), so
    behaviour can be lined up with visualizations later."""

    def __init__(self, name):
        self.name = name
        self.path = os.path.join(LOG_DIR, f"{name}_events.jsonl")
        self.f = open(self.path, "w", encoding="utf-8")
        self.t0 = time.perf_counter()
        self.seq = 0
        self.log("experiment_start", experiment=name)

    def now(self):
        return time.perf_counter() - self.t0

    def log(self, kind, **data):
        self.seq += 1
        ev = {"seq": self.seq, "t": round(self.now(), 4),
              "wall": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
              "exp": self.name, "kind": kind}
        for k, v in data.items():
            ev[("data_" + k) if k in ("seq", "t", "wall", "exp", "kind") else k] = v
        self.f.write(json.dumps(ev, default=_jsonable) + "\n")
        if self.seq % 200 == 0:
            self.f.flush()
        return ev

    def close(self):
        self.log("experiment_end")
        self.f.close()


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return str(o)


class Net:
    """Fully connected network (tanh hidden layers, linear output) trained with
    Adam. `forward` returns the hidden activations so they can be recorded as
    directly measured internal activity."""

    def __init__(self, sizes, seed=0, lr=1e-3):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, 1 / np.sqrt(a), (a, b)).astype(np.float32) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.lr = lr
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.step = 0

    def forward(self, x):
        acts = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.tanh(z) if i < len(self.W) - 1 else z
            acts.append(h)
        return h, acts

    def __call__(self, x):
        return self.forward(x)[0]

    def train(self, x, grad_out_fn):
        """grad_out_fn(output) -> (loss, dLoss/dOutput)."""
        out, acts = self.forward(x)
        loss, g = grad_out_fn(out)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ g
            gb[i] = g.sum(0)
            if i > 0:
                g = (g @ self.W[i].T) * (1 - acts[i] ** 2)
        self._adam(gW + gb)
        return loss

    def _adam(self, grads, b1=0.9, b2=0.999, eps=1e-8):
        self.step += 1
        params = self.W + self.b
        for k, (p, gr) in enumerate(zip(params, grads)):
            gr = np.clip(gr, -5, 5)
            self.m[k] = b1 * self.m[k] + (1 - b1) * gr
            self.v[k] = b2 * self.v[k] + (1 - b2) * gr * gr
            mh = self.m[k] / (1 - b1 ** self.step)
            vh = self.v[k] / (1 - b2 ** self.step)
            p -= self.lr * mh / (np.sqrt(vh) + eps)

    def state(self):
        return {"W": [w.copy() for w in self.W], "b": [b.copy() for b in self.b]}

    def load(self, st):
        self.W = [w.copy() for w in st["W"]]
        self.b = [b.copy() for b in st["b"]]

    def save(self, path):
        np.savez(path, *self.W, *self.b)
