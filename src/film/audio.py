"""Original ambient score + subtle sound design for the research film.
Everything is synthesized here, so there are no licensing questions."""
import wave
import numpy as np

SR = 44100


def note_hz(n):
    """MIDI note number -> Hz."""
    return 440.0 * 2 ** ((n - 69) / 12)


def env_adsr(n, a, r, sr=SR):
    e = np.ones(n)
    na, nr = int(a * sr), int(r * sr)
    na, nr = min(na, n), min(nr, n)
    e[:na] = np.linspace(0, 1, na) ** 2
    if nr:
        e[-nr:] *= np.linspace(1, 0, nr) ** 2
    return e


def lowpass(x, cutoff):
    a = np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    # vectorised one-pole via lfilter-style recursion in chunks
    from itertools import accumulate
    y = np.fromiter(accumulate(x, lambda p, v: a * p + (1 - a) * v, initial=0.0), float)[1:]
    return y


def pad_voice(freq, dur, rng, detune=0.004):
    t = np.arange(int(dur * SR)) / SR
    out = np.zeros((len(t), 2))
    for ch in range(2):
        f = freq * (1 + rng.uniform(-detune, detune))
        ph = rng.uniform(0, 2 * np.pi)
        v = (np.sin(2 * np.pi * f * t + ph) + 0.35 * np.sin(4 * np.pi * f * t + ph * 1.3)
             + 0.12 * np.sin(6 * np.pi * f * t + ph * 0.7))
        v *= 1 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.08, 0.2) * t)
        out[:, ch] = v
    return out


def pluck(freq, dur=2.2, bright=1.0):
    t = np.arange(int(dur * SR)) / SR
    e = np.exp(-t * 3.2)
    att = np.minimum(1, t / 0.004)
    v = (np.sin(2 * np.pi * freq * t) + 0.4 * bright * np.sin(2 * np.pi * 2 * freq * t) * np.exp(-t * 6)
         + 0.15 * bright * np.sin(2 * np.pi * 3.01 * freq * t) * np.exp(-t * 9))
    return v * e * att


def reverb(x, seconds=2.6, mix=0.35, seed=5):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = np.zeros_like(x)
    for ch in range(x.shape[1]):
        ir = rng.normal(0, 1, n) * np.exp(-t * 3.0 / seconds * 2.3)
        ir[: int(0.012 * SR)] = 0
        ir /= np.sqrt((ir ** 2).sum())
        L = len(x) + n
        nfft = 1 << (L - 1).bit_length()
        wet = np.fft.irfft(np.fft.rfft(x[:, ch], nfft) * np.fft.rfft(ir, nfft), nfft)[: len(x)]
        out[:, ch] = (1 - mix) * x[:, ch] + mix * wet * 0.9
    return out


def place(buf, sig, t, gain=1.0, pan=0.0):
    i = int(t * SR)
    if i >= len(buf):
        return
    sig = sig[: len(buf) - i]
    if sig.ndim == 1:
        l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        buf[i:i + len(sig), 0] += sig * gain * l
        buf[i:i + len(sig), 1] += sig * gain * r
    else:
        buf[i:i + len(sig)] += sig * gain


def music(duration=60.0, seed=1):
    rng = np.random.default_rng(seed)
    buf = np.zeros((int((duration + 3) * SR), 2))
    # D minor colour: Dm9 - Bbmaj7 - Fmaj7(add9) - C6/9 ... resolving to Dm(add9)
    prog = [[50, 57, 60, 64, 65], [46, 53, 57, 60, 62], [41, 53, 57, 60, 67], [48, 55, 60, 62, 64],
            [50, 57, 60, 64, 65], [43, 55, 58, 62, 65], [46, 53, 57, 60, 65], [45, 52, 57, 60, 64]]
    chord_len = 4.0
    n_ch = int(np.ceil(duration / chord_len))
    for k in range(n_ch):
        ch = prog[k % len(prog)] if k < n_ch - 2 else [50, 57, 60, 64, 69]
        t0 = k * chord_len
        dur = chord_len + 1.8
        for i, n in enumerate(ch):
            v = pad_voice(note_hz(n), dur, rng)
            v *= env_adsr(len(v), 1.4, 1.8)[:, None]
            g = 0.05 if i else 0.07
            place(buf, v, t0, gain=g)
        # bass drone
        tb = np.arange(int(dur * SR)) / SR
        b = np.sin(2 * np.pi * note_hz(ch[0] - 12) * tb) * env_adsr(len(tb), 1.0, 1.8)
        place(buf, b, t0, gain=0.09)
    # arpeggiated plucks: sparse in opening/closing, steadier while the experiments run
    beat = 60 / 76
    t = 2.0
    step = 0
    while t < duration - 1.5:
        k = int(t // chord_len)
        ch = prog[k % len(prog)] if k < n_ch - 2 else [50, 57, 60, 64, 69]
        busy = 17 <= t <= 53
        if busy or step % 2 == 0:
            pattern = [2, 3, 4, 3, 1, 4, 2, 3]
            n = ch[pattern[step % 8]] + 12
            gain = (0.05 if busy else 0.04) * (0.8 + 0.2 * rng.random())
            place(buf, pluck(note_hz(n)), t, gain=gain, pan=0.5 * np.sin(step * 0.9))
        t += beat / (2 if busy else 1)
        step += 1
    # slow fade in / out
    n = len(buf)
    fade = np.ones(n)
    fi, fo = int(2.5 * SR), int(4.0 * SR)
    fade[:fi] = np.linspace(0, 1, fi) ** 2
    end = int(duration * SR)
    fade[end - fo:end] = np.linspace(1, 0, fo) ** 1.5
    fade[end:] = 0
    buf *= fade[:, None]
    return buf


# ---- sound design ----
def tick():
    t = np.arange(int(0.06 * SR)) / SR
    return np.sin(2 * np.pi * 1800 * t) * np.exp(-t * 90) + 0.3 * np.random.default_rng(0).normal(0, 1, len(t)) * np.exp(-t * 200)


def chime(up=True):
    a = pluck(note_hz(81), 1.2, 0.5)
    b = pluck(note_hz(88 if up else 76), 1.2, 0.5)
    out = np.zeros(len(a) + int(0.09 * SR))
    out[: len(a)] += a
    out[int(0.09 * SR):] += b
    return out


def thud():
    t = np.arange(int(0.5 * SR)) / SR
    return np.sin(2 * np.pi * (90 - 40 * t) * t) * np.exp(-t * 9)


def whoosh(dur=1.4):
    rng = np.random.default_rng(3)
    n = int(dur * SR)
    x = rng.normal(0, 1, n)
    x = lowpass(x, 900)
    e = np.sin(np.linspace(0, np.pi, n)) ** 2
    return x * e


def build_track(cues, duration=60.0, path="soundtrack.wav"):
    m = music(duration)
    fx = np.zeros_like(m)
    lib = {"tick": tick(), "pass": chime(True), "fail": thud(), "whoosh": whoosh()}
    for t, kind, gain in cues:
        place(fx, lib[kind], max(0, t), gain=gain)
    mix = m + fx
    mix = reverb(mix, mix=0.28)
    mix = mix[: int(duration * SR)]
    mix /= np.max(np.abs(mix)) + 1e-9
    mix *= 0.6   # leave headroom; the film is meant to be quiet and calm
    data = (mix * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(data.tobytes())
    return path
