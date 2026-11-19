"""Geometric Path Features (GPF): compact reference implementation of the features used in the paper.

Every collection X is an array of shape (n_series, n_channels, length).
"""
from itertools import combinations, permutations

import numpy as np


def sig(X, level):
    """Path signature truncated at `level` (the Sig baseline)."""
    import iisignature
    return np.array([iisignature.sig(x.T, level) for x in X])


# ── GPF of the main experiments (Table 1) ────────────────────────────────────
def curvature(c):
    """Radius of curvature of a 1-D series (zero-padded ends), divided by 1000 (element-wise, as in the paper's runs)."""
    p = np.concatenate([[0], c, [0]])
    return np.array([(1 + (p[i + 1] - p[i]) ** 2) ** 1.5 / (abs(p[i + 1] - 2 * p[i] + p[i - 1]) + 0.001) / 1000
                     for i in range(1, len(p) - 1)])


def gpf(X, max_k=4):
    """[signed | unsigned] GPF. Each feature sums a curve over time (signed) or its absolute value (unsigned):
    k=1: each channel minus its chord and minus its curvature transform; k=2: x_i - x_j for every pair of channels;
    k=3..max_k: the product of every group of k channels, plus the product of all d channels when 3 <= d <= max_k."""
    F = []
    for x in X:
        d, T = x.shape
        K = min(max_k, d)
        V = [c - v for c in x for v in (np.linspace(c[0], c[-1], T), curvature(c))]
        V += [x[i] - x[j] for i, j in combinations(range(d), 2)]
        V += [np.prod(x[list(g)], axis=0) for k in range(3, K + 1) for g in combinations(range(d), k)]
        V += [np.prod(x, axis=0)] if 3 <= d <= K else []
        V = np.array(V)
        F.append(np.concatenate([V.sum(1), np.abs(V).sum(1)]))
    return np.array(F)


# ── Channel-augmented GPF_SU+Sig (row of the comparison with established feature sets) ──
def znorm(a):
    """z-normalise every channel of every series along time."""
    sd = a.std(-1, keepdims=True)
    return (a - a.mean(-1, keepdims=True)) / np.where(sd < 1e-8, 1.0, sd)


def augment(x):
    """(d, T) -> (8d, T): raw, chord, curvature radius and |x|, each also z-normalised."""
    t = np.linspace(0, 1, x.shape[1])
    p = np.pad(x, ((0, 0), (1, 1)), mode="edge")
    r = np.clip((1 + (p[:, 2:] - p[:, :-2]) ** 2) ** 1.5 / (np.abs(p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]) + 1e-6) / 1000, 0.001, 100)
    V = [x, x[:, [0]] + (x[:, [-1]] - x[:, [0]]) * t, r, np.abs(x)]
    return np.vstack(V + [znorm(v) for v in V])


def gpf_aug(X, budget=3e8):
    """[level-2 signature | signed | unsigned pair GPF] of the augmented channels (float32)."""
    A = np.stack([augment(x).astype(np.float32) for x in X])
    n, m, T = A.shape
    I, J = np.triu_indices(m, 1)
    S, U, c = [], [], max(1, int(budget // (n * T * 4)))
    for q in range(0, len(I), c):
        D = np.clip(A[:, I[q:q + c]] - A[:, J[q:q + c]], -1e10, 1e10)
        S.append(D.sum(2)), U.append(np.abs(D).sum(2))
    G = np.nan_to_num(np.clip(np.concatenate(S + U, 1), -1e10, 1e10), nan=0.0, posinf=1e6, neginf=-1e6)
    B = np.nan_to_num(sig(A, 2).astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    return np.concatenate([B, G], 1)


# ── GPF-W: windowed GPF ──────────────────────────────────────────────────────
def _scales(T):
    """Window lengths T/8, T/16, T/32 (at least 8 points, at most T); the stride is half a window."""
    return sorted({min(T, max(8, T // 2 ** k)) for k in (3, 4, 5)})


def _starts(T, L):
    return np.arange(0, T - L + 1, max(1, L // 2))


def _wmean(cum, T, L):
    """(windows, n, F) window means from a cumulative sum that starts with a zero."""
    a = _starts(T, L)
    return ((cum[:, :, a + L] - cum[:, :, a]) / L).transpose(2, 0, 1).astype(np.float32)


def _cum(V):
    return np.concatenate([np.zeros(V.shape[:2] + (1,)), np.cumsum(V, 2)], 2)


def _st(V, pos=False):
    """Statistics over the windows: max, min, std, position of the max, and the position of the min (pos) or the share > 0."""
    w = max(1, V.shape[0] - 1)
    return np.concatenate([V.max(0), V.min(0), V.std(0), V.argmax(0) / w, V.argmin(0) / w if pos else (V > 0).mean(0)], 1)


def _by_pairs(f, n_pairs, size):
    """Run f on chunks of channel pairs (to bound memory); f returns column blocks, joined chunk-wise, then side by side.
    The chunk sizes are those of the paper's runs: statistics are laid out chunk by chunk, so they fix the column order."""
    parts = [f(slice(q, q + size)) for q in range(0, n_pairs, size)]
    return np.concatenate([np.concatenate(b, 1) for b in zip(*parts)], 1)


def _pairs(A):
    """Signed and unsigned sum of x_i - x_j for every pair, over the whole series and every dyadic segment to depth 3."""
    n, m, T = A.shape
    I, J = np.triu_indices(m, 1)
    seg = [(i * (T >> k), (i + 1) * (T >> k) if i < 2 ** k - 1 else T) for k in range(4) for i in range(2 ** k)]

    def f(q):
        V = np.clip(A[:, I[q]] - A[:, J[q]], -1e10, 1e10)
        W = np.abs(V)
        return [s for a, b in seg for s in (V[:, :, a:b].sum(2), W[:, :, a:b].sum(2))]
    return np.clip(_by_pairs(f, len(I), max(1, int(2e8 // (n * T * 4)))), -1e10, 1e10)


def _sig2(A):
    """Level-2 signature (numpy)."""
    D = np.diff(A.astype(np.float64), axis=2)
    C = np.cumsum(D, 2) - D
    S2 = np.einsum("nat,nbt->nab", C, D) + 0.5 * np.einsum("nat,nbt->nab", D, D)
    return np.nan_to_num(np.concatenate([D.sum(2), S2.reshape(len(A), -1)], 1).astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6)


def _increments_levy(A, budget=6e8):
    """Per window: the increment of each channel and the Lévy area of each pair."""
    n, m, T = A.shape
    Ad = A.astype(np.float64)
    D = np.diff(Ad, axis=2)
    C = np.cumsum(D, 2) - D
    I, J = np.triu_indices(m, 1)

    def levy(a, b, q):
        d = D[:, :, a:b]
        S = np.matmul(C[:, :, a:b], d.transpose(0, 2, 1)) - C[:, :, a][:, :, None] * d.sum(2)[:, None, :]
        return (0.5 * (S - S.transpose(0, 2, 1)))[:, I[q], J[q]]

    def f(q):
        out = []
        for L in _scales(T):
            a = _starts(T, L)
            inc = _st(np.stack([Ad[:, :, s + L - 1] - Ad[:, :, s] for s in a]).astype(np.float32)) if q.start == 0 else np.zeros((n, 0))
            out += [inc, _st(np.stack([levy(s, s + L - 1, q) for s in a]).astype(np.float32))]
        return out
    w = max(len(_starts(T, L)) for L in _scales(T))
    return np.nan_to_num(_by_pairs(f, len(I), max(1, int(budget // (n * w * 16)))), nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float32)


def _windowed(A):
    """Signed and unsigned GPF of every pair inside each window."""
    n, m, T = A.shape
    I, J = np.triu_indices(m, 1)

    def f(q):
        V = A[:, I[q]].astype(np.float64) - A[:, J[q]]
        cs, ca = _cum(V), _cum(np.abs(V))
        return [s for L in _scales(T) for s in (_st(_wmean(cs, T, L)), _st(_wmean(ca, T, L), pos=True))]
    return np.nan_to_num(_by_pairs(f, len(I), max(1, int(2e8 // (n * (T + 1) * 32)))), nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float32)


def _windowed_local(x):
    """mean |z_i - z_j| of every pair inside each window, each channel z-normalised inside the window."""
    n, m, T = x.shape
    I, J = np.triu_indices(m, 1)
    if not len(I):
        return np.zeros((n, 0), np.float32)

    def z(a, L):
        Z = x[:, :, a:a + L] - x[:, :, a:a + L].mean(2, keepdims=True)
        sd = Z.std(2, keepdims=True)
        return Z / np.where(sd < 1e-6, 1.0, sd)
    out = []
    for L in _scales(T):
        a = _starts(T, L)
        c = max(1, int(min(1e8 // (n * L * 8), 2e8 // (len(a) * n * 4))))   # pair chunks, as in the paper's runs
        for q in range(0, len(I), c):
            R = [np.abs(Z[:, I[q:q + c]] - Z[:, J[q:q + c]]).mean(2) for Z in (z(s, L) for s in a)]
            out.append(_st(np.stack(R), pos=True))
    return np.nan_to_num(np.concatenate(out, 1), nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float32)


def _windowed_vector(x, xn):
    """GPF vector between groups of 3 consecutive channels (of x and of its z-normalised copy), per window:
    gross area sum_t ||B_t|| and net components, B_ab = dC1_a (C2 - C1)_b - dC1_b (C2 - C1)_a."""
    n, d, T = x.shape
    p = min(3, d)
    if p < 2:
        return np.zeros((n, 0), np.float32)
    V = np.concatenate([x, xn], 1).astype(np.float64)
    G = [np.arange(i * p, (i + 1) * p) + o for o in (0, d) for i in range(d // p)]
    ia, ib = map(list, zip(*combinations(range(p), 2)))
    out = []
    for g1, g2 in permutations(range(len(G)), 2):
        C1 = V[:, G[g1]]
        dC1 = np.diff(C1, axis=2)
        M = V[:, G[g2]] - C1
        M = 0.5 * (M[:, :, 1:] + M[:, :, :-1])
        B = dC1[:, ia] * M[:, ib] - dC1[:, ib] * M[:, ia]
        cg, cb = _cum(np.sqrt((B * B).sum(1, keepdims=True))), _cum(B)
        out += [s for L in _scales(T - 1) for s in (_st(_wmean(cg, T - 1, L), pos=True), _st(_wmean(cb, T - 1, L)))]
    return np.nan_to_num(np.clip(np.concatenate(out, 1), -1e10, 1e10)).astype(np.float32)


def gpf_w(X):
    """GPF-W of a collection (label-free, so it can be computed on train and test series together)."""
    x, xn = X.astype(np.float32), znorm(X).astype(np.float32)
    A = np.concatenate([x, xn], 1)
    return np.concatenate([_pairs(A), _sig2(A), _increments_levy(A), _windowed(A), _windowed_local(x), _windowed_vector(x, xn)], 1)
