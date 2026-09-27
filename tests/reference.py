"""Per-pixel reference implementation of phase-linking temporal coherence.

A plain loop transcription of the ESA SNAP microwave toolbox phase linking
(PhaseLinkingOp, KSSelector, ADSelector, TLogSelector, CovarianceMatrix,
EVDEstimator, EMIEstimator, TemporalCoherence), used to check the vectorized
module. Only the Hermitian eigensolver is taken from NumPy.

One deliberate difference: the Anderson-Darling statistic uses the 1 / N
prefactor of Scholz and Stephens (1987) eq. 6 (as scipy.stats.anderson_ksamp
with midrank=False) instead of the (N - 1) / N^2 of ADSelector, which belongs
to the midrank statistic of their eq. 7 and scales the no-ties sum by
(N - 1) / N. Also, a coherence zeroed by the bias correction has phase 0
here, where SNAP's atan2 of a signed zero may return +-pi, and so has an
eigenvector component below 1e-3 of the largest, whose phase is rounding
noise in SNAP; phases are relative to the median date (SNAP's default
reference epoch). With the bias correction, a pixel whose zeroed
coherences split the dates into unlinked groups has no temporal coherence
(NaN), where SNAP returns an arbitrary value.
"""

import math
from statistics import NormalDist

import numpy as np


def ks_accept(centre, candidate, alpha):
    n = len(centre)
    critical = math.sqrt(-0.5 * math.log(alpha / 2.0)) * math.sqrt(2.0 / n)
    a, b = sorted(centre), sorted(candidate)
    ia = ib = 0
    fa = fb = d = 0.0
    while ia < n and ib < n:
        if a[ia] <= b[ib]:
            fa += 1.0 / n
            ia += 1
        else:
            fb += 1.0 / n
            ib += 1
        d = max(d, abs(fa - fb))
    while ia < n:
        fa += 1.0 / n
        ia += 1
        d = max(d, abs(fa - fb))
    while ib < n:
        fb += 1.0 / n
        ib += 1
        d = max(d, abs(fa - fb))
    return d <= critical


def ad_statistic(centre, candidate, sigma):
    n = len(centre)
    total = 2 * n
    norm = 1.0 / total * (2.0 / n)
    a, b = sorted(centre), sorted(candidate)
    ia = ib = m = 0
    s = 0.0
    for j in range(1, total):
        if ia < n and (ib >= n or a[ia] <= b[ib]):
            m += 1
            ia += 1
        else:
            ib += 1
        s += (total * m - j * n) ** 2 / (j * (total - j))
    return (norm * s - 1.0) / sigma


def tlog_accept(centre, candidate, alpha):
    n = len(centre)
    critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    lc = [math.log(v) for v in centre]
    lx = [math.log(v) for v in candidate]
    mc, mx = sum(lc) / n, sum(lx) / n
    vc = sum((v - mc) ** 2 for v in lc) / (n - 1)
    vx = sum((v - mx) ** 2 for v in lx) / (n - 1)
    se = math.sqrt(vc / n + vx / n)
    if not se > 0:
        return True
    return abs(mc - mx) / se <= critical


AD_ALPHAS = (0.25, 0.10, 0.05, 0.025, 0.01, 0.005, 0.001)
AD_CRITICAL = (0.325, 1.226, 1.961, 2.718, 3.752, 4.592, 6.546)


def ad_sigma(n):
    """Standard deviation of A2 under the null, two samples of size n."""
    k, total = 2, 2 * n
    big_h = 2.0 / n
    h = sum(1.0 / j for j in range(1, total))
    g = sum(1.0 / ((total - i) * j) for i in range(1, total - 1) for j in range(i + 1, total))
    a = (4 * g - 6) * (k - 1) + (10 - 6 * g) * big_h
    b = (2 * g - 4) * k * k + 8 * h * k + (2 * g - 14 * h - 4) * big_h - 8 * h + 4 * g - 6
    c = (6 * h + 2 * g - 2) * k * k + (4 * h - 4 * g + 6) * k + (2 * h - 6) * big_h + 4 * h
    d = (2 * h + 6) * k * k - 4 * h * k
    var = (a * total**3 + b * total**2 + c * total + d) / (
        (total - 1.0) * (total - 2.0) * (total - 3.0)
    )
    return math.sqrt(var) if var > 0 else 1.0


def ad_critical(alpha):
    """T_1 critical value, linear in log(alpha) between table levels."""
    return float(np.interp(np.log(alpha), np.log(AD_ALPHAS[::-1]), AD_CRITICAL[::-1]))


def coherence_matrix(cov, looks, bias_correction):
    n = cov.shape[0]
    t = np.zeros((n, n), dtype=complex)
    diag = [math.sqrt(cov[i, i].real / looks) for i in range(n)]
    for i in range(n):
        t[i, i] = 1.0
        for j in range(i + 1, n):
            dij = diag[i] * diag[j]
            if dij <= 0:
                value = 0j
            else:
                value = cov[i, j] / looks / dij
                if bias_correction and looks > 1:
                    m2 = abs(value) ** 2
                    m2c = (looks * m2 - 1.0) / (looks - 1.0)
                    value *= math.sqrt(m2c / m2) if m2 > 0 and m2c > 0 else 0.0
            t[i, j] = value
            t[j, i] = value.conjugate()
    return t


def phases(t, estimator):
    n = t.shape[0]
    if estimator == "evd":
        _w, v = np.linalg.eigh(t)
        u = v[:, -1]
    else:
        gamma = np.abs(t)
        w, v = np.linalg.eigh(gamma)
        inv = np.zeros((n, n))
        for k in range(n):
            if abs(w[k]) >= 1e-9:
                inv += np.outer(v[:, k], v[:, k]) / w[k]
        _w, v = np.linalg.eigh(inv * t)
        u = v[:, 0]
    # Phases relative to the median date, as SNAP's default reference epoch
    # (or to the strongest component when that one is negligible). A
    # negligible component has no phase of its own and takes the phase of
    # that datum.
    top = max(abs(z) for z in u)
    floor = 1e-3 * top
    ref = n // 2
    if abs(u[ref]) <= floor:
        ref = int(np.argmax(np.abs(u)))
    datum = math.atan2(u[ref].imag, u[ref].real)
    return [math.atan2(z.imag, z.real) - datum if abs(z) > floor else 0.0 for z in u]


def temporal_coherence(t, phi):
    n = t.shape[0]
    re = im = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            # Adding 0.0 clears signed zeros: SNAP's atan2(-0.0, -0.0) = -pi
            # gives a pair zeroed by the bias correction an arbitrary phase.
            residual = math.atan2(t[i, j].imag + 0.0, t[i, j].real + 0.0) - (
                phi[i] - phi[j]
            )
            re += math.cos(residual)
            im += math.sin(residual)
    return math.hypot(re, im) / (0.5 * n * (n - 1))


def connected(t):
    """Whether the non-zero coherences link all dates (not in SNAP)."""
    n = t.shape[0]
    seen, todo = {0}, [0]
    while todo:
        i = todo.pop()
        for j in range(n):
            if j not in seen and t[i, j] != 0:
                seen.add(j)
                todo.append(j)
    return len(seen) == n


def stack_coherence(
    slc,
    window=(21, 7),
    shp_test="ks",
    alpha=0.05,
    min_shp=20,
    estimator="evd",
    bias_correction=False,
):
    """Temporal coherence and SHP count of a stack slc (N, rows, cols).

    NaN (zero samples) marks no-data. Return (coherence, count) with NaN and
    -1 where the centre is invalid, NaN coherence below the SHP minimum.
    """
    n, rows, cols = slc.shape
    ha, hr = window[0] // 2, window[1] // 2
    valid = np.all(np.isfinite(slc) & (slc != 0), axis=0)
    finite = np.where(np.isfinite(slc), slc, 0)
    # sqrt(re^2 + im^2) as SNAP: samples with the same power tie exactly,
    # where np.abs (hypot) may differ by one unit in the last place.
    amp = np.sqrt(finite.real**2 + finite.imag**2)
    coh = np.full((rows, cols), np.nan)
    count = np.full((rows, cols), -1)
    min_looks = max(min_shp, n)
    if shp_test == "ad":
        sigma, critical = ad_sigma(n), ad_critical(alpha)
    for y in range(rows):
        for x in range(cols):
            if not valid[y, x]:
                continue
            centre = list(amp[:, y, x])
            cov = np.zeros((n, n), dtype=complex)
            looks = 0
            for yy in range(max(0, y - ha), min(rows, y + ha + 1)):
                for xx in range(max(0, x - hr), min(cols, x + hr + 1)):
                    if not valid[yy, xx]:
                        continue
                    cand = list(amp[:, yy, xx])
                    if shp_test == "ks":
                        ok = ks_accept(centre, cand, alpha)
                    elif shp_test == "ad":
                        ok = ad_statistic(centre, cand, sigma) <= critical
                    else:
                        ok = tlog_accept(centre, cand, alpha)
                    if not ok:
                        continue
                    s = slc[:, yy, xx]
                    cov += np.outer(s, s.conj())
                    looks += 1
            count[y, x] = looks
            if looks < min_looks:
                continue
            t = coherence_matrix(cov, looks, bias_correction)
            if bias_correction and not connected(t):
                continue
            coh[y, x] = temporal_coherence(t, phases(t, estimator))
    return coh, count
