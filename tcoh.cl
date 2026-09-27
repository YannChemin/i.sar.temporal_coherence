/* Phase-linking temporal coherence, one work-item per output pixel.
 *
 * Single precision only: no double precision extension is required.
 *
 * Compile-time constants (set by the host with -D):
 *   N          number of dates
 *   WA, WR     search window size in lines and samples (odd)
 *   W          width of the padded strip (columns + WR - 1)
 *   SHP_TEST   0 Kolmogorov-Smirnov, 1 Anderson-Darling, 2 t-test on logs
 *   EMI        1 for the EMI estimator, 0 for EVD
 *   BIAS       1 to remove the bias of the coherence magnitudes
 *   MIN_LOOKS  minimum number of SHPs
 *   KS_MAX     largest accepted KS distance, in steps of 1 / N
 *   AD_NORM, AD_SIGMA, AD_CRIT  Anderson-Darling normalization and critical
 *   TLOG_CRIT  critical value of the t-test
 *
 * Buffers cover the padded strip, row-major with W columns: slc holds the N
 * complex samples of a pixel, amp their amplitudes sorted increasingly and
 * valid is non-zero where every date has a non-null, non-zero sample.
 */

#define MAX_SWEEPS   40
#define EIG_FLOOR    1.0e-6f
/* Smallest Cholesky pivot of |T| (unit diagonal) before falling back to
   the eigendecomposition pseudo-inverse. */
#define CHOL_FLOOR   1.0e-5f
/* Relative magnitude below which a phasor component has no phase. */
#define PHASOR_FLOOR 1.0e-3f

int ks_accept(__global const float *a, __global const float *b)
{
    /* Merge walk with the centre sample first on ties; |F_a - F_b| is
       |ia - ib| / N and only decreases once one sample is exhausted. */
    int ia = 0, ib = 0, d = 0;

    while (ia < N && ib < N) {
        int diff;

        if (a[ia] <= b[ib])
            ia++;
        else
            ib++;
        diff = ia - ib;
        if (diff < 0)
            diff = -diff;
        if (diff > d)
            d = diff;
    }
    return d <= KS_MAX;
}

int ad_accept(__global const float *a, __global const float *b)
{
    /* Scholz and Stephens (1987) eq. 6 rank statistic, standardized. */
    int ia = 0, ib = 0, m = 0, j;
    float s = 0.0f;

    for (j = 1; j < 2 * N; j++) {
        float num;

        if (ia < N && (ib >= N || a[ia] <= b[ib])) {
            m++;
            ia++;
        }
        else
            ib++;
        num = (float)(2 * N * m - j * N);
        s += num * num / (float)(j * (2 * N - j));
    }
    return (AD_NORM * s - 1.0f) / AD_SIGMA <= AD_CRIT;
}

int tlog_accept(float mean_a, float var_a, __global const float *b)
{
    float mean_b = 0.0f, var_b = 0.0f, se;
    int k;

    for (k = 0; k < N; k++)
        mean_b += log(b[k]);
    mean_b /= N;
    for (k = 0; k < N; k++) {
        float d = log(b[k]) - mean_b;

        var_b += d * d;
    }
    var_b /= N - 1;
    se = sqrt((var_a + var_b) / N);
    if (!(se > 0.0f))
        return 1;
    return fabs(mean_a - mean_b) / se <= TLOG_CRIT;
}

/* Cyclic Jacobi eigendecomposition of the Hermitian matrix ar + i ai
 * (N x N, row-major, both triangles stored). The matrix is destroyed and
 * its diagonal holds the eigenvalues; the columns of vr + i vi are the
 * eigenvectors. Each rotation first turns a_pq real by rescaling row and
 * column q with its phase, then applies the real Jacobi rotation. */
void hermitian_eigen(float *ar, float *ai, float *vr, float *vi)
{
    int p, q, k, sweep;

    for (p = 0; p < N * N; p++) {
        vr[p] = 0.0f;
        vi[p] = 0.0f;
    }
    for (p = 0; p < N; p++)
        vr[p * N + p] = 1.0f;

    for (sweep = 0; sweep < MAX_SWEEPS; sweep++) {
        float off = 0.0f, diag = 0.0f;

        for (p = 0; p < N; p++) {
            diag += ar[p * N + p] * ar[p * N + p];
            for (q = p + 1; q < N; q++)
                off += ar[p * N + q] * ar[p * N + q] +
                       ai[p * N + q] * ai[p * N + q];
        }
        if (off <= 1.0e-14f * diag)
            break;

        for (p = 0; p < N - 1; p++) {
            for (q = p + 1; q < N; q++) {
                float xr = ar[p * N + q], xi = ai[p * N + q];
                float r = hypot(xr, xi);
                float cph, sph, theta, t, c, s, app, aqq;

                if (r == 0.0f)
                    continue;
                /* Column q times exp(-i phi), row q times exp(i phi). */
                cph = xr / r;
                sph = xi / r;
                for (k = 0; k < N; k++) {
                    float re = ar[k * N + q], im = ai[k * N + q];

                    ar[k * N + q] = re * cph + im * sph;
                    ai[k * N + q] = im * cph - re * sph;
                    re = ar[q * N + k];
                    im = ai[q * N + k];
                    ar[q * N + k] = re * cph - im * sph;
                    ai[q * N + k] = im * cph + re * sph;
                    re = vr[k * N + q];
                    im = vi[k * N + q];
                    vr[k * N + q] = re * cph + im * sph;
                    vi[k * N + q] = im * cph - re * sph;
                }

                /* Real rotation annihilating a_pq = r. */
                theta = (ar[q * N + q] - ar[p * N + p]) / (2.0f * r);
                if (fabs(theta) > 1.0e18f)
                    t = 0.5f / theta;
                else
                    t = (theta >= 0.0f ? 1.0f : -1.0f) /
                        (fabs(theta) + sqrt(theta * theta + 1.0f));
                c = rsqrt(t * t + 1.0f);
                s = t * c;
                app = ar[p * N + p] - t * r;
                aqq = ar[q * N + q] + t * r;
                for (k = 0; k < N; k++) {
                    float kpr, kpi, kqr, kqi;

                    if (k != p && k != q) {
                        kpr = ar[k * N + p];
                        kpi = ai[k * N + p];
                        kqr = ar[k * N + q];
                        kqi = ai[k * N + q];
                        ar[k * N + p] = c * kpr - s * kqr;
                        ai[k * N + p] = c * kpi - s * kqi;
                        ar[k * N + q] = s * kpr + c * kqr;
                        ai[k * N + q] = s * kpi + c * kqi;
                        ar[p * N + k] = ar[k * N + p];
                        ai[p * N + k] = -ai[k * N + p];
                        ar[q * N + k] = ar[k * N + q];
                        ai[q * N + k] = -ai[k * N + q];
                    }
                    kpr = vr[k * N + p];
                    kpi = vi[k * N + p];
                    kqr = vr[k * N + q];
                    kqi = vi[k * N + q];
                    vr[k * N + p] = c * kpr - s * kqr;
                    vi[k * N + p] = c * kpi - s * kqi;
                    vr[k * N + q] = s * kpr + c * kqr;
                    vi[k * N + q] = s * kpi + c * kqi;
                }
                ar[p * N + p] = app;
                ar[q * N + q] = aqq;
                ai[p * N + p] = 0.0f;
                ai[q * N + q] = 0.0f;
                ar[p * N + q] = 0.0f;
                ai[p * N + q] = 0.0f;
                ar[q * N + p] = 0.0f;
                ai[q * N + p] = 0.0f;
            }
        }
    }
}

/* Reduce the Hermitian matrix ar + i ai (both triangles stored) to the real
 * symmetric tridiagonal T = Q^H A Q, Q = H_0 ... H_{N-2}, as LAPACK zhetd2
 * (lower): diagonal d, sub-diagonal e. H_k = I - tau_k v_k v_k^H acts on
 * rows k+1.., v_k = (1, A[k+2.., k]); tau_k is (taur, taui)[k]. The
 * trailing matrix is overwritten; wr, wi are work arrays of N. */
void tridiagonalize(float *ar, float *ai, float *d, float *e, float *taur,
                    float *taui, float *wr, float *wi)
{
    int k, i, j;

    for (k = 0; k < N - 1; k++) {
        const float alr = ar[(k + 1) * N + k], ali = ai[(k + 1) * N + k];
        float xnorm2 = 0.0f, beta, tr_ = 0.0f, ti_ = 0.0f;

        d[k] = ar[k * N + k];
        for (i = k + 2; i < N; i++)
            xnorm2 +=
                ar[i * N + k] * ar[i * N + k] + ai[i * N + k] * ai[i * N + k];
        if (xnorm2 == 0.0f && ali == 0.0f)
            beta = alr;
        else {
            /* zlarfg: H^H (alpha, x) = (beta, 0), beta real. */
            float dr, di, den, sr, si;

            beta = -copysign(sqrt(alr * alr + ali * ali + xnorm2), alr);
            tr_ = (beta - alr) / beta;
            ti_ = -ali / beta;
            dr = alr - beta;
            di = ali;
            den = dr * dr + di * di;
            sr = dr / den;
            si = -di / den;
            for (i = k + 2; i < N; i++) {
                const float xr = ar[i * N + k], xi = ai[i * N + k];

                ar[i * N + k] = xr * sr - xi * si;
                ai[i * N + k] = xr * si + xi * sr;
            }
        }
        e[k] = beta;
        taur[k] = tr_;
        taui[k] = ti_;
        ar[(k + 1) * N + k] = 1.0f;
        ai[(k + 1) * N + k] = 0.0f;
        if (tr_ == 0.0f && ti_ == 0.0f)
            continue;

        /* A22 := H^H A22 H: w = tau A22 v, w += -tau/2 (w^H v) v,
           A22 -= v w^H + w v^H. */
        {
            float pr = 0.0f, pi = 0.0f, hr, hi;

            for (i = k + 1; i < N; i++) {
                float s_r = 0.0f, s_i = 0.0f;

                for (j = k + 1; j < N; j++) {
                    const float a_r = ar[i * N + j], a_i = ai[i * N + j];
                    const float v_r = ar[j * N + k], v_i = ai[j * N + k];

                    s_r += a_r * v_r - a_i * v_i;
                    s_i += a_r * v_i + a_i * v_r;
                }
                wr[i] = tr_ * s_r - ti_ * s_i;
                wi[i] = tr_ * s_i + ti_ * s_r;
                /* w^H v */
                pr += wr[i] * ar[i * N + k] + wi[i] * ai[i * N + k];
                pi += wr[i] * ai[i * N + k] - wi[i] * ar[i * N + k];
            }
            hr = -0.5f * (tr_ * pr - ti_ * pi);
            hi = -0.5f * (tr_ * pi + ti_ * pr);
            for (i = k + 1; i < N; i++) {
                const float v_r = ar[i * N + k], v_i = ai[i * N + k];

                wr[i] += hr * v_r - hi * v_i;
                wi[i] += hr * v_i + hi * v_r;
            }
            for (i = k + 1; i < N; i++) {
                const float vir = ar[i * N + k], vii = ai[i * N + k];

                for (j = k + 1; j < N; j++) {
                    const float vjr = ar[j * N + k], vji = ai[j * N + k];

                    /* v_i conj(w_j) + w_i conj(v_j) */
                    ar[i * N + j] -=
                        vir * wr[j] + vii * wi[j] + wr[i] * vjr + wi[i] * vji;
                    ai[i * N + j] -=
                        vii * wr[j] - vir * wi[j] + wi[i] * vjr - wr[i] * vji;
                }
            }
        }
    }
    d[N - 1] = ar[(N - 1) * N + N - 1];
}

/* Number of eigenvalues of the tridiagonal (d, e) smaller than x. */
int sturm_count(const float *d, const float *e, float x, float pivmin)
{
    float q = d[0] - x;
    int i, count = q < 0.0f;

    for (i = 1; i < N; i++) {
        if (fabs(q) < pivmin)
            q = -pivmin;
        q = d[i] - x - e[i - 1] * e[i - 1] / q;
        count += q < 0.0f;
    }
    return count;
}

/* Eigenvalue of rank index (0 smallest) of the tridiagonal, by bisection
   from the Gershgorin interval. */
float tridiagonal_eigenvalue(const float *d, const float *e, int index,
                             float *norm)
{
    float lo = d[0], hi = d[0], pivmin;
    int i, it;

    for (i = 0; i < N; i++) {
        float r =
            (i > 0 ? fabs(e[i - 1]) : 0.0f) + (i < N - 1 ? fabs(e[i]) : 0.0f);

        lo = fmin(lo, d[i] - r);
        hi = fmax(hi, d[i] + r);
    }
    *norm = fmax(fabs(lo), fabs(hi));
    pivmin = 1.0e-30f + 1.0e-7f * *norm * 1.0e-7f;
    for (it = 0; it < 64; it++) {
        const float mid = 0.5f * (lo + hi);

        if (mid <= lo || mid >= hi)
            break;
        if (sturm_count(d, e, mid, pivmin) > index)
            hi = mid;
        else
            lo = mid;
    }
    return 0.5f * (lo + hi);
}

/* Eigenvector y of the tridiagonal (d, e) for the eigenvalue lambda, by
   inverse iteration with a partially pivoted LU of T - lambda I. */
void tridiagonal_eigenvector(const float *d, const float *e, float lambda,
                             float norm, float *y, float *u0, float *u1,
                             float *u2, float *l, int *piv)
{
    const float tiny = 1.0e-6f * fmax(norm, 1.0e-30f);
    float dd = d[0] - lambda, s1 = N > 1 ? e[0] : 0.0f;
    int k, it;

    for (k = 0; k < N - 1; k++) {
        const float sub = e[k];
        const float nd = d[k + 1] - lambda;
        const float ns = k + 1 < N - 1 ? e[k + 1] : 0.0f;

        if (fabs(dd) >= fabs(sub)) {
            if (fabs(dd) < tiny)
                dd = dd < 0.0f ? -tiny : tiny;
            l[k] = sub / dd;
            piv[k] = 0;
            u0[k] = dd;
            u1[k] = s1;
            u2[k] = 0.0f;
            dd = nd - l[k] * s1;
            s1 = ns;
        }
        else {
            l[k] = dd / sub;
            piv[k] = 1;
            u0[k] = sub;
            u1[k] = nd;
            u2[k] = ns;
            dd = s1 - l[k] * nd;
            s1 = -l[k] * ns;
        }
    }
    if (fabs(dd) < tiny)
        dd = dd < 0.0f ? -tiny : tiny;
    u0[N - 1] = dd;
    u1[N - 1] = 0.0f;
    u2[N - 1] = 0.0f;

    for (k = 0; k < N; k++)
        y[k] = 1.0f;
    for (it = 0; it < 3; it++) {
        float nrm = 0.0f;

        for (k = 0; k < N - 1; k++) {
            if (piv[k]) {
                const float t = y[k];

                y[k] = y[k + 1];
                y[k + 1] = t;
            }
            y[k + 1] -= l[k] * y[k];
        }
        for (k = N - 1; k >= 0; k--) {
            float v = y[k];

            if (k + 1 < N)
                v -= u1[k] * y[k + 1];
            if (k + 2 < N)
                v -= u2[k] * y[k + 2];
            y[k] = v / u0[k];
        }
        for (k = 0; k < N; k++)
            nrm = fmax(nrm, fabs(y[k]));
        for (k = 0; k < N; k++)
            y[k] /= nrm;
    }
}

/* Eigenvector zr + i zi of the Hermitian matrix ar + i ai (destroyed) for
   its eigenvalue of rank index (0 smallest, N - 1 largest). */
void hermitian_eigenvector(float *ar, float *ai, int index, float *zr,
                           float *zi)
{
    float d[N], e[N], taur[N], taui[N], u0[N], u1[N], u2[N], l[N];
    int piv[N], k, i;
    float lambda, norm;

    tridiagonalize(ar, ai, d, e, taur, taui, u0, u1);
    lambda = tridiagonal_eigenvalue(d, e, index, &norm);
    tridiagonal_eigenvector(d, e, lambda, norm, zr, u0, u1, u2, l, piv);
    for (k = 0; k < N; k++)
        zi[k] = 0.0f;
    /* z = H_0 ... H_{N-2} y, H = I - tau v v^H. */
    for (k = N - 2; k >= 0; k--) {
        float sr = 0.0f, si = 0.0f, hr, hi;

        if (taur[k] == 0.0f && taui[k] == 0.0f)
            continue;
        for (i = k + 1; i < N; i++) {
            /* v^H z */
            sr += ar[i * N + k] * zr[i] + ai[i * N + k] * zi[i];
            si += ar[i * N + k] * zi[i] - ai[i * N + k] * zr[i];
        }
        hr = taur[k] * sr - taui[k] * si;
        hi = taur[k] * si + taui[k] * sr;
        for (i = k + 1; i < N; i++) {
            zr[i] -= ar[i * N + k] * hr - ai[i * N + k] * hi;
            zi[i] -= ar[i * N + k] * hi + ai[i * N + k] * hr;
        }
    }
}

/* Inverse of the symmetric positive definite g (N x N) into inv by
   Cholesky, using work (N x N). Return 0 when a pivot falls below
   CHOL_FLOOR, i.e. g is not safely positive definite. */
int cholesky_inverse(const float *g, float *inv, float *work)
{
    int i, j, k;

    /* g = L L^T, L lower in work. */
    for (j = 0; j < N; j++) {
        float s = g[j * N + j];

        for (k = 0; k < j; k++)
            s -= work[j * N + k] * work[j * N + k];
        if (!(s > CHOL_FLOOR))
            return 0;
        work[j * N + j] = sqrt(s);
        for (i = j + 1; i < N; i++) {
            float t = g[i * N + j];

            for (k = 0; k < j; k++)
                t -= work[i * N + k] * work[j * N + k];
            work[i * N + j] = t / work[j * N + j];
        }
    }
    /* L^-1 in place (lower triangle). */
    for (j = 0; j < N; j++) {
        work[j * N + j] = 1.0f / work[j * N + j];
        for (i = j + 1; i < N; i++) {
            float t = 0.0f;

            for (k = j; k < i; k++)
                t -= work[i * N + k] * work[k * N + j];
            work[i * N + j] = t / work[i * N + i];
        }
    }
    /* g^-1 = L^-T L^-1. */
    for (i = 0; i < N; i++)
        for (j = i; j < N; j++) {
            float t = 0.0f;

            for (k = j; k < N; k++)
                t += work[k * N + i] * work[k * N + j];
            inv[i * N + j] = t;
            inv[j * N + i] = t;
        }
    return 1;
}

/* Whether the graph of the non-zero coherences of T links all dates. */
int connected(const float *tr, const float *ti)
{
    int seen[N], stack[N], top = 0, reached = 1, i, j;

    for (i = 0; i < N; i++)
        seen[i] = 0;
    seen[0] = 1;
    stack[top++] = 0;
    while (top > 0) {
        i = stack[--top];
        for (j = 0; j < N; j++)
            if (!seen[j] && (tr[i * N + j] != 0.0f || ti[i * N + j] != 0.0f)) {
                seen[j] = 1;
                reached++;
                stack[top++] = j;
            }
    }
    return reached == N;
}

__kernel void temporal_coherence(__global const float2 *slc,
                                 __global const float *amp,
                                 __global const uchar *valid,
                                 __global float *coh, __global int *count,
                                 const int rows, const int cols)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    float tr[N * N], ti[N * N], ar[N * N], ai[N * N], vr[N * N], vi[N * N];
    float zr[N], zi[N];
    __global const float *a;
    float mean_a = 0.0f, var_a = 0.0f, sr, si, lmin;
    int centre, looks = 0, dy, dx, i, j, k;

    if (x >= cols || y >= rows)
        return;
    centre = (y + WA / 2) * W + x + WR / 2;
    if (!valid[centre]) {
        coh[y * cols + x] = NAN;
        count[y * cols + x] = -1;
        return;
    }
    a = amp + (size_t)centre * N;
    if (SHP_TEST == 2) {
        for (k = 0; k < N; k++)
            mean_a += log(a[k]);
        mean_a /= N;
        for (k = 0; k < N; k++) {
            float d = log(a[k]) - mean_a;

            var_a += d * d;
        }
        var_a /= N - 1;
    }

    /* Sample covariance of the SHPs, upper triangle and diagonal. */
    for (i = 0; i < N * N; i++) {
        tr[i] = 0.0f;
        ti[i] = 0.0f;
    }
    for (dy = 0; dy < WA; dy++) {
        for (dx = 0; dx < WR; dx++) {
            const int cand = (y + dy) * W + x + dx;
            __global const float *b = amp + (size_t)cand * N;
            __global const float2 *s = slc + (size_t)cand * N;
            int ok;

            if (!valid[cand])
                continue;
            if (SHP_TEST == 0)
                ok = ks_accept(a, b);
            else if (SHP_TEST == 1)
                ok = ad_accept(a, b);
            else
                ok = tlog_accept(mean_a, var_a, b);
            if (!ok)
                continue;
            looks++;
            for (i = 0; i < N; i++) {
                const float2 si2 = s[i];

                tr[i * N + i] += si2.x * si2.x + si2.y * si2.y;
                for (j = i + 1; j < N; j++) {
                    const float2 sj = s[j];

                    tr[i * N + j] += si2.x * sj.x + si2.y * sj.y;
                    ti[i * N + j] += si2.y * sj.x - si2.x * sj.y;
                }
            }
        }
    }
    count[y * cols + x] = looks;
    if (looks < MIN_LOOKS) {
        coh[y * cols + x] = NAN;
        return;
    }

    /* Coherence matrix, optionally with debiased magnitudes. */
    for (i = 0; i < N; i++)
        zr[i] = sqrt(tr[i * N + i]);
    for (i = 0; i < N; i++) {
        for (j = i + 1; j < N; j++) {
            float d = zr[i] * zr[j], re = 0.0f, im = 0.0f;

            if (d > 0.0f) {
                re = tr[i * N + j] / d;
                im = ti[i * N + j] / d;
                if (BIAS && looks > 1) {
                    float m2 = re * re + im * im;
                    float m2c = (looks * m2 - 1.0f) / (looks - 1.0f);
                    float f = (m2 > 0.0f && m2c > 0.0f) ? sqrt(m2c / m2) : 0.0f;

                    re *= f;
                    im *= f;
                }
            }
            tr[i * N + j] = re;
            ti[i * N + j] = im;
            tr[j * N + i] = re;
            ti[j * N + i] = -im;
        }
        tr[i * N + i] = 1.0f;
        ti[i * N + i] = 0.0f;
    }

    /* With the bias correction, coherences can be zeroed until the dates
       split into groups without coherence between them: the relative phase
       of the groups, hence the temporal coherence, is then undefined. */
    if (BIAS && !connected(tr, ti)) {
        coh[y * cols + x] = NAN;
        return;
    }

    if (!EMI) {
        /* EVD: eigenvector of the largest eigenvalue of T. */
        for (i = 0; i < N * N; i++) {
            ar[i] = tr[i];
            ai[i] = ti[i];
        }
        hermitian_eigenvector(ar, ai, N - 1, zr, zi);
    }
    else {
        /* EMI: eigenvector of the smallest eigenvalue of |T|^-1 o T. |T| is
           inverted by Cholesky, or, when it is close to singular, by the
           pseudo-inverse of its eigendecomposition. */
        for (i = 0; i < N * N; i++)
            vr[i] = hypot(tr[i], ti[i]);
        if (!cholesky_inverse(vr, ar, vi)) {
            for (i = 0; i < N * N; i++) {
                ar[i] = vr[i];
                ai[i] = 0.0f;
            }
            hermitian_eigen(ar, ai, vr, vi);
            lmin = 0.0f;
            for (k = 0; k < N; k++) {
                zr[k] = ar[k * N + k];
                if (fabs(zr[k]) > lmin)
                    lmin = fabs(zr[k]);
            }
            lmin *= EIG_FLOOR;
            for (k = 0; k < N; k++)
                zi[k] = fabs(zr[k]) < lmin ? 0.0f : 1.0f / zr[k];
            for (i = 0; i < N; i++)
                for (j = i; j < N; j++) {
                    float g = 0.0f;

                    for (k = 0; k < N; k++)
                        g += vr[i * N + k] * vr[j * N + k] * zi[k];
                    ar[i * N + j] = g;
                    ar[j * N + i] = g;
                }
        }
        /* M = |T|^-1 o T, Hermitian. */
        for (i = 0; i < N * N; i++) {
            ai[i] = ar[i] * ti[i];
            ar[i] = ar[i] * tr[i];
        }
        hermitian_eigenvector(ar, ai, 0, zr, zi);
    }

    /* Unit phasors exp(i phi) of the linked phases, relative to the
       chronological median date as in SNAP (or to the strongest component
       when that date is negligible). A negligible component has no phase of
       its own and gets the phase of that datum. */
    {
        int ref = N / 2;
        float refr, refi, refm;

        lmin = 0.0f;
        for (k = 0; k < N; k++)
            lmin = fmax(lmin, hypot(zr[k], zi[k]));
        if (hypot(zr[ref], zi[ref]) <= PHASOR_FLOOR * lmin)
            for (k = 0; k < N; k++)
                if (hypot(zr[k], zi[k]) == lmin)
                    ref = k;
        refm = hypot(zr[ref], zi[ref]);
        refr = zr[ref] / refm;
        refi = -zi[ref] / refm;
        lmin *= PHASOR_FLOOR;
        for (k = 0; k < N; k++) {
            /* z_k conj(z_ref) / |z_ref| */
            float re = zr[k] * refr - zi[k] * refi;
            float im = zr[k] * refi + zi[k] * refr;
            float m = hypot(re, im);

            zr[k] = m > lmin ? re / m : 1.0f;
            zi[k] = m > lmin ? im / m : 0.0f;
        }
    }

    /* Goodness of fit: |sum_{i<j} unit(T_ij) conj(z_i) z_j| / pairs, with
       phase 0 for a zero coherence. */
    sr = 0.0f;
    si = 0.0f;
    for (i = 0; i < N; i++) {
        for (j = i + 1; j < N; j++) {
            float re = tr[i * N + j], im = ti[i * N + j];
            float m = hypot(re, im), ur, ui, wr, wi;

            ur = m > 0.0f ? re / m : 1.0f;
            ui = m > 0.0f ? im / m : 0.0f;
            /* w = conj(z_i) z_j */
            wr = zr[i] * zr[j] + zi[i] * zi[j];
            wi = zr[i] * zi[j] - zi[i] * zr[j];
            sr += ur * wr - ui * wi;
            si += ur * wi + ui * wr;
        }
    }
    coh[y * cols + x] = hypot(sr, si) / (0.5f * N * (N - 1));
}
