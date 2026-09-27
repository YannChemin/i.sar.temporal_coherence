/* Flat-earth and topographic phase from the orbits (range-Doppler
 * geometry): for a pixel of the reference date at a given height, the
 * ground point P is located on the reference orbit, then every date k sees
 * it at the slant range R_k of its zero-Doppler time. With the SLC phase
 * convention -4 pi R / lambda (as in SNAP), multiplying date k by
 * exp(i 4 pi (R_k - R_ref) / lambda) removes the geometric phase of every
 * pair. It is computed exactly on a coarse grid of the region and
 * interpolated bilinearly in line and sample and quadratically in height. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "local_proto.h"

#define SPEED_OF_LIGHT 299792458.0
#define WGS84_A        6378137.0
#define WGS84_B        6356752.314245
/* Grid node spacing in lines and samples. */
#define GRID_STEP      16

static double dot(const double *a, const double *b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/* Zero-Doppler time of point p on the orbit, starting from its closest
   state vector. */
static double zero_doppler(const struct orbit *o, const double *p)
{
    double t, best = -1.0;
    int i, it;

    t = o->t[0];
    for (i = 0; i < o->n; i++) {
        double d[3], dist;
        int k;

        for (k = 0; k < 3; k++)
            d[k] = p[k] - o->pos[i][k];
        dist = dot(d, d);
        if (best < 0 || dist < best) {
            best = dist;
            t = o->t[i];
        }
    }
    for (it = 0; it < 30; it++) {
        double s[3], v[3], d[3], dt;
        int k;

        orbit_state(o, t, s, v);
        for (k = 0; k < 3; k++)
            d[k] = p[k] - s[k];
        /* f(t) = v . (p - s), f'(t) ~ -|v|^2 (the acceleration term is
           about 1e-3 of it). */
        dt = dot(v, d) / dot(v, v);
        t += dt;
        if (fabs(dt) < 1e-10)
            break;
    }
    if (t < o->t[0] || t > o->t[o->n - 1])
        G_fatal_error(_("A ground point is seen outside of the %s orbit "
                        "coverage: is the stack coregistered on option "
                        "reference?"),
                      o->source);
    return t;
}

/* Solve the 3 x 3 system a x = b by Cramer's rule. */
static int solve3(double a[3][3], const double *b, double *x)
{
    double det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
                 a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
                 a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    int c, r;

    if (det == 0.0)
        return 0;
    for (c = 0; c < 3; c++) {
        double m[3][3];

        memcpy(m, a, sizeof(m));
        for (r = 0; r < 3; r++)
            m[r][c] = b[r];
        x[c] = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])) /
               det;
    }
    return 1;
}

/* Ground point p at height h seen at zero-Doppler time t and slant range
   range from the orbit, right-looking. p holds the initial guess when
   have_guess is set. */
static void geolocate(const struct orbit *o, double t, double range, double h,
                      int have_guess, double *p)
{
    const double a = WGS84_A + h, b = WGS84_B + h;
    double s[3], v[3];
    int it, k;

    orbit_state(o, t, s, v);
    if (!have_guess) {
        /* Nadir point, moved to the right of the track by the ground
           range of a flat earth. */
        double rs = sqrt(dot(s, s)), right[3], nr, ground;

        right[0] = v[1] * s[2] - v[2] * s[1];
        right[1] = v[2] * s[0] - v[0] * s[2];
        right[2] = v[0] * s[1] - v[1] * s[0];
        nr = sqrt(dot(right, right));
        ground = sqrt(fmax(range * range - (rs - a) * (rs - a), 0.0));
        for (k = 0; k < 3; k++)
            p[k] = s[k] * a / rs + right[k] / nr * ground;
    }
    for (it = 0; it < 50; it++) {
        double d[3], jac[3][3], f[3], dx[3], dist;

        for (k = 0; k < 3; k++)
            d[k] = p[k] - s[k];
        dist = sqrt(dot(d, d));
        f[0] = -dot(v, d);
        f[1] = -(dist - range);
        f[2] = -((p[0] * p[0] + p[1] * p[1]) / (a * a) + p[2] * p[2] / (b * b) -
                 1.0);
        for (k = 0; k < 3; k++) {
            jac[0][k] = v[k];
            jac[1][k] = d[k] / dist;
        }
        jac[2][0] = 2.0 * p[0] / (a * a);
        jac[2][1] = 2.0 * p[1] / (a * a);
        jac[2][2] = 2.0 * p[2] / (b * b);
        if (!solve3(jac, f, dx))
            G_fatal_error(_("Geolocation failed (singular geometry)"));
        for (k = 0; k < 3; k++)
            p[k] += dx[k];
        if (sqrt(dot(dx, dx)) < 1e-6)
            return;
    }
    G_fatal_error(_("Geolocation did not converge at azimuth time %.6f, "
                    "slant range %.3f m"),
                  t, range);
}

static double required_number(const struct epoch *e, const char *path)
{
    double x = e->meta ? json_get_number(e->meta, path) : NAN;

    if (isnan(x))
        G_fatal_error(_("No <%s> in the metadata of <%s>, needed for the "
                        "flat-earth and topographic phase"),
                      path, e->basename);
    return x;
}

void geometry_init(struct geometry *g, const struct epoch *epochs, int n,
                   int ref, const struct orbit *orbits,
                   const struct Cell_head *region, double hmin, double hmax,
                   double h0)
{
    const struct epoch *e = &epochs[ref];
    char *first = epoch_attribute(e, "raster_geometry.first_line_time");
    const double dt =
        required_number(e, "raster_geometry.azimuth_time_interval");
    const double tau0 =
        required_number(e, "raster_geometry.slant_range_time_first_sample");
    const double fs = required_number(e, "swath.range_sampling_rate");
    const double lambda = required_number(e, "swath.wavelength");
    double t_first, line0, sample0;
    struct utc u;
    int gr, gc, hl, k;

    if (!first || sscanf(first, "%d-%d-%dT%d:%d:%lf", &u.year, &u.month, &u.day,
                         &u.hour, &u.minute, &u.second) != 6)
        G_fatal_error(_("No raster_geometry.first_line_time in the metadata "
                        "of <%s>"),
                      e->basename);
    G_free(first);
    t_first = utc_seconds(&u);
    for (k = 0; k < n; k++) {
        double lk = required_number(&epochs[k], "swath.wavelength");

        if (fabs(lk - lambda) > 1e-9 * lambda)
            G_fatal_error(_("Inputs <%s> and <%s> have different radar "
                            "wavelengths"),
                          epochs[k].basename, e->basename);
    }

    /* Line and sample of the first region cell in the reference image,
       which spans 0..cols east and 0..rows north. */
    line0 = (e->head.north - region->north) / region->ns_res;
    sample0 = (region->west - e->head.west) / region->ew_res;

    g->ndates = n;
    g->step = GRID_STEP;
    g->grid_rows = (region->rows - 1) / GRID_STEP + 2;
    g->grid_cols = (region->cols - 1) / GRID_STEP + 2;
    g->h0 = h0;
    if (hmax - hmin < 1.0) {
        g->nheights = 1;
        g->heights[0] = 0.5 * (hmin + hmax);
    }
    else {
        g->nheights = 3;
        g->heights[0] = hmin;
        g->heights[1] = 0.5 * (hmin + hmax);
        g->heights[2] = hmax;
    }
    g->phase = G_malloc((size_t)g->grid_rows * g->grid_cols * g->nheights * n *
                        sizeof(double));

    G_message(_("Computing the flat-earth%s phase of %d dates from the %s "
                "orbits..."),
              g->nheights > 1 ? _(" and topographic") : "", n,
              orbits[0].source);
    for (gr = 0; gr < g->grid_rows; gr++) {
        const double line = line0 + (double)gr * GRID_STEP;
        const double t = t_first + line * dt;

        G_percent(gr, g->grid_rows, 5);
        for (gc = 0; gc < g->grid_cols; gc++) {
            const double sample = sample0 + (double)gc * GRID_STEP;
            const double range = 0.5 * SPEED_OF_LIGHT * (tau0 + sample / fs);
            double p[3];

            for (hl = 0; hl < g->nheights; hl++) {
                double *phase =
                    g->phase +
                    (((size_t)gr * g->grid_cols + gc) * g->nheights + hl) * n;
                double r_ref = 0.0;

                geolocate(&orbits[ref], t, range, g->heights[hl], hl > 0, p);
                for (k = 0; k < n; k++) {
                    double s[3], v[3], d[3];
                    double tk = zero_doppler(&orbits[k], p);
                    int c;

                    orbit_state(&orbits[k], tk, s, v);
                    for (c = 0; c < 3; c++)
                        d[c] = p[c] - s[c];
                    phase[k] = sqrt(dot(d, d));
                    if (k == ref)
                        r_ref = phase[k];
                }
                for (k = 0; k < n; k++)
                    phase[k] = 4.0 * M_PI / lambda * (phase[k] - r_ref);
            }
        }
    }
    G_percent(1, 1, 1);
}

/* Phase to add to every date at region cell (row, col) and height h. */
void geometry_phases(const struct geometry *g, int row, int col, double h,
                     double *phase)
{
    const int n = g->ndates;
    const int gr = row / g->step, gc = col / g->step;
    const double fr = (double)(row - gr * g->step) / g->step;
    const double fc = (double)(col - gc * g->step) / g->step;
    double wh[3] = {1.0, 0.0, 0.0};
    int hl, k;

    if (g->nheights == 3) {
        /* Quadratic Lagrange weights of h over the three levels. */
        const double *x = g->heights;

        wh[0] = (h - x[1]) * (h - x[2]) / ((x[0] - x[1]) * (x[0] - x[2]));
        wh[1] = (h - x[0]) * (h - x[2]) / ((x[1] - x[0]) * (x[1] - x[2]));
        wh[2] = (h - x[0]) * (h - x[1]) / ((x[2] - x[0]) * (x[2] - x[1]));
    }
    for (k = 0; k < n; k++)
        phase[k] = 0.0;
    for (hl = 0; hl < g->nheights; hl++) {
        const size_t stride = (size_t)g->nheights * n;
        const double *p00 =
            g->phase +
            (((size_t)gr * g->grid_cols + gc) * g->nheights + hl) * n;
        const double *p01 = p00 + stride;
        const double *p10 = p00 + (size_t)g->grid_cols * stride;
        const double *p11 = p10 + stride;

        for (k = 0; k < n; k++)
            phase[k] +=
                wh[hl] * ((1.0 - fr) * ((1.0 - fc) * p00[k] + fc * p01[k]) +
                          fr * ((1.0 - fc) * p10[k] + fc * p11[k]));
    }
}

void geometry_free(struct geometry *g)
{
    G_free(g->phase);
}
