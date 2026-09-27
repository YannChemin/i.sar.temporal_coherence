/* Critical values and normalizations of the SHP tests. */

#include <math.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "local_proto.h"

/* Upper-tail points of the Scholz-Stephens T_1 distribution (two samples),
   b0 + b1 + b2 of Scholz and Stephens (1987), table 1, as in
   scipy.stats.anderson_ksamp; significance levels decreasing. */
static const double ad_alpha[] = {0.25, 0.10, 0.05, 0.025, 0.01, 0.005, 0.001};
static const double ad_crit[] = {0.325, 1.226, 1.961, 2.718,
                                 3.752, 4.592, 6.546};
#define AD_LEVELS 7

/* Standard deviation of the Anderson-Darling A2 under the null hypothesis
   for k = 2 samples of sizes na, nb (Scholz and Stephens 1987, eq. 4). */
static double ad_sigma(int na, int nb)
{
    const int k = 2, total = na + nb;
    double big_h = 1.0 / na + 1.0 / nb, h = 0.0, g = 0.0, a, b, c, d, var;
    int i, j;

    if (total <= 3)
        return 1.0;
    for (j = 1; j < total; j++)
        h += 1.0 / j;
    for (i = 1; i <= total - 2; i++)
        for (j = i + 1; j <= total - 1; j++)
            g += 1.0 / ((double)(total - i) * j);
    a = (4 * g - 6) * (k - 1) + (10 - 6 * g) * big_h;
    b = (2 * g - 4) * k * k + 8 * h * k + (2 * g - 14 * h - 4) * big_h - 8 * h +
        4 * g - 6;
    c = (6 * h + 2 * g - 2) * k * k + (4 * h - 4 * g + 6) * k +
        (2 * h - 6) * big_h + 4 * h;
    d = (2 * h + 6) * k * k - 4 * h * k;
    var = (a * pow(total, 3) + b * (double)total * total + c * total + d) /
          ((total - 1.0) * (total - 2.0) * (total - 3.0));
    return var > 0 ? sqrt(var) : 1.0;
}

/* Two-sided standard normal quantile z with P(|Z| > z) = alpha. */
static double normal_quantile(double alpha)
{
    double lo = 0.0, hi = 40.0;
    int i;

    for (i = 0; i < 200; i++) {
        double mid = 0.5 * (lo + hi);

        if (erfc(mid / sqrt(2.0)) > alpha)
            lo = mid;
        else
            hi = mid;
    }
    return 0.5 * (lo + hi);
}

void shp_constants(struct settings *s)
{
    const int n = s->ndates;
    int i;

    /* Kolmogorov-Smirnov: D = d / n is accepted up to
       c(alpha) sqrt(2 / n), c(alpha) = sqrt(-ln(alpha / 2) / 2). */
    s->ks_max =
        (int)floor(sqrt(-0.5 * log(s->alpha / 2.0)) * sqrt(2.0 / n) * n);

    /* Anderson-Darling: A2 = 1/N (1/n + 1/n) sum ..., N = 2n. */
    s->ad_norm = 1.0 / (2.0 * n) * (2.0 / n);
    s->ad_sigma = ad_sigma(n, n);
    s->ad_crit = 0.0;
    if (s->test == SHP_AD) {
        if (s->alpha > ad_alpha[0] || s->alpha < ad_alpha[AD_LEVELS - 1])
            G_fatal_error(_("Anderson-Darling test: alpha must be between "
                            "%g and %g"),
                          ad_alpha[AD_LEVELS - 1], ad_alpha[0]);
        /* Linear interpolation in log(alpha). */
        for (i = 0; i < AD_LEVELS - 1; i++)
            if (s->alpha >= ad_alpha[i + 1])
                break;
        if (i == AD_LEVELS - 1)
            i = AD_LEVELS - 2;
        {
            double f = (log(s->alpha) - log(ad_alpha[i])) /
                       (log(ad_alpha[i + 1]) - log(ad_alpha[i]));

            s->ad_crit = ad_crit[i] + f * (ad_crit[i + 1] - ad_crit[i]);
        }
    }

    s->tlog_crit = normal_quantile(s->alpha);
}
