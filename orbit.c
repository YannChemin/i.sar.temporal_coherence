/* Satellite orbits: annotation state vectors or Sentinel-1 precise and
 * restituted orbit files (POEORB, RESORB), read locally or downloaded from
 * the ESA STEP mirror used by SNAP, and Lagrange interpolation. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cpl_string.h>
#include <cpl_vsi.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "local_proto.h"

#define STEP_ORBITS_URL "http://step.esa.int/auxdata/orbits/Sentinel-1/"
/* State vectors kept around the acquisition, seconds. */
#define ORBIT_MARGIN    120.0
/* Points of the Lagrange interpolation. */
#define LAGRANGE_POINTS 8

static void orbit_alloc(struct orbit *o, int n)
{
    o->n = 0;
    o->t = G_malloc(n * sizeof(double));
    o->pos = G_malloc(n * sizeof(*o->pos));
    o->vel = G_malloc(n * sizeof(*o->vel));
}

static void orbit_add(struct orbit *o, double t, const double *pos,
                      const double *vel)
{
    int k;

    o->t[o->n] = t;
    for (k = 0; k < 3; k++) {
        o->pos[o->n][k] = pos[k];
        o->vel[o->n][k] = vel[k];
    }
    o->n++;
}

static int parse_utc(const char *text, double *sec)
{
    struct utc t;

    if (sscanf(text, "%d-%d-%dT%d:%d:%lf", &t.year, &t.month, &t.day, &t.hour,
               &t.minute, &t.second) != 6)
        return 0;
    *sec = utc_seconds(&t);
    return 1;
}

/* Acquisition start and stop of the epoch, seconds. */
static void acquisition_span(const struct epoch *e, double *start, double *stop)
{
    char *a = epoch_attribute(e, "swath.start_time");
    char *b = epoch_attribute(e, "swath.stop_time");

    if (!a || !b || !parse_utc(a, start) || !parse_utc(b, stop)) {
        if (!e->has_time)
            G_fatal_error(_("No acquisition time for <%s>"), e->basename);
        *start = utc_seconds(&e->start);
        *stop = utc_seconds(&e->end);
    }
    G_free(a);
    G_free(b);
}

static void orbit_check(const struct orbit *o, const struct epoch *e,
                        double start, double stop)
{
    if (o->n < LAGRANGE_POINTS || o->t[0] > start || o->t[o->n - 1] < stop)
        G_fatal_error(_("The %s orbit of <%s> does not cover its acquisition "
                        "with %d state vectors"),
                      o->source, e->basename, LAGRANGE_POINTS);
}

void orbit_from_annotation(const struct epoch *e, struct orbit *o)
{
    const char *base = "swath.orbit_state_vectors";
    char path[128], *time;
    double start, stop;
    int n, i, k;

    n = e->meta ? json_array_length(e->meta, base) : -1;
    if (n <= 0)
        G_fatal_error(_("No orbit state vectors in the metadata of <%s>"),
                      e->basename);
    acquisition_span(e, &start, &stop);
    orbit_alloc(o, n);
    G_strlcpy(o->source, "annotation", sizeof(o->source));
    for (i = 0; i < n; i++) {
        double t, pos[3], vel[3];

        snprintf(path, sizeof(path), "%s.%d.time", base, i);
        time = epoch_attribute(e, path);
        if (!time || !parse_utc(time, &t))
            G_fatal_error(_("Invalid orbit state vector %d of <%s>"), i,
                          e->basename);
        G_free(time);
        for (k = 0; k < 3; k++) {
            snprintf(path, sizeof(path), "%s.%d.position.%d", base, i, k);
            pos[k] = json_get_number(e->meta, path);
            snprintf(path, sizeof(path), "%s.%d.velocity.%d", base, i, k);
            vel[k] = json_get_number(e->meta, path);
            if (isnan(pos[k]) || isnan(vel[k]))
                G_fatal_error(_("Invalid orbit state vector %d of <%s>"), i,
                              e->basename);
        }
        orbit_add(o, t, pos, vel);
    }
    orbit_check(o, e, start, stop);
}

/* Validity V<start>_<stop> of an orbit file name, seconds. */
static int file_validity(const char *name, double *v0, double *v1)
{
    const char *p = strstr(name, "_V");
    struct utc a, b;

    if (!p || sscanf(p, "_V%4d%2d%2dT%2d%2d%lf_%4d%2d%2dT%2d%2d%lf", &a.year,
                     &a.month, &a.day, &a.hour, &a.minute, &a.second, &b.year,
                     &b.month, &b.day, &b.hour, &b.minute, &b.second) != 12)
        return 0;
    *v0 = utc_seconds(&a);
    *v1 = utc_seconds(&b);
    return 1;
}

/* Best file of the list covering [start, stop]: the latest production. */
static char *pick_file(char **names, const char *prefix, double start,
                       double stop, int zipped)
{
    char *best = NULL;
    int i;

    for (i = 0; names && names[i]; i++) {
        const char *name = names[i];
        size_t len = strlen(name);
        double v0, v1;

        if (strncmp(name, prefix, strlen(prefix)) != 0)
            continue;
        if (zipped ? (len < 8 || strcmp(name + len - 8, ".EOF.zip") != 0)
                   : (len < 4 || strcmp(name + len - 4, ".EOF") != 0))
            continue;
        if (!file_validity(name, &v0, &v1) || v0 > start || v1 < stop)
            continue;
        if (!best || strcmp(name, best) > 0)
            best = (char *)name;
    }
    return best ? G_store(best) : NULL;
}

static char *read_vsi(const char *path)
{
    VSILFILE *fp = VSIFOpenL(path, "rb");
    vsi_l_offset size;
    char *text;

    if (!fp)
        return NULL;
    VSIFSeekL(fp, 0, SEEK_END);
    size = VSIFTellL(fp);
    VSIFSeekL(fp, 0, SEEK_SET);
    text = G_malloc(size + 1);
    if (VSIFReadL(text, 1, size, fp) != size) {
        VSIFCloseL(fp);
        G_free(text);
        return NULL;
    }
    text[size] = '\0';
    VSIFCloseL(fp);
    return text;
}

/* Parse the state vectors of an EOF file within [t0, t1]. */
static void parse_eof(const char *text, double t0, double t1, struct orbit *o)
{
    static const char *tags[6] = {"<X", "<Y", "<Z", "<VX", "<VY", "<VZ"};
    const char *p = text;
    int n = 0;

    while ((p = strstr(p, "<OSV>")))
        n++, p += 5;
    orbit_alloc(o, n);
    for (p = strstr(text, "<OSV>"); p; p = strstr(p + 5, "<OSV>")) {
        const char *end = strstr(p, "</OSV>"), *u = strstr(p, "<UTC>UTC=");
        double t, v[6];
        int k;

        if (!end || !u || u > end || !parse_utc(u + 9, &t))
            continue;
        if (t < t0 || t > t1)
            continue;
        for (k = 0; k < 6; k++) {
            const char *q = strstr(p, tags[k]);

            /* "<X" also matches "<X unit"; skip to the value after '>'. */
            q = q && q < end ? strchr(q, '>') : NULL;
            if (!q || q > end || sscanf(q + 1, "%lf", &v[k]) != 1)
                G_fatal_error(_("Invalid state vector in orbit file"));
        }
        orbit_add(o, t, v, v + 3);
    }
}

void orbit_from_file(const struct epoch *e, const char *kind,
                     const char *cache_dir, struct orbit *o)
{
    char *mission = epoch_attribute(e, "swath.mission"), prefix[128], *name,
         path[GPATH_MAX], *text = NULL;
    double start, stop;
    char **names;

    if (!mission)
        G_fatal_error(_("No mission in the metadata of <%s>"), e->basename);
    acquisition_span(e, &start, &stop);
    snprintf(prefix, sizeof(prefix), "%s_OPER_AUX_%s_OPOD_", mission, kind);

    /* Local file first. */
    names = VSIReadDir(cache_dir);
    name =
        pick_file(names, prefix, start - ORBIT_MARGIN, stop + ORBIT_MARGIN, 0);
    CSLDestroy(names);
    if (name) {
        snprintf(path, sizeof(path), "%s/%s", cache_dir, name);
        text = read_vsi(path);
        G_verbose_message(_("Orbit of <%s>: %s"), e->basename, path);
    }
    else {
        /* The ESA STEP mirror files orbits by the month of their validity
           start, the day before the acquisition. */
        struct utc months[2];
        int m;

        utc_from_seconds(start - 86400.0, &months[0]);
        utc_from_seconds(start, &months[1]);
        for (m = 0; m < 2 && !text; m++) {
            char url[1024];

            snprintf(url, sizeof(url), "/vsicurl/%s%s/%s/%04d/%02d/",
                     STEP_ORBITS_URL, kind, mission, months[m].year,
                     months[m].month);
            names = VSIReadDir(url);
            name = pick_file(names, prefix, start - ORBIT_MARGIN,
                             stop + ORBIT_MARGIN, 1);
            CSLDestroy(names);
            if (!name)
                continue;
            G_message(_("Downloading orbit %s..."), name);
            snprintf(path, sizeof(path), "/vsizip/%s%s/%.*s", url, name,
                     (int)(strlen(name) - 4), name);
            text = read_vsi(path);
            if (!text)
                G_fatal_error(_("Unable to download <%s%s>"), url + 9, name);
            /* Cache the unzipped file. */
            if (G_mkdir(cache_dir) != 0 && access(cache_dir, W_OK) != 0)
                G_warning(_("Unable to create the orbit directory <%s>"),
                          cache_dir);
            else {
                FILE *fp;

                snprintf(path, sizeof(path), "%s/%.*s", cache_dir,
                         (int)(strlen(name) - 4), name);
                if ((fp = fopen(path, "w"))) {
                    fputs(text, fp);
                    fclose(fp);
                }
            }
        }
        if (!text)
            G_fatal_error(
                _("No %s orbit file of %s covers <%s> in <%s> nor on %s "
                  "(precise orbits are published about three weeks after "
                  "acquisition; try orbit=restituted or orbit=annotation)"),
                kind, mission, e->basename, cache_dir, STEP_ORBITS_URL);
    }
    snprintf(o->source, sizeof(o->source), "%s", kind);
    parse_eof(text, start - ORBIT_MARGIN, stop + ORBIT_MARGIN, o);
    G_free(text);
    G_free(name);
    G_free(mission);
    orbit_check(o, e, start, stop);
}

/* Position and velocity at time t by Lagrange interpolation of the
   LAGRANGE_POINTS state vectors around t. */
void orbit_state(const struct orbit *o, double t, double *pos, double *vel)
{
    int first = 0, i, j, k;

    while (first + LAGRANGE_POINTS < o->n &&
           o->t[first + LAGRANGE_POINTS / 2] < t)
        first++;
    for (k = 0; k < 3; k++)
        pos[k] = vel[k] = 0.0;
    for (i = first; i < first + LAGRANGE_POINTS; i++) {
        double w = 1.0;

        for (j = first; j < first + LAGRANGE_POINTS; j++)
            if (j != i)
                w *= (t - o->t[j]) / (o->t[i] - o->t[j]);
        for (k = 0; k < 3; k++) {
            pos[k] += w * o->pos[i][k];
            vel[k] += w * o->vel[i][k];
        }
    }
}

void orbit_free(struct orbit *o)
{
    G_free(o->t);
    G_free(o->pos);
    G_free(o->vel);
}
