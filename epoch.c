/* Input dates: raster pairs, r.in.s1slc metadata and stack checks. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <grass/datetime.h>
#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/raster.h>

#include "local_proto.h"

/* Days since 1970-01-01 of a proleptic Gregorian date (H. Hinnant). */
static long days_from_civil(int y, int m, int d)
{
    long era, yoe, doy, doe;

    y -= m <= 2;
    era = (y >= 0 ? y : y - 399) / 400;
    yoe = y - era * 400;
    doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;
}

double utc_seconds(const struct utc *t)
{
    return days_from_civil(t->year, t->month, t->day) * 86400.0 +
           t->hour * 3600.0 + t->minute * 60.0 + t->second;
}

void utc_from_seconds(double sec, struct utc *t)
{
    long z = (long)floor(sec / 86400.0), era, doe, yoe, doy, mp;
    double rest = sec - z * 86400.0;

    z += 719468;
    era = (z >= 0 ? z : z - 146096) / 146097;
    doe = z - era * 146097;
    yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    mp = (5 * doy + 2) / 153;
    t->day = doy - (153 * mp + 2) / 5 + 1;
    t->month = mp < 10 ? mp + 3 : mp - 9;
    t->year = yoe + era * 400 + (t->month <= 2);
    t->hour = (int)(rest / 3600.0);
    t->minute = (int)((rest - t->hour * 3600.0) / 60.0);
    t->second = rest - t->hour * 3600.0 - t->minute * 60.0;
}

/* Parse an ISO time such as 2023-01-12T14:25:09.020000. */
static int parse_iso(const char *text, struct utc *t)
{
    return text && sscanf(text, "%d-%d-%dT%d:%d:%lf", &t->year, &t->month,
                          &t->day, &t->hour, &t->minute, &t->second) == 6;
}

static void from_datetime(const DateTime *dt, struct utc *t)
{
    t->year = dt->year;
    t->month = dt->month;
    t->day = dt->day;
    t->hour = dt->hour;
    t->minute = dt->minute;
    t->second = dt->second;
}

static void read_time_span(struct epoch *e)
{
    char *first = NULL, *last = NULL;
    struct TimeStamp ts;

    e->has_time = 0;
    if (e->meta) {
        first = json_get(e->meta, "raster_geometry.first_line_time");
        last = json_get(e->meta, "raster_geometry.last_line_time");
    }
    if (parse_iso(first, &e->start)) {
        if (!parse_iso(last, &e->end))
            e->end = e->start;
        e->has_time = 1;
    }
    else if (G_read_raster_timestamp(e->name[0], e->mapset[0], &ts) == 1) {
        DateTime dt1, dt2;
        int count;

        G_get_timestamps(&ts, &dt1, &dt2, &count);
        if (count >= 1 && datetime_is_absolute(&dt1) &&
            dt1.to >= DATETIME_SECOND) {
            from_datetime(&dt1, &e->start);
            if (count == 2 && dt2.to >= DATETIME_SECOND)
                from_datetime(&dt2, &e->end);
            else
                e->end = e->start;
            e->has_time = 1;
        }
    }
    if (e->has_time)
        e->start_sec = utc_seconds(&e->start);
    G_free(first);
    G_free(last);
}

void epoch_load(struct epoch *e, const char *basename)
{
    char path[GPATH_MAX];
    int k;

    if (strlen(basename) + 3 > GNAME_MAX)
        G_fatal_error(_("Input name <%s> is too long"), basename);
    G_strlcpy(e->basename, basename, sizeof(e->basename));
    for (k = 0; k < 2; k++) {
        const char *mapset;

        snprintf(e->name[k], GNAME_MAX, "%s_%c", basename, k ? 'q' : 'i');
        mapset = G_find_raster2(e->name[k], "");
        if (!mapset)
            G_fatal_error(_("Raster map <%s> not found (input <%s> must be "
                            "the basename of an r.in.s1slc complex pair)"),
                          e->name[k], basename);
        G_strlcpy(e->mapset[k], mapset, GMAPSET_MAX);
    }
    Rast_get_cellhd(e->name[0], e->mapset[0], &e->head);

    G_file_name_misc(path, "cell_misc", "description.json", e->name[0],
                     e->mapset[0]);
    e->meta = read_text_file(path);
    read_time_span(e);
}

/* Metadata value at a dotted path, NULL if unknown. */
char *epoch_attribute(const struct epoch *e, const char *path)
{
    return e->meta ? json_get(e->meta, path) : NULL;
}

static int same_grid(const struct Cell_head *a, const struct Cell_head *b)
{
    return a->rows == b->rows && a->cols == b->cols &&
           fabs(a->north - b->north) < 1e-9 * a->ns_res &&
           fabs(a->south - b->south) < 1e-9 * a->ns_res &&
           fabs(a->east - b->east) < 1e-9 * a->ew_res &&
           fabs(a->west - b->west) < 1e-9 * a->ew_res;
}

static int same_value(const char *a, const char *b)
{
    if (!a || !b)
        return a == b;
    return !strcmp(a, b);
}

void check_stack(struct epoch *epochs, int n)
{
    const char *keys[][2] = {
        {"swath.swath", _("sub-swath")},
        {"swath.polarization", _("polarization")},
        {"swath.pass", _("pass")},
        {"product.relative_orbit_start", _("relative orbit")},
        {"calibration", _("calibration")},
    };
    int i, j, k, missing = 0, undated = 0;

    if (n < 3)
        G_fatal_error(_("Temporal coherence needs at least 3 dates, got %d"),
                      n);
    if (n > MAX_DATES)
        G_fatal_error(_("At most %d dates are supported, got %d"), MAX_DATES,
                      n);
    for (i = 0; i < n; i++)
        for (j = i + 1; j < n; j++)
            if (!strcmp(epochs[i].basename, epochs[j].basename))
                G_fatal_error(_("Input <%s> is given more than once"),
                              epochs[i].basename);

    for (i = 0; i < n; i++) {
        for (k = 0; k < 2; k++) {
            struct Cell_head head;

            Rast_get_cellhd(epochs[i].name[k], epochs[i].mapset[k], &head);
            if (!same_grid(&head, &epochs[0].head))
                G_fatal_error(
                    _("Raster map <%s> is not on the grid of <%s>: inputs "
                      "must be coregistered to a common grid (rows, columns "
                      "and extent)"),
                    epochs[i].name[k], epochs[0].name[0]);
        }
        missing += epochs[i].meta == NULL;
        undated += !epochs[i].has_time;
    }

    if (missing)
        G_warning(_("No r.in.s1slc metadata for %d of the %d inputs: stack "
                    "consistency not checked for them"),
                  missing, n);
    for (k = 0; k < (int)(sizeof(keys) / sizeof(keys[0])); k++) {
        char *ref = NULL;
        int have_ref = 0;

        for (i = 0; i < n; i++) {
            char *value;

            if (!epochs[i].meta)
                continue;
            value = json_get(epochs[i].meta, keys[k][0]);
            if (!have_ref) {
                ref = value;
                have_ref = i + 1;
                continue;
            }
            if (!same_value(ref, value))
                G_fatal_error(_("Inputs differ in %s: <%s> has %s, <%s> has "
                                "%s"),
                              keys[k][1], epochs[have_ref - 1].basename,
                              ref ? ref : "none", epochs[i].basename,
                              value ? value : "none");
            G_free(value);
        }
        G_free(ref);
    }
    for (i = 0; i < n; i++) {
        char *measure = epoch_attribute(&epochs[i], "measure");

        if (measure && strcmp(measure, "i") != 0)
            G_fatal_error(_("Raster map <%s> is not the in-phase component "
                            "of a complex image"),
                          epochs[i].name[0]);
        G_free(measure);
    }

    if (undated)
        G_warning(_("Some inputs have no acquisition time; dates are not "
                    "checked"));
    for (i = 0; i < n; i++)
        for (j = i + 1; j < n; j++)
            if (epochs[i].has_time && epochs[j].has_time &&
                epochs[i].start_sec == epochs[j].start_sec)
                G_fatal_error(_("Inputs <%s> and <%s> have the same "
                                "acquisition time"),
                              epochs[i].basename, epochs[j].basename);
}

void check_region(const struct epoch *e, const struct Cell_head *region)
{
    const struct Cell_head *h = &e->head;
    double shift;

    if (fabs(region->ns_res - h->ns_res) > 1e-9 * h->ns_res ||
        fabs(region->ew_res - h->ew_res) > 1e-9 * h->ew_res)
        G_fatal_error(_("Current region resolution %g x %g differs from the "
                        "input resolution %g x %g; the SHP window is counted "
                        "in pixels. Run: g.region raster=%s"),
                      region->ns_res, region->ew_res, h->ns_res, h->ew_res,
                      e->name[0]);
    shift = (region->north - h->north) / h->ns_res;
    if (fabs(shift - floor(shift + 0.5)) > 1e-6)
        G_fatal_error(_("Current region is not aligned with the input grid. "
                        "Run: g.region align=%s"),
                      e->name[0]);
    shift = (region->west - h->west) / h->ew_res;
    if (fabs(shift - floor(shift + 0.5)) > 1e-6)
        G_fatal_error(_("Current region is not aligned with the input grid. "
                        "Run: g.region align=%s"),
                      e->name[0]);
}
