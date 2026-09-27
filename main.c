/****************************************************************************
 *
 * MODULE:    i.sar.temporal_coherence
 * AUTHOR(S): Yann Chemin
 * PURPOSE:   Computes the phase-linking temporal coherence (Pepe-Lanari
 *            goodness of fit) of a coregistered stack of SAR SLC images,
 *            with statistically homogeneous pixel selection and EVD or EMI
 *            phase estimation, on an OpenCL device.
 * COPYRIGHT: (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 ****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <grass/datetime.h>
#include <grass/gis.h>
#include <grass/gjson.h>
#include <grass/glocale.h>
#include <grass/raster.h>

#include "local_proto.h"

/* Strip buffers on the host, row-major over the padded strip. */
struct strip {
    int padded_rows, padded_cols;
    float *slc;           /* N complex samples per pixel */
    float *amp;           /* N amplitudes per pixel, sorted increasingly */
    unsigned char *valid; /* every date non-null and non-zero */
};

static int compare_epochs(const void *pa, const void *pb)
{
    const struct epoch *a = pa, *b = pb;

    if (a->has_time != b->has_time)
        return b->has_time - a->has_time;
    if (a->has_time && a->start_sec != b->start_sec)
        return a->start_sec < b->start_sec ? -1 : 1;
    return strcmp(a->basename, b->basename);
}

/* Read region rows row0..row0+strip->padded_rows-1 (possibly outside the
   region, then invalid) into the padded strip, halo_cols invalid columns on
   each side. With geo, the flat-earth and topographic phase is removed from
   every date, at the height of the elevation map (fd_elev >= 0; a null
   height makes the pixel invalid) or at the constant height geo->h0. */
static void read_strip(struct strip *st, int (*fd)[2], int n, int row0,
                       int nrows, int ncols, int halo_cols, FCELL *buf_i,
                       FCELL *buf_q, const struct geometry *geo, int fd_elev,
                       DCELL *buf_h, double *phase)
{
    const size_t npad = (size_t)st->padded_rows * st->padded_cols;
    int r, c, k;

    memset(st->valid, 0, npad);
    memset(st->slc, 0, npad * n * 2 * sizeof(float));
    memset(st->amp, 0, npad * n * sizeof(float));
    for (r = 0; r < st->padded_rows; r++) {
        const int row = row0 + r;
        size_t base = (size_t)r * st->padded_cols + halo_cols;

        if (row < 0 || row >= nrows)
            continue;
        memset(st->valid + base, 1, ncols);
        for (k = 0; k < n; k++) {
            Rast_get_f_row(fd[k][0], buf_i, row);
            Rast_get_f_row(fd[k][1], buf_q, row);
            for (c = 0; c < ncols; c++) {
                size_t pix = base + c;
                float *s = st->slc + (pix * n + k) * 2;

                if (Rast_is_f_null_value(&buf_i[c]) ||
                    Rast_is_f_null_value(&buf_q[c]) ||
                    (buf_i[c] == 0.0f && buf_q[c] == 0.0f)) {
                    st->valid[pix] = 0;
                    continue;
                }
                s[0] = buf_i[c];
                s[1] = buf_q[c];
            }
        }
        if (fd_elev >= 0)
            Rast_get_d_row(fd_elev, buf_h, row);
        for (c = 0; c < ncols; c++) {
            size_t pix = base + c;
            float *a = st->amp + pix * n;

            if (fd_elev >= 0 && Rast_is_d_null_value(&buf_h[c]))
                st->valid[pix] = 0;
            if (!st->valid[pix]) {
                memset(st->slc + pix * n * 2, 0, n * 2 * sizeof(float));
                continue;
            }
            /* Amplitudes, sorted by insertion, before the phase rotation that
               leaves them unchanged but for rounding. */
            for (k = 0; k < n; k++) {
                const float *s = st->slc + (pix * n + k) * 2;
                float v = sqrtf(s[0] * s[0] + s[1] * s[1]);
                int j = k;

                while (j > 0 && a[j - 1] > v) {
                    a[j] = a[j - 1];
                    j--;
                }
                a[j] = v;
            }
            /* Remove the geometric phase: date k times exp(i phase_k). */
            if (geo) {
                geometry_phases(geo, row, c, fd_elev >= 0 ? buf_h[c] : geo->h0,
                                phase);
                for (k = 0; k < n; k++) {
                    float *s = st->slc + (pix * n + k) * 2;
                    const double cs = cos(phase[k]), sn = sin(phase[k]);
                    const double re = s[0], im = s[1];

                    s[0] = (float)(re * cs - im * sn);
                    s[1] = (float)(re * sn + im * cs);
                }
            }
        }
    }
}

static void set_timestamp(const char *name, const struct utc *start,
                          const struct utc *end)
{
    struct TimeStamp ts;
    DateTime dt[2];
    const struct utc *t[2] = {start, end};
    int i;

    for (i = 0; i < 2; i++) {
        datetime_set_type(&dt[i], DATETIME_ABSOLUTE, DATETIME_YEAR,
                          DATETIME_SECOND, 0);
        datetime_set_year(&dt[i], t[i]->year);
        datetime_set_month(&dt[i], t[i]->month);
        datetime_set_day(&dt[i], t[i]->day);
        datetime_set_hour(&dt[i], t[i]->hour);
        datetime_set_minute(&dt[i], t[i]->minute);
        datetime_set_second(&dt[i], floor(t[i]->second));
    }
    G_init_timestamp(&ts);
    G_set_timestamp_range(&ts, &dt[0], &dt[1]);
    G_write_raster_timestamp(name, &ts);
}

static void set_json_attribute(G_JSON_Object *obj, const char *key,
                               const struct epoch *e, const char *path)
{
    char *value = epoch_attribute(e, path);

    if (value)
        G_json_object_set_string(obj, key, value);
    else
        G_json_object_set_null(obj, key);
    G_free(value);
}

static void write_metadata(const char *name, const struct settings *s,
                           const char *shp_test, const char *estimator,
                           const struct epoch *epochs, int n)
{
    G_JSON_Value *root = G_json_value_init_object(), *params, *stack, *list;
    G_JSON_Object *obj = G_json_object(root), *po, *so;
    G_JSON_Array *arr;
    char path[GPATH_MAX], *text;
    FILE *fp;
    int i;

    params = G_json_value_init_object();
    po = G_json_object(params);
    G_json_object_set_string(
        po, "algorithm",
        "phase linking temporal coherence (Pepe-Lanari goodness of fit)");
    G_json_object_set_string(po, "estimator", estimator);
    G_json_object_set_string(po, "shp_test", shp_test);
    G_json_object_set_number(po, "alpha", s->alpha);
    G_json_object_set_number(po, "window_azimuth", s->win_az);
    G_json_object_set_number(po, "window_range", s->win_rg);
    G_json_object_set_number(po, "min_shp", s->min_looks);
    G_json_object_set_boolean(po, "bias_correction", s->bias);
    G_json_object_set_number(po, "dates", s->ndates);
    G_json_object_set_string(po, "precision", "single");
    G_json_object_set_string(po, "phase_reference", s->phase_reference);
    if (s->orbit)
        G_json_object_set_string(po, "orbit", s->orbit);
    else
        G_json_object_set_null(po, "orbit");
    G_json_object_set_value(obj, "temporal_coherence", params);

    stack = G_json_value_init_object();
    so = G_json_object(stack);
    set_json_attribute(so, "mission", &epochs[0], "swath.mission");
    set_json_attribute(so, "swath", &epochs[0], "swath.swath");
    set_json_attribute(so, "polarization", &epochs[0], "swath.polarization");
    set_json_attribute(so, "pass", &epochs[0], "swath.pass");
    set_json_attribute(so, "relative_orbit", &epochs[0],
                       "product.relative_orbit_start");
    G_json_object_set_value(obj, "stack", stack);

    list = G_json_value_init_array();
    arr = G_json_array(list);
    for (i = 0; i < n; i++) {
        G_JSON_Value *ev = G_json_value_init_object(), *maps;
        G_JSON_Object *eo = G_json_object(ev);
        G_JSON_Array *ma;
        char iso[64], full[GNAME_MAX + GMAPSET_MAX + 1];
        int k;

        G_json_object_set_string(eo, "input", epochs[i].basename);
        maps = G_json_value_init_array();
        ma = G_json_array(maps);
        for (k = 0; k < 2; k++) {
            snprintf(full, sizeof(full), "%s@%s", epochs[i].name[k],
                     epochs[i].mapset[k]);
            G_json_array_append_string(ma, full);
        }
        G_json_object_set_value(eo, "maps", maps);
        if (epochs[i].has_time) {
            const struct utc *t = &epochs[i].start;

            snprintf(iso, sizeof(iso), "%04d-%02d-%02dT%02d:%02d:%09.6f",
                     t->year, t->month, t->day, t->hour, t->minute, t->second);
            G_json_object_set_string(eo, "start_time", iso);
        }
        else
            G_json_object_set_null(eo, "start_time");
        set_json_attribute(eo, "product", &epochs[i], "product.product_name");
        set_json_attribute(eo, "absolute_orbit", &epochs[i],
                           "product.absolute_orbit_start");
        if (epochs[i].orbit[0])
            G_json_object_set_string(eo, "orbit", epochs[i].orbit);
        else
            G_json_object_set_null(eo, "orbit");
        G_json_array_append_value(arr, ev);
    }
    G_json_object_set_value(obj, "epochs", list);

    G__make_mapset_element_misc("cell_misc", name);
    G_file_name_misc(path, "cell_misc", "description.json", name, G_mapset());
    text = G_json_serialize_to_string_pretty(root);
    fp = fopen(path, "w");
    if (!fp)
        G_fatal_error(_("Unable to write <%s>"), path);
    fputs(text, fp);
    fclose(fp);
    G_json_free_serialized_string(text);
    G_json_value_free(root);
}

static void grey_colors(const char *name)
{
    struct Colors colors;
    DCELL v0 = 0.0, v1 = 1.0;

    Rast_init_colors(&colors);
    Rast_add_d_color_rule(&v0, 0, 0, 0, &v1, 255, 255, 255, &colors);
    Rast_write_colors(name, G_mapset(), &colors);
}

/* Compact date of an epoch for map names, YYYYMMDD or YYYYMMDDTHHMMSS. */
static void date_tag(const struct epoch *e, int with_time, char *buf,
                     size_t len)
{
    const struct utc *t = &e->start;

    if (with_time)
        snprintf(buf, len, "%04d%02d%02dT%02d%02d%02d", t->year, t->month,
                 t->day, t->hour, t->minute, (int)t->second);
    else
        snprintf(buf, len, "%04d%02d%02d", t->year, t->month, t->day);
}

static void write_support(const char *name, const char *title,
                          const char *units, const char *label,
                          const char *sources, const char *description,
                          const struct epoch *epochs, int n)
{
    struct History hist;
    int i, dated = 1;

    Rast_put_cell_title(name, title);
    Rast_write_units(name, units);
    Rast_write_semantic_label(name, label);
    Rast_short_history(name, "raster", &hist);
    Rast_set_history(&hist, HIST_DATSRC_1, sources);
    Rast_set_history(&hist, HIST_KEYWRD, description);
    Rast_command_history(&hist);
    Rast_write_history(name, &hist);

    for (i = 0; i < n; i++)
        dated &= epochs[i].has_time;
    if (dated) {
        /* Whole seconds, the last one rounded up. */
        struct utc end = epochs[n - 1].end;

        utc_from_seconds(floor(utc_seconds(&end)) + 1.0, &end);
        set_timestamp(name, &epochs[0].start, &end);
    }
}

int main(int argc, char *argv[])
{
    struct GModule *module;
    struct {
        struct Option *input, *output, *shp_count, *pairs, *pairs_mode, *window,
            *shp_test, *alpha, *min_shp, *estimator, *reference, *elevation,
            *orbit, *orbit_dir, *memory, *platform, *device;
    } opt;
    struct {
        struct Flag *bias, *flat, *list;
    } flag;
    struct settings s;
    struct epoch *epochs;
    struct Cell_head region;
    struct strip st;
    struct ocl *o;
    struct geometry geo, *geop = NULL;
    int n, i, j, k, p, nrows, ncols, halo_az, halo_rg, strip_rows, row0, fd_out,
        fd_shp = -1, fd_elev = -1, ref = -1, (*fd)[2], *fd_pairs = NULL,
        (*pair_dates)[2] = NULL;
    size_t max_alloc, per_row, budget;
    FCELL *buf_i, *buf_q, *out_row;
    DCELL *buf_h = NULL;
    CELL *shp_row;
    float *coh, *pairs = NULL;
    int *count;
    double sum = 0.0, *phase;
    long estimated = 0;
    char *pol, *swath, title[512], label[128], stack_label[64], *sources,
        description[1024], **pair_names = NULL;

    G_gisinit(argv[0]);

    module = G_define_module();
    G_add_keyword(_("imagery"));
    G_add_keyword(_("SAR"));
    G_add_keyword(_("radar"));
    G_add_keyword(_("Sentinel-1"));
    G_add_keyword(_("SLC"));
    G_add_keyword(_("InSAR"));
    G_add_keyword(_("coherence"));
    G_add_keyword(_("time series"));
    G_add_keyword(_("OpenCL"));
    G_add_keyword(_("GPU"));
    module->description = _("Computes the temporal coherence of a "
                            "coregistered SAR SLC stack by phase linking.");

    opt.input = G_define_option();
    opt.input->key = "input";
    opt.input->type = TYPE_STRING;
    opt.input->required = YES;
    opt.input->multiple = YES;
    opt.input->key_desc = "basename";
    opt.input->label = _("Basenames of the coregistered complex SLC images "
                         "(at least 3 dates)");
    opt.input->description = _("Each image is the raster pair "
                               "<basename>_i, <basename>_q written by "
                               "r.in.s1slc measure=complex");

    opt.output = G_define_standard_option(G_OPT_R_OUTPUT);
    opt.output->description = _("Name for output temporal coherence raster "
                                "map");

    opt.shp_count = G_define_standard_option(G_OPT_R_OUTPUT);
    opt.shp_count->key = "shp_count";
    opt.shp_count->required = NO;
    opt.shp_count->label = _("Name for output raster map of the number of "
                             "statistically homogeneous pixels");
    opt.shp_count->description = _("Written for every pixel whose stack is "
                                   "valid, also below min_shp");
    opt.shp_count->guisection = _("Output");

    opt.pairs = G_define_standard_option(G_OPT_R_BASENAME_OUTPUT);
    opt.pairs->key = "pairs";
    opt.pairs->required = NO;
    opt.pairs->label = _("Basename for output pair coherence raster maps");
    opt.pairs->description = _("One map <basename>_<date1>_<date2> of |T_ij| "
                               "per pair of dates");
    opt.pairs->guisection = _("Output");

    opt.pairs_mode = G_define_option();
    opt.pairs_mode->key = "pairs_mode";
    opt.pairs_mode->type = TYPE_STRING;
    opt.pairs_mode->required = NO;
    opt.pairs_mode->options = "consecutive,all";
    opt.pairs_mode->answer = "consecutive";
    opt.pairs_mode->label = _("Pairs of dates written with option pairs");
    G_asprintf((char **)&opt.pairs_mode->descriptions, "consecutive;%s;all;%s",
               _("Each date with the next one (N - 1 maps)"),
               _("Every pair of dates (N (N - 1) / 2 maps)"));
    opt.pairs_mode->guisection = _("Output");

    opt.window = G_define_option();
    opt.window->key = "window";
    opt.window->type = TYPE_INTEGER;
    opt.window->required = NO;
    opt.window->multiple = YES;
    opt.window->key_desc = "azimuth,range";
    opt.window->answer = "21,7";
    opt.window->label = _("Size of the SHP search window in lines (azimuth) "
                          "and samples (range)");
    opt.window->description = _("Both sizes must be odd");
    opt.window->guisection = _("Estimation");

    opt.shp_test = G_define_option();
    opt.shp_test->key = "shp_test";
    opt.shp_test->type = TYPE_STRING;
    opt.shp_test->required = NO;
    opt.shp_test->options = "ks,ad,tlog";
    opt.shp_test->answer = "ks";
    opt.shp_test->label = _("Two-sample test of the amplitude time series "
                            "used to select SHPs");
    G_asprintf((char **)&opt.shp_test->descriptions, "ks;%s;ad;%s;tlog;%s",
               _("Kolmogorov-Smirnov"), _("Anderson-Darling (Scholz-Stephens)"),
               _("Welch t-test on log-amplitude"));
    opt.shp_test->guisection = _("Estimation");

    opt.alpha = G_define_option();
    opt.alpha->key = "alpha";
    opt.alpha->type = TYPE_DOUBLE;
    opt.alpha->required = NO;
    opt.alpha->answer = "0.05";
    opt.alpha->label = _("Significance level of the SHP test");
    opt.alpha->description = _("Anderson-Darling supports 0.001 to 0.25");
    opt.alpha->guisection = _("Estimation");

    opt.min_shp = G_define_option();
    opt.min_shp->key = "min_shp";
    opt.min_shp->type = TYPE_INTEGER;
    opt.min_shp->required = NO;
    opt.min_shp->answer = "20";
    opt.min_shp->label = _("Minimum number of SHPs to estimate a pixel");
    opt.min_shp->description = _("Never less than the number of dates, so "
                                 "that the covariance matrix has full rank");
    opt.min_shp->guisection = _("Estimation");

    opt.estimator = G_define_option();
    opt.estimator->key = "estimator";
    opt.estimator->type = TYPE_STRING;
    opt.estimator->required = NO;
    opt.estimator->options = "evd,emi";
    opt.estimator->answer = "evd";
    opt.estimator->label = _("Phase linking estimator");
    G_asprintf((char **)&opt.estimator->descriptions, "evd;%s;emi;%s",
               _("Dominant eigenvector of the coherence matrix (SqueeSAR)"),
               _("Eigendecomposition-based maximum likelihood (Ansari et "
                 "al. 2018)"));
    opt.estimator->guisection = _("Estimation");

    opt.reference = G_define_option();
    opt.reference->key = "reference";
    opt.reference->type = TYPE_STRING;
    opt.reference->required = NO;
    opt.reference->key_desc = "basename";
    opt.reference->label = _("Input date the stack is coregistered on");
    opt.reference->description = _("Its grid timing and orbit define the "
                                   "geometry; required with -f or "
                                   "elevation");
    opt.reference->guisection = _("Geometry");

    opt.elevation = G_define_standard_option(G_OPT_R_ELEV);
    opt.elevation->required = NO;
    opt.elevation->label = _("Name of the elevation raster map in radar "
                             "geometry, on the grid of the stack");
    opt.elevation->description = _("Heights in meters above the WGS84 "
                                   "ellipsoid; removes the topographic "
                                   "phase (implies -f)");
    opt.elevation->guisection = _("Geometry");

    opt.orbit = G_define_option();
    opt.orbit->key = "orbit";
    opt.orbit->type = TYPE_STRING;
    opt.orbit->required = NO;
    opt.orbit->options = "best,precise,restituted,annotation";
    opt.orbit->answer = "best";
    opt.orbit->label = _("Orbits used for the flat-earth and topographic "
                         "phase");
    opt.orbit->description = _("Orbit files are searched in orbit_dir, then "
                               "downloaded");
    G_asprintf((char **)&opt.orbit->descriptions,
               "best;%s;precise;%s;restituted;%s;annotation;%s",
               _("Per date, precise if available, else restituted, else "
                 "annotation"),
               _("Precise orbit files only (POEORB, about 5 cm)"),
               _("Restituted orbit files only (RESORB)"),
               _("State vectors of the product annotation"));
    opt.orbit->guisection = _("Geometry");

    opt.orbit_dir = G_define_standard_option(G_OPT_M_DIR);
    opt.orbit_dir->key = "orbit_dir";
    opt.orbit_dir->required = NO;
    opt.orbit_dir->label = _("Directory of the orbit files");
    opt.orbit_dir->description = _("Searched first, downloaded files are "
                                   "stored in it (default: "
                                   "$HOME/.grass8/sentinel1_orbits)");
    opt.orbit_dir->guisection = _("Geometry");

    opt.memory = G_define_standard_option(G_OPT_MEMORYMB);

    opt.platform = G_define_option();
    opt.platform->key = "platform";
    opt.platform->type = TYPE_INTEGER;
    opt.platform->required = NO;
    opt.platform->label = _("Index of the OpenCL platform (see -l)");
    opt.platform->description = _("Default: first GPU of any platform, else "
                                  "the first device");
    opt.platform->guisection = _("OpenCL");

    opt.device = G_define_option();
    opt.device->key = "device";
    opt.device->type = TYPE_INTEGER;
    opt.device->required = NO;
    opt.device->label = _("Index of the OpenCL device on the platform (see "
                          "-l)");
    opt.device->guisection = _("OpenCL");

    flag.bias = G_define_flag();
    flag.bias->key = 'b';
    flag.bias->label = _("Remove the bias of the sample coherence "
                         "magnitudes");
    flag.bias->description = _("First-order correction for the number of "
                               "looks; recommended with estimator=evd only");
    flag.bias->guisection = _("Estimation");

    flag.flat = G_define_flag();
    flag.flat->key = 'f';
    flag.flat->label = _("Remove the flat-earth phase");
    flag.flat->description = _("From the orbits, at the annotated terrain "
                               "height; use elevation to follow the relief");
    flag.flat->guisection = _("Geometry");

    flag.list = G_define_flag();
    flag.list->key = 'l';
    flag.list->description = _("List the OpenCL platforms and devices and "
                               "exit");
    flag.list->suppress_required = YES;
    flag.list->guisection = _("OpenCL");

    if (G_parser(argc, argv))
        exit(EXIT_FAILURE);

    if (flag.list->answer) {
        ocl_list_devices();
        exit(EXIT_SUCCESS);
    }

    /* Settings. */
    for (i = 0; opt.window->answers[i]; i++)
        ;
    if (i != 2)
        G_fatal_error(_("Option window needs two sizes azimuth,range"));
    s.win_az = atoi(opt.window->answers[0]);
    s.win_rg = atoi(opt.window->answers[1]);
    if (s.win_az < 1 || s.win_rg < 1 || s.win_az % 2 == 0 || s.win_rg % 2 == 0)
        G_fatal_error(_("Window sizes must be odd, got %d,%d"), s.win_az,
                      s.win_rg);
    s.alpha = atof(opt.alpha->answer);
    if (!(s.alpha > 0 && s.alpha < 1))
        G_fatal_error(_("Option alpha must be between 0 and 1"));
    s.test = !strcmp(opt.shp_test->answer, "ad")     ? SHP_AD
             : !strcmp(opt.shp_test->answer, "tlog") ? SHP_TLOG
                                                     : SHP_KS;
    s.emi = !strcmp(opt.estimator->answer, "emi");
    s.bias = flag.bias->answer;
    s.phase_reference = opt.elevation->answer ? "elevation"
                        : flag.flat->answer   ? "ellipsoid"
                                              : "none";
    s.orbit = NULL;

    /* Inputs, sorted chronologically. */
    for (n = 0; opt.input->answers[n]; n++)
        ;
    epochs = G_calloc(n, sizeof(*epochs));
    for (i = 0; i < n; i++)
        epoch_load(&epochs[i], opt.input->answers[i]);
    check_stack(epochs, n);
    qsort(epochs, n, sizeof(*epochs), compare_epochs);
    s.ndates = n;
    s.min_looks = atoi(opt.min_shp->answer);
    if (s.min_looks < n)
        s.min_looks = n;
    if (s.win_az * s.win_rg < s.min_looks)
        G_fatal_error(_("The %dx%d window holds fewer pixels than the %d SHPs "
                        "needed"),
                      s.win_az, s.win_rg, s.min_looks);
    shp_constants(&s);
    if (opt.reference->answer) {
        for (i = 0; i < n; i++)
            if (!strcmp(epochs[i].basename, opt.reference->answer))
                ref = i;
        if (ref < 0)
            G_fatal_error(_("Reference <%s> is not one of the inputs"),
                          opt.reference->answer);
    }

    /* Pairs of dates to write, named after their dates. */
    s.pairs_all = !strcmp(opt.pairs_mode->answer, "all");
    s.npairs = 0;
    if (opt.pairs->answer) {
        int dated = 1, with_time = 0;

        s.npairs = s.pairs_all ? n * (n - 1) / 2 : n - 1;
        pair_dates = G_malloc(s.npairs * sizeof(*pair_dates));
        pair_names = G_malloc(s.npairs * sizeof(char *));
        for (i = 0, p = 0; i < n - 1; i++)
            for (j = i + 1; j < (s.pairs_all ? n : i + 2); j++, p++) {
                pair_dates[p][0] = i;
                pair_dates[p][1] = j;
            }
        for (i = 0; i < n; i++) {
            dated &= epochs[i].has_time;
            if (i > 0 && dated &&
                epochs[i].start.year == epochs[i - 1].start.year &&
                epochs[i].start.month == epochs[i - 1].start.month &&
                epochs[i].start.day == epochs[i - 1].start.day)
                with_time = 1;
        }
        for (p = 0; p < s.npairs; p++) {
            char a[32], b[32];

            if (dated) {
                date_tag(&epochs[pair_dates[p][0]], with_time, a, sizeof(a));
                date_tag(&epochs[pair_dates[p][1]], with_time, b, sizeof(b));
            }
            else {
                snprintf(a, sizeof(a), "%d", pair_dates[p][0] + 1);
                snprintf(b, sizeof(b), "%d", pair_dates[p][1] + 1);
            }
            G_asprintf(&pair_names[p], "%s_%s_%s", opt.pairs->answer, a, b);
            if (G_legal_filename(pair_names[p]) < 0)
                G_fatal_error(_("<%s> is an illegal file name"), pair_names[p]);
            if (!G_get_overwrite() && G_find_raster2(pair_names[p], G_mapset()))
                G_fatal_error(_("Raster map <%s> already exists"),
                              pair_names[p]);
        }
        if (!dated)
            G_warning(_("Some inputs have no acquisition time: pair maps are "
                        "named by chronological rank"));
    }

    G_get_window(&region);
    check_region(&epochs[0], &region);
    nrows = region.rows;
    ncols = region.cols;
    halo_az = s.win_az / 2;
    halo_rg = s.win_rg / 2;
    st.padded_cols = ncols + 2 * halo_rg;

    /* Flat-earth and topographic phase from the orbits. */
    if (flag.flat->answer || opt.elevation->answer) {
        struct orbit *orbits = G_malloc(n * sizeof(*orbits));
        double hmin, hmax, h0;
        char cache[GPATH_MAX];

        if (ref < 0)
            G_fatal_error(_("Option reference is required to remove the "
                            "flat-earth and topographic phase"));
        if (opt.orbit_dir->answer)
            G_strlcpy(cache, opt.orbit_dir->answer, sizeof(cache));
        else
            snprintf(cache, sizeof(cache), "%s/sentinel1_orbits",
                     G_config_path());
        for (i = 0; i < n; i++) {
            if (!strcmp(opt.orbit->answer, "annotation"))
                orbit_from_annotation(&epochs[i], &orbits[i]);
            else if (!strcmp(opt.orbit->answer, "best"))
                orbit_best(&epochs[i], cache, &orbits[i]);
            else
                orbit_from_file(
                    &epochs[i],
                    !strcmp(opt.orbit->answer, "precise") ? "POEORB" : "RESORB",
                    cache, 1, &orbits[i]);
            G_strlcpy(epochs[i].orbit, orbits[i].source,
                      sizeof(epochs[i].orbit));
        }
        s.orbit = opt.orbit->answer;

        h0 = epochs[ref].meta
                 ? json_get_number(epochs[ref].meta, "swath.terrain_height")
                 : NAN;
        if (isnan(h0))
            h0 = 0.0;
        hmin = hmax = h0;
        if (opt.elevation->answer) {
            const char *mapset = G_find_raster2(opt.elevation->answer, "");
            struct FPRange range;
            DCELL lo, hi;

            if (!mapset)
                G_fatal_error(_("Raster map <%s> not found"),
                              opt.elevation->answer);
            if (Rast_read_fp_range(opt.elevation->answer, mapset, &range) < 0)
                G_fatal_error(_("Unable to read the range of <%s>"),
                              opt.elevation->answer);
            Rast_get_fp_range_min_max(&range, &lo, &hi);
            if (Rast_is_d_null_value(&lo))
                G_fatal_error(_("Raster map <%s> has no data"),
                              opt.elevation->answer);
            hmin = lo;
            hmax = hi;
            fd_elev = Rast_open_old(opt.elevation->answer, mapset);
            buf_h = Rast_allocate_d_buf();
        }
        geometry_init(&geo, epochs, n, ref, orbits, &region, hmin, hmax, h0);
        geop = &geo;
        for (i = 0; i < n; i++)
            orbit_free(&orbits[i]);
        G_free(orbits);
    }
    else if (opt.reference->answer)
        G_warning(_("Option reference is only used with -f or elevation"));
    phase = G_malloc(n * sizeof(double));

    o = ocl_open(opt.platform->answer ? atoi(opt.platform->answer) : -1,
                 opt.device->answer ? atoi(opt.device->answer) : -1, &s,
                 st.padded_cols, &max_alloc);

    /* Strip height: the host strip (slc, amp, valid) and the outputs within
       the memory budget, the largest device buffer within the allocation
       limit. */
    per_row = (size_t)st.padded_cols * (12 * n + 1) +
              (size_t)ncols * 4 * (2 + s.npairs);
    budget = (size_t)atoi(opt.memory->answer) * 1024 * 1024;
    strip_rows = (int)(budget / per_row) - (s.win_az - 1);
    if (max_alloc > 0) {
        int dev_rows = (int)(max_alloc / ((size_t)st.padded_cols * n * 8)) -
                       (s.win_az - 1);

        if (dev_rows < strip_rows)
            strip_rows = dev_rows;
        if (s.npairs) {
            dev_rows = (int)(max_alloc / ((size_t)ncols * s.npairs * 4));
            if (dev_rows < strip_rows)
                strip_rows = dev_rows;
        }
    }
    if (strip_rows < 1)
        G_fatal_error(_("Not enough memory for one row of %d columns and %d "
                        "dates; increase memory or reduce the region"),
                      ncols, n);
    if (strip_rows > nrows)
        strip_rows = nrows;
    st.padded_rows = strip_rows + 2 * halo_az;
    G_verbose_message(_("Strips of %d rows"), strip_rows);

    st.slc = G_malloc((size_t)st.padded_rows * st.padded_cols * n * 2 *
                      sizeof(float));
    st.amp =
        G_malloc((size_t)st.padded_rows * st.padded_cols * n * sizeof(float));
    st.valid = G_malloc((size_t)st.padded_rows * st.padded_cols);
    coh = G_malloc((size_t)strip_rows * ncols * sizeof(float));
    count = G_malloc((size_t)strip_rows * ncols * sizeof(int));
    if (s.npairs)
        pairs = G_malloc((size_t)s.npairs * strip_rows * ncols * sizeof(float));
    buf_i = Rast_allocate_f_buf();
    buf_q = Rast_allocate_f_buf();
    out_row = Rast_allocate_f_buf();
    shp_row = Rast_allocate_c_buf();

    fd = G_malloc(n * sizeof(*fd));
    for (i = 0; i < n; i++)
        for (k = 0; k < 2; k++)
            fd[i][k] = Rast_open_old(epochs[i].name[k], epochs[i].mapset[k]);
    fd_out = Rast_open_new(opt.output->answer, FCELL_TYPE);
    if (opt.shp_count->answer)
        fd_shp = Rast_open_new(opt.shp_count->answer, CELL_TYPE);
    if (s.npairs) {
        fd_pairs = G_malloc(s.npairs * sizeof(int));
        for (p = 0; p < s.npairs; p++)
            fd_pairs[p] = Rast_open_new(pair_names[p], FCELL_TYPE);
    }

    G_message(_("Estimating the temporal coherence of %d dates (%d rows x %d "
                "columns)..."),
              n, nrows, ncols);
    for (row0 = 0; row0 < nrows; row0 += strip_rows) {
        const int rows = row0 + strip_rows <= nrows ? strip_rows : nrows - row0;
        const size_t plane = (size_t)rows * ncols;
        int r, c;

        G_percent(row0, nrows, 2);
        st.padded_rows = rows + 2 * halo_az;
        read_strip(&st, fd, n, row0 - halo_az, nrows, ncols, halo_rg, buf_i,
                   buf_q, geop, fd_elev, buf_h, phase);
        ocl_run(o, st.slc, st.amp, st.valid, st.padded_rows, rows, ncols, coh,
                count, pairs);
        for (r = 0; r < rows; r++) {
            for (c = 0; c < ncols; c++) {
                const float v = coh[(size_t)r * ncols + c];
                const int m = count[(size_t)r * ncols + c];

                if (isnan(v))
                    Rast_set_f_null_value(&out_row[c], 1);
                else {
                    out_row[c] = v;
                    sum += v;
                    estimated++;
                }
                if (m < 0)
                    Rast_set_c_null_value(&shp_row[c], 1);
                else
                    shp_row[c] = m;
            }
            Rast_put_f_row(fd_out, out_row);
            if (fd_shp >= 0)
                Rast_put_c_row(fd_shp, shp_row);
            for (p = 0; p < s.npairs; p++) {
                const float *src = pairs + p * plane + (size_t)r * ncols;

                for (c = 0; c < ncols; c++) {
                    if (isnan(src[c]))
                        Rast_set_f_null_value(&out_row[c], 1);
                    else
                        out_row[c] = src[c];
                }
                Rast_put_f_row(fd_pairs[p], out_row);
            }
        }
    }
    G_percent(1, 1, 1);
    ocl_close(o);
    for (i = 0; i < n; i++)
        for (k = 0; k < 2; k++)
            Rast_close(fd[i][k]);
    if (fd_elev >= 0)
        Rast_close(fd_elev);
    Rast_close(fd_out);
    if (fd_shp >= 0)
        Rast_close(fd_shp);
    for (p = 0; p < s.npairs; p++)
        Rast_close(fd_pairs[p]);
    if (geop)
        geometry_free(geop);

    /* Support files and metadata. */
    swath = epoch_attribute(&epochs[0], "swath.swath");
    pol = epoch_attribute(&epochs[0], "swath.polarization");
    snprintf(stack_label, sizeof(stack_label), "%s%s%s%s", swath ? swath : "",
             swath ? " " : "", pol ? pol : "", pol ? " " : "");
    {
        size_t len = 1;

        for (i = 0; i < n; i++)
            len += strlen(epochs[i].basename) + 1;
        sources = G_calloc(len, 1);
        for (i = 0; i < n; i++) {
            if (i)
                strcat(sources, ",");
            strcat(sources, epochs[i].basename);
        }
    }
    snprintf(
        description, sizeof(description),
        "%d dates, %s estimator, %s SHP test (alpha %g), %dx%d "
        "window%s%s%s%s, OpenCL single precision",
        n, s.emi ? "EMI" : "EVD",
        s.test == SHP_AD ? "AD" : (s.test == SHP_TLOG ? "TLOG" : "KS"), s.alpha,
        s.win_az, s.win_rg, s.bias ? ", bias corrected" : "",
        !strcmp(s.phase_reference, "elevation") ? ", topographic phase removed"
        : !strcmp(s.phase_reference, "ellipsoid") ? ", flat-earth phase removed"
                                                  : "",
        s.orbit ? " with " : "", s.orbit ? s.orbit : "");
    if (s.orbit)
        strncat(description, " orbits",
                sizeof(description) - strlen(description) - 1);

    snprintf(title, sizeof(title),
             "Temporal coherence of the %d-date %sSLC stack", n, stack_label);
    if (pol)
        snprintf(label, sizeof(label), "S1_%s_TEMPORAL_COHERENCE", pol);
    else
        G_strlcpy(label, "TEMPORAL_COHERENCE", sizeof(label));
    write_support(opt.output->answer, title, "", label, sources, description,
                  epochs, n);
    write_metadata(opt.output->answer, &s, opt.shp_test->answer,
                   opt.estimator->answer, epochs, n);
    grey_colors(opt.output->answer);
    if (opt.shp_count->answer) {
        snprintf(title, sizeof(title),
                 "Number of statistically homogeneous pixels of the %d-date "
                 "%sSLC stack",
                 n, stack_label);
        if (pol)
            snprintf(label, sizeof(label), "S1_%s_SHP_COUNT", pol);
        else
            G_strlcpy(label, "SHP_COUNT", sizeof(label));
        write_support(opt.shp_count->answer, title, "pixels", label, sources,
                      description, epochs, n);
        write_metadata(opt.shp_count->answer, &s, opt.shp_test->answer,
                       opt.estimator->answer, epochs, n);
    }
    for (p = 0; p < s.npairs; p++) {
        struct epoch pair[2];
        char pair_sources[2 * GNAME_MAX + 2];

        pair[0] = epochs[pair_dates[p][0]];
        pair[1] = epochs[pair_dates[p][1]];
        snprintf(title, sizeof(title),
                 "Coherence of the %sSLC pair %s, %s (SHPs of the %d-date "
                 "stack)",
                 stack_label, pair[0].basename, pair[1].basename, n);
        if (pol)
            snprintf(label, sizeof(label), "S1_%s_COHERENCE", pol);
        else
            G_strlcpy(label, "COHERENCE", sizeof(label));
        snprintf(pair_sources, sizeof(pair_sources), "%s,%s", pair[0].basename,
                 pair[1].basename);
        write_support(pair_names[p], title, "", label, pair_sources,
                      description, pair, 2);
        write_metadata(pair_names[p], &s, opt.shp_test->answer,
                       opt.estimator->answer, pair, 2);
        grey_colors(pair_names[p]);
    }

    G_message(_("Temporal coherence <%s>: %ld pixels estimated, mean %.3f"),
              opt.output->answer, estimated, estimated ? sum / estimated : NAN);
    if (s.npairs)
        G_message(_("%d pair coherence maps <%s_*>"), s.npairs,
                  opt.pairs->answer);
    exit(EXIT_SUCCESS);
}
