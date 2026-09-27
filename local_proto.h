#ifndef I_SAR_TEMPORAL_COHERENCE_LOCAL_PROTO_H
#define I_SAR_TEMPORAL_COHERENCE_LOCAL_PROTO_H

#include <stddef.h>

#include <grass/gis.h>
#include <grass/raster.h>

/* Kernel private arrays hold six N x N matrices per work-item. */
#define MAX_DATES 64

enum shp_test { SHP_KS = 0, SHP_AD = 1, SHP_TLOG = 2 };

/* Broken-down UTC time, second with fraction. */
struct utc {
    int year, month, day, hour, minute;
    double second;
};

/* One date of the stack: complex raster pair and its metadata. */
struct epoch {
    char basename[GNAME_MAX];
    char name[2][GNAME_MAX]; /* <basename>_i, <basename>_q */
    char mapset[2][GMAPSET_MAX];
    struct Cell_head head;
    char *meta; /* description.json text, NULL if absent */
    int has_time;
    struct utc start, end;
    double start_sec;
    char orbit[32]; /* orbit source used for the phase, "" if none */
};

/* Estimation settings shared by the host and the kernel build. */
struct settings {
    int ndates;
    int win_az, win_rg;
    enum shp_test test;
    double alpha;
    int emi;
    int bias;
    int min_looks;
    int ks_max;
    double ad_norm, ad_sigma, ad_crit;
    double tlog_crit;
    int npairs;    /* pair coherence maps to write, 0 for none */
    int pairs_all; /* all pairs i < j instead of consecutive ones */
    const char *phase_reference; /* none, ellipsoid or elevation */
    const char *orbit;           /* orbit source, NULL without phase */
};

/* Orbit state vectors, times in seconds since 1970 (UTC). */
struct orbit {
    int n;
    double *t;
    double (*pos)[3], (*vel)[3];
    char source[32];
};

/* Flat-earth and topographic phase of every date relative to the
   reference date, on a coarse grid of the region, per height level. */
struct geometry {
    int ndates;
    int grid_rows, grid_cols, step;
    int nheights;
    double heights[3];
    double *phase; /* [node][height][date] */
    double h0;     /* height used without elevation map */
};

/* epoch.c */
void epoch_load(struct epoch *e, const char *basename);
void check_stack(struct epoch *epochs, int n);
void check_region(const struct epoch *e, const struct Cell_head *region);
char *epoch_attribute(const struct epoch *e, const char *path);
double utc_seconds(const struct utc *t);
void utc_from_seconds(double sec, struct utc *t);

/* jsonget.c */
char *json_get(const char *text, const char *path);
double json_get_number(const char *text, const char *path);
int json_array_length(const char *text, const char *path);
char *read_text_file(const char *path);

/* stats.c */
void shp_constants(struct settings *s);

/* orbit.c */
void orbit_from_annotation(const struct epoch *e, struct orbit *o);
int orbit_from_file(const struct epoch *e, const char *kind,
                    const char *cache_dir, int required, struct orbit *o);
void orbit_best(const struct epoch *e, const char *cache_dir, struct orbit *o);
void orbit_state(const struct orbit *o, double t, double *pos, double *vel);
void orbit_free(struct orbit *o);

/* geometry.c */
void geometry_init(struct geometry *g, const struct epoch *epochs, int n,
                   int ref, const struct orbit *orbits,
                   const struct Cell_head *region, double hmin, double hmax,
                   double h0);
void geometry_phases(const struct geometry *g, int row, int col, double h,
                     double *phase);
void geometry_free(struct geometry *g);

/* ocl.c */
struct ocl;
void ocl_list_devices(void);
struct ocl *ocl_open(int platform, int device, const struct settings *s,
                     int padded_cols, size_t *max_alloc);
void ocl_run(struct ocl *o, const float *slc, const float *amp,
             const unsigned char *valid, int padded_rows, int rows, int cols,
             float *coh, int *count, float *pairs);
void ocl_close(struct ocl *o);

#endif
