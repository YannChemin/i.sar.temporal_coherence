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
    char name[2][GNAME_MAX];   /* <basename>_i, <basename>_q */
    char mapset[2][GMAPSET_MAX];
    struct Cell_head head;
    char *meta;                /* description.json text, NULL if absent */
    int has_time;
    struct utc start, end;
    double start_sec;
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
char *read_text_file(const char *path);

/* stats.c */
void shp_constants(struct settings *s);

/* ocl.c */
struct ocl;
void ocl_list_devices(void);
struct ocl *ocl_open(int platform, int device, const struct settings *s,
                     int padded_cols, size_t *max_alloc);
void ocl_run(struct ocl *o, const float *slc, const float *amp,
             const unsigned char *valid, int padded_rows, int rows,
             int cols, float *coh, int *count);
void ocl_close(struct ocl *o);

#endif
