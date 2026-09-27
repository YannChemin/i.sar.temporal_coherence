/* OpenCL device selection, kernel build and execution. */

#define CL_TARGET_OPENCL_VERSION 120

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <CL/cl.h>

/* ICD loader status when no platform is installed (cl_ext.h). */
#ifndef CL_PLATFORM_NOT_FOUND_KHR
#define CL_PLATFORM_NOT_FOUND_KHR -1001
#endif

#include <grass/gis.h>
#include <grass/glocale.h>

#include "local_proto.h"

/* Aimed duration of one kernel launch, well below GPU watchdog limits. */
#define TARGET_SECONDS 0.5

static const char *kernel_source =
#include "tcoh_cl.h"
    ;

struct ocl {
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_device_id device;
    int padded_cols;
    int ndates;
    int chunk_rows; /* rows per kernel launch */
    size_t local[2];
    cl_mem b_slc, b_amp, b_valid, b_coh, b_count, b_pairs;
    size_t n_slc, n_amp, n_valid, n_coh, n_count, n_pairs;
    int npairs;
};

#define CHECK(err, what)                                                   \
    do {                                                                   \
        if ((err) != CL_SUCCESS)                                           \
            G_fatal_error(_("OpenCL error %d in %s"), (int)(err), (what)); \
    } while (0)

static cl_platform_id *get_platforms(cl_uint *count)
{
    cl_platform_id *ids;
    cl_int err = clGetPlatformIDs(0, NULL, count);

    if (err == CL_PLATFORM_NOT_FOUND_KHR || *count == 0)
        G_fatal_error(_("No OpenCL platform found; install an OpenCL driver "
                        "(ICD) for the device"));
    CHECK(err, "clGetPlatformIDs");
    ids = G_malloc(*count * sizeof(*ids));
    CHECK(clGetPlatformIDs(*count, ids, NULL), "clGetPlatformIDs");
    return ids;
}

static cl_device_id *get_devices(cl_platform_id platform, cl_uint *count)
{
    cl_device_id *ids;
    cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, count);

    if (err == CL_DEVICE_NOT_FOUND || *count == 0) {
        *count = 0;
        return NULL;
    }
    CHECK(err, "clGetDeviceIDs");
    ids = G_malloc(*count * sizeof(*ids));
    CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, *count, ids, NULL),
          "clGetDeviceIDs");
    return ids;
}

static void device_string(cl_device_id dev, cl_device_info what, char *buf,
                          size_t len)
{
    if (clGetDeviceInfo(dev, what, len, buf, NULL) != CL_SUCCESS)
        G_strlcpy(buf, "?", len);
}

static void platform_string(cl_platform_id p, cl_platform_info what, char *buf,
                            size_t len)
{
    if (clGetPlatformInfo(p, what, len, buf, NULL) != CL_SUCCESS)
        G_strlcpy(buf, "?", len);
}

static const char *type_name(cl_device_type type)
{
    if (type & CL_DEVICE_TYPE_GPU)
        return "GPU";
    if (type & CL_DEVICE_TYPE_CPU)
        return "CPU";
    if (type & CL_DEVICE_TYPE_ACCELERATOR)
        return "accelerator";
    return "other";
}

void ocl_list_devices(void)
{
    cl_uint np, nd, p, d;
    cl_platform_id *platforms = get_platforms(&np);

    fprintf(stdout, "platform|device|type|name|version|compute_units|"
                    "global_memory_mb\n");
    for (p = 0; p < np; p++) {
        cl_device_id *devices = get_devices(platforms[p], &nd);

        for (d = 0; d < nd; d++) {
            char name[256], version[256];
            cl_device_type type;
            cl_uint units = 0;
            cl_ulong mem = 0;

            device_string(devices[d], CL_DEVICE_NAME, name, sizeof(name));
            device_string(devices[d], CL_DEVICE_VERSION, version,
                          sizeof(version));
            clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(type), &type,
                            NULL);
            clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(units), &units, NULL);
            clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem),
                            &mem, NULL);
            fprintf(stdout, "%u|%u|%s|%s|%s|%u|%lu\n", p, d, type_name(type),
                    name, version, units, (unsigned long)(mem / (1024 * 1024)));
        }
        G_free(devices);
    }
    G_free(platforms);
}

/* Pick the requested device, or by default the first GPU, else the first
   device of any type. */
static cl_device_id select_device(int platform, int device)
{
    cl_uint np, nd, p, d;
    cl_platform_id *platforms = get_platforms(&np);
    cl_device_id chosen = NULL, *devices;

    if (platform >= 0) {
        if ((cl_uint)platform >= np)
            G_fatal_error(_("OpenCL platform %d not found (%u platforms, see "
                            "-l)"),
                          platform, np);
        devices = get_devices(platforms[platform], &nd);
        if (device < 0)
            device = 0;
        if ((cl_uint)device >= nd)
            G_fatal_error(_("OpenCL device %d not found on platform %d (%u "
                            "devices, see -l)"),
                          device, platform, nd);
        chosen = devices[device];
        G_free(devices);
    }
    else {
        if (device >= 0)
            G_fatal_error(_("Option device requires option platform"));
        for (p = 0; p < np && !chosen; p++) {
            devices = get_devices(platforms[p], &nd);
            for (d = 0; d < nd && !chosen; d++) {
                cl_device_type type;

                clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(type), &type,
                                NULL);
                if (type & CL_DEVICE_TYPE_GPU)
                    chosen = devices[d];
            }
            G_free(devices);
        }
        for (p = 0; p < np && !chosen; p++) {
            devices = get_devices(platforms[p], &nd);
            if (nd > 0) {
                chosen = devices[0];
                G_warning(_("No OpenCL GPU found, using the first device"));
            }
            G_free(devices);
        }
        if (!chosen)
            G_fatal_error(_("No OpenCL device found"));
    }
    G_free(platforms);
    return chosen;
}

struct ocl *ocl_open(int platform, int device, const struct settings *s,
                     int padded_cols, size_t *max_alloc)
{
    struct ocl *o = G_calloc(1, sizeof(*o));
    char name[256], options[1024], pname[256];
    cl_platform_id pid;
    cl_ulong alloc = 0;
    cl_int err;
    size_t group = 1;

    o->device = select_device(platform, device);
    o->padded_cols = padded_cols;
    o->ndates = s->ndates;
    o->npairs = s->npairs;
    o->chunk_rows = 1;
    device_string(o->device, CL_DEVICE_NAME, name, sizeof(name));
    clGetDeviceInfo(o->device, CL_DEVICE_PLATFORM, sizeof(pid), &pid, NULL);
    platform_string(pid, CL_PLATFORM_NAME, pname, sizeof(pname));
    G_message(_("OpenCL device: %s (%s)"), name, pname);
    clGetDeviceInfo(o->device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(alloc),
                    &alloc, NULL);
    *max_alloc = (size_t)alloc;

    o->context = clCreateContext(NULL, 1, &o->device, NULL, NULL, &err);
    CHECK(err, "clCreateContext");
    o->queue = clCreateCommandQueue(o->context, o->device, 0, &err);
    CHECK(err, "clCreateCommandQueue");
    o->program =
        clCreateProgramWithSource(o->context, 1, &kernel_source, NULL, &err);
    CHECK(err, "clCreateProgramWithSource");

    snprintf(options, sizeof(options),
             "-DN=%d -DWA=%d -DWR=%d -DW=%d -DSHP_TEST=%d -DEMI=%d "
             "-DBIAS=%d -DMIN_LOOKS=%d -DKS_MAX=%d -DAD_NORM=%.9ef "
             "-DAD_SIGMA=%.9ef -DAD_CRIT=%.9ef -DTLOG_CRIT=%.9ef "
             "-DNPAIRS=%d -DPAIRS_ALL=%d",
             s->ndates, s->win_az, s->win_rg, padded_cols, (int)s->test, s->emi,
             s->bias, s->min_looks, s->ks_max, s->ad_norm, s->ad_sigma,
             s->ad_crit, s->tlog_crit, s->npairs, s->pairs_all);
    G_debug(1, "OpenCL build options: %s", options);
    err = clBuildProgram(o->program, 1, &o->device, options, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len = 0;
        char *log;

        clGetProgramBuildInfo(o->program, o->device, CL_PROGRAM_BUILD_LOG, 0,
                              NULL, &len);
        log = G_malloc(len + 1);
        clGetProgramBuildInfo(o->program, o->device, CL_PROGRAM_BUILD_LOG, len,
                              log, NULL);
        log[len] = '\0';
        G_fatal_error(_("OpenCL kernel build failed (error %d):\n%s"), (int)err,
                      log);
    }
    o->kernel = clCreateKernel(o->program, "temporal_coherence", &err);
    CHECK(err, "clCreateKernel");

    /* Small work-groups: every work-item has large private arrays. */
    clGetKernelWorkGroupInfo(o->kernel, o->device, CL_KERNEL_WORK_GROUP_SIZE,
                             sizeof(group), &group, NULL);
    o->local[0] = group >= 64 ? 64 : group;
    o->local[1] = 1;
    return o;
}

static double now(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* (Re)allocate device buffer b of at least size bytes. */
static void ensure_buffer(struct ocl *o, cl_mem *b, size_t *have, size_t size,
                          cl_mem_flags flags)
{
    cl_int err;

    if (*b && *have >= size)
        return;
    if (*b)
        clReleaseMemObject(*b);
    *b = clCreateBuffer(o->context, flags, size, NULL, &err);
    CHECK(err, "clCreateBuffer");
    *have = size;
}

void ocl_run(struct ocl *o, const float *slc, const float *amp,
             const unsigned char *valid, int padded_rows, int rows, int cols,
             float *coh, int *count, float *pairs)
{
    const size_t npad = (size_t)padded_rows * o->padded_cols;
    const size_t nout = (size_t)rows * cols;
    const size_t n = o->ndates;
    size_t global[2];
    cl_int r = rows, c = cols;

    ensure_buffer(o, &o->b_slc, &o->n_slc, npad * n * 2 * sizeof(float),
                  CL_MEM_READ_ONLY);
    ensure_buffer(o, &o->b_amp, &o->n_amp, npad * n * sizeof(float),
                  CL_MEM_READ_ONLY);
    ensure_buffer(o, &o->b_valid, &o->n_valid, npad, CL_MEM_READ_ONLY);
    ensure_buffer(o, &o->b_coh, &o->n_coh, nout * sizeof(float),
                  CL_MEM_WRITE_ONLY);
    ensure_buffer(o, &o->b_count, &o->n_count, nout * sizeof(int),
                  CL_MEM_WRITE_ONLY);
    /* A kernel argument needs a buffer even without pairs. */
    ensure_buffer(o, &o->b_pairs, &o->n_pairs,
                  (o->npairs ? o->npairs * nout : 1) * sizeof(float),
                  CL_MEM_WRITE_ONLY);

    CHECK(clEnqueueWriteBuffer(o->queue, o->b_slc, CL_FALSE, 0,
                               npad * n * 2 * sizeof(float), slc, 0, NULL,
                               NULL),
          "clEnqueueWriteBuffer");
    CHECK(clEnqueueWriteBuffer(o->queue, o->b_amp, CL_FALSE, 0,
                               npad * n * sizeof(float), amp, 0, NULL, NULL),
          "clEnqueueWriteBuffer");
    CHECK(clEnqueueWriteBuffer(o->queue, o->b_valid, CL_FALSE, 0, npad, valid,
                               0, NULL, NULL),
          "clEnqueueWriteBuffer");

    CHECK(clSetKernelArg(o->kernel, 0, sizeof(cl_mem), &o->b_slc),
          "clSetKernelArg");
    CHECK(clSetKernelArg(o->kernel, 1, sizeof(cl_mem), &o->b_amp),
          "clSetKernelArg");
    CHECK(clSetKernelArg(o->kernel, 2, sizeof(cl_mem), &o->b_valid),
          "clSetKernelArg");
    CHECK(clSetKernelArg(o->kernel, 3, sizeof(cl_mem), &o->b_coh),
          "clSetKernelArg");
    CHECK(clSetKernelArg(o->kernel, 4, sizeof(cl_mem), &o->b_count),
          "clSetKernelArg");
    CHECK(clSetKernelArg(o->kernel, 5, sizeof(cl_mem), &o->b_pairs),
          "clSetKernelArg");
    CHECK(clSetKernelArg(o->kernel, 6, sizeof(cl_int), &r), "clSetKernelArg");
    CHECK(clSetKernelArg(o->kernel, 7, sizeof(cl_int), &c), "clSetKernelArg");

    /* Launch in chunks of rows lasting about TARGET_SECONDS each: a single
       long kernel is killed by the GPU driver watchdog (amdgpu lockup
       timeout). The chunk height adapts to the measured speed. The global
       size is rounded up to the work-group size; the kernel ignores the
       extra work-items. */
    global[0] = (cols + o->local[0] - 1) / o->local[0] * o->local[0];
    {
        int y0 = 0;

        while (y0 < rows) {
            size_t offset[2] = {0, (size_t)y0};
            int chunk = o->chunk_rows < rows - y0 ? o->chunk_rows : rows - y0;
            double t0, dt;

            global[1] = chunk;
            t0 = now();
            CHECK(clEnqueueNDRangeKernel(o->queue, o->kernel, 2, offset, global,
                                         o->local, 0, NULL, NULL),
                  "clEnqueueNDRangeKernel");
            CHECK(clFinish(o->queue), "clFinish");
            dt = now() - t0;
            y0 += chunk;
            if (chunk == o->chunk_rows) {
                double scale = dt > 0 ? TARGET_SECONDS / dt : 4.0;

                if (scale > 4.0)
                    scale = 4.0;
                o->chunk_rows = (int)(o->chunk_rows * scale);
                if (o->chunk_rows < 1)
                    o->chunk_rows = 1;
                G_debug(1, "Kernel chunk %d rows in %.3f s, next %d rows",
                        chunk, dt, o->chunk_rows);
            }
        }
    }
    CHECK(clEnqueueReadBuffer(o->queue, o->b_coh, CL_FALSE, 0,
                              nout * sizeof(float), coh, 0, NULL, NULL),
          "clEnqueueReadBuffer");
    if (o->npairs)
        CHECK(clEnqueueReadBuffer(o->queue, o->b_pairs, CL_FALSE, 0,
                                  o->npairs * nout * sizeof(float), pairs, 0,
                                  NULL, NULL),
              "clEnqueueReadBuffer");
    CHECK(clEnqueueReadBuffer(o->queue, o->b_count, CL_TRUE, 0,
                              nout * sizeof(int), count, 0, NULL, NULL),
          "clEnqueueReadBuffer");
}

void ocl_close(struct ocl *o)
{
    cl_mem *buffers[] = {&o->b_slc, &o->b_amp,   &o->b_valid,
                         &o->b_coh, &o->b_count, &o->b_pairs};
    size_t i;

    for (i = 0; i < sizeof(buffers) / sizeof(buffers[0]); i++)
        if (*buffers[i])
            clReleaseMemObject(*buffers[i]);
    clReleaseKernel(o->kernel);
    clReleaseProgram(o->program);
    clReleaseCommandQueue(o->queue);
    clReleaseContext(o->context);
    G_free(o);
}
