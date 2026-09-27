/* Minimal JSON lookup: the value at a dotted path of a document, without
 * building a tree. Only what the r.in.s1slc metadata need is supported:
 * objects, arrays, strings with simple escapes, numbers, true,
 * false and null. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <grass/gis.h>

#include "local_proto.h"

static const char *skip_space(const char *p)
{
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
        p++;
    return p;
}

/* Return the end of the string starting at the quote p, or NULL. */
static const char *skip_string(const char *p)
{
    for (p++; *p; p++) {
        if (*p == '\\') {
            if (!*++p)
                return NULL;
        }
        else if (*p == '"')
            return p + 1;
    }
    return NULL;
}

/* Return the end of the value starting at p, or NULL if malformed. */
static const char *skip_value(const char *p)
{
    int depth = 0;

    p = skip_space(p);
    do {
        if (!*p)
            return NULL;
        if (*p == '"') {
            p = skip_string(p);
            if (!p)
                return NULL;
            continue;
        }
        if (*p == '{' || *p == '[')
            depth++;
        else if (*p == '}' || *p == ']')
            depth--;
        else if (depth == 0) {
            /* Scalar: runs until a delimiter. */
            while (*p && *p != ',' && *p != '}' && *p != ']' && *p != ' ' &&
                   *p != '\n' && *p != '\r' && *p != '\t')
                p++;
            return p;
        }
        p++;
    } while (depth > 0);
    return p;
}

/* Copy the string starting at the quote p, decoding simple escapes. */
static char *copy_string(const char *p)
{
    const char *end = skip_string(p);
    char *out, *o;

    if (!end)
        return NULL;
    out = o = G_malloc(end - p);
    for (p++; p < end - 1; p++) {
        if (*p == '\\') {
            p++;
            switch (*p) {
            case 'n':
                *o++ = '\n';
                break;
            case 't':
                *o++ = '\t';
                break;
            default:
                *o++ = *p;
            }
        }
        else
            *o++ = *p;
    }
    *o = '\0';
    return out;
}

/* Start of the value at the dotted path, NULL if absent. A numeric path
 * component indexes an array (e.g. "swath.orbit_state_vectors.0.time"). */
static const char *json_locate(const char *text, const char *path)
{
    const char *p = skip_space(text);
    char key[256];

    while (*path) {
        const char *dot = strchr(path, '.');
        size_t len = dot ? (size_t)(dot - path) : strlen(path);
        int found = 0;

        if (len >= sizeof(key))
            return NULL;
        memcpy(key, path, len);
        key[len] = '\0';
        path += len + (dot ? 1 : 0);

        if (*p == '[') {
            char *end;
            long index = strtol(key, &end, 10), k;

            if (*end || index < 0)
                return NULL;
            p = skip_space(p + 1);
            for (k = 0; k < index; k++) {
                if (*p == ']')
                    return NULL;
                p = skip_value(p);
                if (!p)
                    return NULL;
                p = skip_space(p);
                if (*p != ',')
                    return NULL;
                p = skip_space(p + 1);
            }
            if (*p == ']')
                return NULL;
            continue;
        }
        if (*p != '{')
            return NULL;
        p = skip_space(p + 1);
        while (*p == '"') {
            const char *kend = skip_string(p);
            int match;

            if (!kend)
                return NULL;
            match = (size_t)(kend - p - 2) == len && !strncmp(p + 1, key, len);
            p = skip_space(kend);
            if (*p != ':')
                return NULL;
            p = skip_space(p + 1);
            if (match) {
                found = 1;
                break;
            }
            p = skip_value(p);
            if (!p)
                return NULL;
            p = skip_space(p);
            if (*p == ',')
                p = skip_space(p + 1);
        }
        if (!found)
            return NULL;
    }
    return p;
}

/* Value at the dotted path (e.g. "swath.polarization") as a newly
 * allocated string: string contents, or the literal text of a number or
 * boolean. NULL if the path is absent, null, an object or an array. */
char *json_get(const char *text, const char *path)
{
    const char *p = json_locate(text, path), *end;
    char *out;

    if (!p)
        return NULL;
    if (*p == '"')
        return copy_string(p);
    if (*p == '{' || *p == '[' || !strncmp(p, "null", 4))
        return NULL;
    end = skip_value(p);
    if (!end || end == p)
        return NULL;
    out = G_malloc(end - p + 1);
    memcpy(out, p, end - p);
    out[end - p] = '\0';
    return out;
}

/* Number value at the dotted path, NAN if absent or not a number. */
double json_get_number(const char *text, const char *path)
{
    char *value = json_get(text, path), *end;
    double x;

    if (!value)
        return NAN;
    x = strtod(value, &end);
    if (end == value || *end)
        x = NAN;
    G_free(value);
    return x;
}

/* Number of elements of the array at the dotted path, -1 if not an array. */
int json_array_length(const char *text, const char *path)
{
    const char *p = json_locate(text, path);
    int n = 0;

    if (!p || *p != '[')
        return -1;
    p = skip_space(p + 1);
    while (*p && *p != ']') {
        p = skip_value(p);
        if (!p)
            return -1;
        n++;
        p = skip_space(p);
        if (*p == ',')
            p = skip_space(p + 1);
    }
    return n;
}

/* Whole file as a newly allocated string, NULL if it cannot be read. */
char *read_text_file(const char *path)
{
    FILE *fp = fopen(path, "rb");
    long size;
    char *text;

    if (!fp)
        return NULL;
    if (fseek(fp, 0, SEEK_END) != 0 || (size = ftell(fp)) < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    text = G_malloc(size + 1);
    if (fread(text, 1, size, fp) != (size_t)size) {
        fclose(fp);
        G_free(text);
        return NULL;
    }
    text[size] = '\0';
    fclose(fp);
    return text;
}
