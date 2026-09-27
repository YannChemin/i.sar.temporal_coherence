# i.sar.temporal_coherence

A [GRASS GIS](https://grass.osgeo.org/) addon that computes the
**phase-linking temporal coherence** of a coregistered stack of SAR
Single Look Complex (SLC) images, such as Sentinel-1 IW/EW SLC imported
by [r.in.s1slc](https://github.com/YannChemin/r.in.s1slc), on a GPU with
**OpenCL** (C module, single precision kernel).

```sh
g.region raster=s1c_20230112_iw2_vv_i
i.sar.temporal_coherence \
    input=s1c_20230112_iw2_vv,s1c_20230124_iw2_vv,s1c_20230205_iw2_vv \
    output=tcoh_vv shp_count=nshp_vv
```

## Why

Temporal coherence tells, pixel by pixel, how well one consistent phase
history explains all N(N−1)/2 interferograms of an N-date stack. It is
the quality mask of distributed-scatterer InSAR (SqueeSAR, MiaplPy,
dolphin): it keeps the natural surfaces whose single interferograms are
too noisy, and rejects the decorrelated ones.

## How it works

For every pixel, following the ESA SNAP `PhaseLinking` operator:

1. **SHP selection**: neighbours in a search window (`window=21,7`)
   whose amplitude time series pass a two-sample test against the
   centre's (`shp_test=ks|ad|tlog`, `alpha=0.05`).
2. **Coherence matrix**: N×N sample covariance over the SHPs,
   normalised; optional magnitude bias correction (`-b`).
3. **Phase linking**: one phase per date from the dominant eigenvector
   (`estimator=evd`) or the EMI estimator (`estimator=emi`).
4. **Temporal coherence** (Pepe-Lanari):
   γ = 2/(N(N−1)) |Σ<sub>i<j</sub> exp(j(arg T<sub>ij</sub> − (φ<sub>i</sub> − φ<sub>j</sub>)))|.

Pixels with a NULL or zero sample at any date, or fewer than
`min_shp` (≥ N) SHPs, are NULL.

## Pair coherence

`pairs=<basename>` also writes the coherence of each pair of consecutive
dates (`pairs_mode=all`: every pair), `<basename>_<date1>_<date2>`,
estimated over the same adaptive SHP windows. A drop in the series dates
a change of the surface, e.g. the harvest of a field.

## Flat-earth and topographic phase, orbits

`-f` removes the flat-earth phase and `elevation=` (heights in radar
geometry on the stack grid) the topographic phase, computed from the
orbits by range-Doppler geometry on the grid of the coregistration
`reference=` date, as SNAP does for its interferograms. Otherwise these
fringes lower the coherence estimated over the window. Orbits are the
precise POEORB files by default (`orbit=precise|restituted|annotation`),
downloaded through GDAL from the ESA STEP mirror used by SNAP and cached
in `orbit_dir` (`~/.grass8/sentinel1_orbits`).

## OpenCL

One work-item per pixel runs the whole chain; the eigenvector comes from
a Householder tridiagonal reduction, bisection and inverse iteration
(Cholesky inverse of |T| for EMI). The kernel needs OpenCL C 1.1 and no
double precision, so it runs on Mesa Clover. `-l` lists the devices,
`platform=`/`device=` pick one (default: first GPU). Launches are split
into ~0.5 s chunks to stay clear of GPU watchdogs.

With 20 dates and the default window, an AMD Radeon Pro WX 7100 (Mesa
Clover) processes about 120 000 pixels/s: a three-burst IW sub-swath
(~37 M pixels) in about five minutes.

## Input

Each date is the basename of an *r.in.s1slc* complex pair
(`<basename>_i`, `<basename>_q`), i.e. its imagery group name. The dates
must be **coregistered** to one grid (a separate step,
*i.sar.coregistration*); the module fails on differing grids and, from
the *r.in.s1slc* metadata, on differing sub-swath, polarization, pass,
relative orbit, calibration or duplicate dates.

## Requirements

- GRASS GIS 8.4 or later
- GDAL (for the orbit downloads)
- An OpenCL 1.1 driver (ICD) and headers, e.g. Debian `ocl-icd-opencl-dev`
  with `mesa-opencl-icd` (AMD) or `pocl-opencl-icd` (CPU)
- For the tests: Python 3 with NumPy and pytest (SciPy optional)

## Installation

```sh
g.extension extension=i.sar.temporal_coherence url=https://github.com/YannChemin/i.sar.temporal_coherence
```

or from a source tree:

```sh
make MODULE_TOPDIR=$HOME/dev/grass
```

## Tests

The tests compare the module, pixel by pixel, with a double precision
plain-loop transcription of the SNAP algorithm on synthetic
distributed-scatterer stacks (agreement within 10⁻⁴), check the SHP
statistics against SciPy, and cover NULL handling, region subsets, strip
seams, metadata and failure modes. Run them with the GRASS the module was
built in, so that it is on the `PATH` (or set `I_SAR_TEMPORAL_COHERENCE`
to the binary):

```sh
make MODULE_TOPDIR=$HOME/dev/grass
$HOME/dev/grass/bin.x86_64-pc-linux-gnu/grass --tmp-project XY \
    --exec python3 -m pytest tests
```

## References

- Ferretti et al. (2011), SqueeSAR, *IEEE TGRS* 49(9).
- Ansari, De Zan, Bamler (2018), EMI, *IEEE TGRS* 56(7).
- Pepe, Lanari (2006), *IEEE TGRS* 44(9).
- [ESA SNAP microwave toolbox](https://github.com/senbox-org/microwave-toolbox)

## License

GPL-2.0-or-later, see [LICENSE](LICENSE).

## Author

Yann Chemin
