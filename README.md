# i.sar.temporal_coherence

A [GRASS GIS](https://grass.osgeo.org/) addon that computes the
**phase-linking temporal coherence** of a coregistered stack of SAR
Single Look Complex (SLC) images, such as Sentinel-1 IW/EW SLC imported
by [r.in.s1slc](https://github.com/YannChemin/r.in.s1slc).

```sh
g.region raster=s1c_20230112_iw2_vv_i
i.sar.temporal_coherence \
    input=s1c_20230112_iw2_vv,s1c_20230124_iw2_vv,s1c_20230205_iw2_vv \
    output=tcoh_vv shp_count=nshp_vv nprocs=4
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
`min_shp` (≥ N) SHPs, are NULL. The computation is vectorized with
NumPy, bounded by `memory` and parallelized over `nprocs` threads.

## Input

Each date is the basename of an *r.in.s1slc* complex pair
(`<basename>_i`, `<basename>_q`), i.e. its imagery group name. The dates
must be **coregistered** to one grid (a separate step,
*i.sar.coregistration*); the module fails on differing grids and, from
the *r.in.s1slc* metadata, on differing sub-swath, polarization, pass,
relative orbit, calibration or duplicate dates.

## Requirements

- GRASS GIS 8.4 or later
- Python 3 with NumPy

## Installation

```sh
g.extension extension=i.sar.temporal_coherence url=https://github.com/YannChemin/i.sar.temporal_coherence
```

or from a source tree:

```sh
make MODULE_TOPDIR=$HOME/dev/grass
```

## Tests

The tests compare the module, pixel by pixel, with a plain-loop
transcription of the SNAP algorithm on synthetic distributed-scatterer
stacks, check the SHP statistics against SciPy, and cover NULL handling,
region subsets, strip and block seams, metadata and failure modes.

```sh
grass --tmp-project XY --exec python3 -m pytest tests
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
