## DESCRIPTION

*i.sar.temporal_coherence* computes the **temporal coherence** of a
stack of coregistered SAR Single Look Complex (SLC) images of the same
area at different dates. It is the phase-linking quality measure of
distributed-scatterer interferometry (SqueeSAR, FRInGE, MiaplPy,
dolphin): for every pixel it tells how well one consistent phase
history explains all N(N−1)/2 interferograms of the N dates. Values
range from 0 (random phases) to 1 (perfectly consistent), and the
result is the standard mask for choosing which pixels to trust in an
InSAR time series.

The algorithm follows the `PhaseLinking` operator of the ESA SNAP
microwave toolbox. It runs on an **OpenCL** device, preferably a GPU,
one work-item per pixel. For every pixel:

1. **Statistically homogeneous pixels (SHPs)** are selected in a search
    window of **window** lines × samples (default 21×7) centred on the
    pixel. A candidate is kept when a two-sample test at significance
    **alpha** does not reject that its amplitude time series and the one
    of the centre pixel share one distribution: Kolmogorov-Smirnov
    (**shp_test=ks**, default), Anderson-Darling (**shp_test=ad**) or a
    Welch t-test on the log-amplitudes (**shp_test=tlog**, fastest). The
    centre pixel is always one of its SHPs.
2. The SHPs are the looks of the N×N sample covariance matrix
    *C = (1/L) Σ s s<sup>H</sup>* of the stack, normalised to the
    coherence matrix *T<sub>ij</sub> = C<sub>ij</sub> /
    √(C<sub>ii</sub> C<sub>jj</sub>)*. With the **-b** flag the magnitude
    of every coherence is corrected for its bias with the number of looks
    L: *|γ|² ← max(0, (L|γ|² − 1)/(L − 1))*, phase unchanged.
3. One phase per date *φ<sub>n</sub>* is estimated from *T*: the
    dominant eigenvector of *T* (**estimator=evd**, SqueeSAR) or the
    eigenvector of the smallest eigenvalue of *|T|<sup>−1</sup> ∘ T*
    (**estimator=emi**, lower bias when the coherence is low).
4. The temporal coherence is the goodness of fit of these phases to all
    pairs (Pepe and Lanari):
    *γ<sub>T</sub> = 2/(N(N−1)) |Σ<sub>i&lt;j</sub>
    exp(j(arg T<sub>ij</sub> − (φ<sub>i</sub> − φ<sub>j</sub>)))|*.

A pixel is not estimated (NULL in **output**) when one of its dates is
NULL or has a zero sample (SLC no-data), or when it has fewer SHPs than
**min_shp** (default 20). The minimum is never less than the number of
dates, so that the covariance matrix has full rank. With **-b**, a pixel
is also NULL when the zeroed coherences split the dates into groups with
no coherence between them: the relative phase of the groups, hence the
temporal coherence, is then undefined. The optional
**shp_count** map holds the number of SHPs of every pixel whose stack is
valid, including those below **min_shp**; it helps to tune **window**
and **alpha**.

The phases are linked relative to the chronological median date, as
the default reference epoch of SNAP. The result does not depend on the
order of the inputs.

### Input stack

Each date is given by its basename: the raster maps `<basename>_i` and
`<basename>_q` holding the in-phase and quadrature components, as
written by *r.in.s1slc* with **measure=complex** (calibrated or not).
The basename is also the name of the imagery group made by
*r.in.s1slc*, e.g. `s1_20230112_iw2_vv`. At least 3 dates are needed.

The images must be **coregistered** to one common grid (same rows,
columns and extent; see *i.sar.coregistration*): images imported
separately by *r.in.s1slc* are not, since their lines start at different
azimuth times. The module fails when the grids differ. When the
*r.in.s1slc* metadata (`cell_misc/<basename>_i/description.json`) are
present, it also fails if the images differ in sub-swath, polarization,
pass, relative orbit or calibration, or if two of them have the same
acquisition time.

The computation follows the current region, which must have the
resolution of the inputs and be aligned with them, since the SHP window
is counted in pixels: run `g.region raster=<basename>_i` first, then
optionally zoom to the area of interest. The search window is cut at the
region edges.

### Output metadata

The output maps receive a title, a semantic label (e.g.
`S1_VV_TEMPORAL_COHERENCE`, `S1_VV_SHP_COUNT`), the time span of the
stack as timestamp, the command history and a JSON file
`$MAPSET/cell_misc/<map>/description.json` listing the dates, their
products and orbits and the estimation parameters. The temporal
coherence map gets a grey color table from 0 (black) to 1 (white).

## NOTES

Phase linking assumes that the amplitude and phase statistics are
stationary over the window: a larger window gives more looks and a
smoother, higher temporal coherence over homogeneous surfaces, a smaller
one keeps edges between surfaces. **estimator=evd** is the robust
default; **estimator=emi** is better on stacks with strong temporal
decorrelation (vegetation, long time spans). The bias correction (**-b**)
helps EVD with few looks but can degrade EMI.

### OpenCL

The module reads the region in strips of rows bounded by **memory** and
by the largest buffer of the device, and computes every strip on the
OpenCL device. The kernel launches are split into chunks of rows lasting
about half a second each, so that the GPU driver watchdog (e.g. the
amdgpu lockup timeout) never resets a long computation.

By default the first GPU of any OpenCL platform is used, else the first
device (e.g. a CPU through PoCL, with a warning). The **-l** flag lists
the platforms and devices with their indices, to be selected with
**platform** and **device**. The kernel is compiled for the stack at run
time (OpenCL C 1.1, e.g. Mesa Clover) and works in **single precision**:
no double precision support is needed. The results agree with a double
precision implementation to about 10<sup>−5</sup>.

The eigenvector is computed by Householder reduction of the Hermitian
coherence matrix to a real tridiagonal one, bisection of the needed
eigenvalue and inverse iteration. For EMI, |T| is inverted by Cholesky
decomposition, or, when it is not safely positive definite, by the
pseudo-inverse of its Jacobi eigendecomposition.

The cost grows with the window size and with the cube of the number of
dates. With 20 dates and the default window, an AMD Radeon Pro WX 7100
(Mesa Clover) processes about 120 000 pixels per second, i.e. a
sub-swath of three IW bursts (about 37 million pixels) in about five
minutes. At most 64 dates are supported: every work-item keeps six
N×N matrices in private memory.

Some details differ from SNAP (microwave toolbox 2026), where it gives
wrong or arbitrary values: the Anderson-Darling statistic uses the 1/N
normalization of Scholz and Stephens (1987) eq. 6, as
*scipy.stats.anderson_ksamp*, where SNAP scales it by (N−1)/N; a pair
whose coherence is zeroed by the bias correction counts with phase 0 in
*γ<sub>T</sub>*; a date whose eigenvector component is negligible (below
10<sup>−3</sup> of the largest) takes the phase of the reference date;
pixels with unlinked groups of dates are NULL (see above). The critical
value of the t-test is the normal quantile of **alpha**.

The module requires an OpenCL 1.1 driver (ICD) and its development
files (e.g. Debian `ocl-icd-opencl-dev` and `mesa-opencl-icd` or
`pocl-opencl-icd`).

## EXAMPLES

Import the same bursts of five dates over Sharjah (UAE), coregister them
on the first one, then compute their temporal coherence:

```sh
for zip in S1A_IW_SLC__1SDV_2023*.zip ; do
    date=$(echo $zip | cut -c18-25)
    r.in.s1slc input=$zip output=s1_$date swath=IW2 polarization=VV \
        bbox=55.35,25.25,55.50,25.40 measure=complex
done
# Coregistration on the first date (i.sar.coregistration), producing
# s1c_<date>_iw2_vv_i and _q maps on the grid of s1_20230112_iw2_vv.

g.region raster=s1c_20230112_iw2_vv_i
i.sar.temporal_coherence \
    input=s1c_20230112_iw2_vv,s1c_20230124_iw2_vv,s1c_20230205_iw2_vv,s1c_20230217_iw2_vv,s1c_20230301_iw2_vv \
    output=tcoh_vv shp_count=nshp_vv

# Keep the reliable pixels.
r.mapcalc "tcoh_mask = if(tcoh_vv >= 0.7, 1, null())"
```

List the OpenCL devices, then run on the second device of the first
platform:

```sh
i.sar.temporal_coherence -l
i.sar.temporal_coherence input=... output=tcoh_vv platform=0 device=1
```

EMI estimation with an Anderson-Darling SHP test in a larger window:

```sh
i.sar.temporal_coherence input=... output=tcoh_emi \
    estimator=emi shp_test=ad alpha=0.01 window=31,9
```

## REFERENCES

- Ferretti, A., Fumagalli, A., Novali, F., Prati, C., Rocca, F., Rucci,
  A. (2011). A new algorithm for processing interferometric data-stacks:
  SqueeSAR. *IEEE Transactions on Geoscience and Remote Sensing*, 49(9),
  3460–3470.
- Ansari, H., De Zan, F., Bamler, R. (2018). Efficient phase estimation
  for interferogram stacks. *IEEE Transactions on Geoscience and Remote
  Sensing*, 56(7), 4109–4125.
- Pepe, A., Lanari, R. (2006). On the extension of the minimum cost flow
  algorithm for phase unwrapping of multitemporal differential SAR
  interferograms. *IEEE Transactions on Geoscience and Remote Sensing*,
  44(9), 2374–2383.
- Scholz, F. W., Stephens, M. A. (1987). K-sample Anderson-Darling tests.
  *Journal of the American Statistical Association*, 82(399), 918–924.
- ESA SNAP microwave toolbox, `PhaseLinking` operator:
  <https://github.com/senbox-org/microwave-toolbox>

## SEE ALSO

*[i.group](https://grass.osgeo.org/grass-stable/manuals/i.group.html),
[i.sar.coregistration](i.sar.coregistration.html),
[i.sar.interferometry](i.sar.interferometry.html),
[r.in.s1slc](r.in.s1slc.html),
[r.mapcalc](https://grass.osgeo.org/grass-stable/manuals/r.mapcalc.html),
[r.support](https://grass.osgeo.org/grass-stable/manuals/r.support.html)*

## AUTHORS

Yann Chemin
