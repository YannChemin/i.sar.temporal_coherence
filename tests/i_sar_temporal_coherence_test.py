"""Tests of i.sar.temporal_coherence on synthetic coregistered SLC stacks."""

import json
import os
import re
from datetime import timedelta

import numpy as np
import pytest
import reference
from conftest import (
    COLS,
    ROWS,
    T0,
    geometry_stack,
    read_map,
    run_module,
    synthetic_stack,
    write_epoch,
    write_map,
    write_stack,
)

import grass.script as gs

# Largest difference to the double precision reference.
TOLERANCE = 1e-4


@pytest.mark.filterwarnings("ignore:p-value capped")
def test_reference_statistics_match_scipy():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    n = 12
    sigma = reference.ad_sigma(n)
    for _k in range(50):
        a, b = rng.rayleigh(size=n), rng.rayleigh(scale=1.5, size=n)
        expected = stats.anderson_ksamp([a, b], midrank=False).statistic
        assert reference.ad_statistic(list(a), list(b), sigma) == pytest.approx(
            expected, rel=1e-9
        )
        d = stats.ks_2samp(a, b).statistic
        critical = np.sqrt(-0.5 * np.log(0.025)) * np.sqrt(2.0 / n)
        assert reference.ks_accept(list(a), list(b), 0.05) == (d <= critical)
    assert reference.ad_critical(0.05) == pytest.approx(1.961)
    assert reference.ad_critical(0.01) == pytest.approx(3.752)


def reference_for(stack, **kwargs):
    return reference.stack_coherence(stack, **kwargs)


def assert_matches(session, stack, output, shp_count, **kwargs):
    coh_ref, count_ref = reference_for(stack, **kwargs)
    coh = read_map(session, output)
    count = read_map(session, shp_count)
    assert np.array_equal(np.isnan(count), count_ref < 0)
    assert np.array_equal(count[count_ref >= 0], count_ref[count_ref >= 0])
    assert np.array_equal(np.isnan(coh), np.isnan(coh_ref))
    # The kernel works in single precision.
    assert np.allclose(coh[~np.isnan(coh)], coh_ref[~np.isnan(coh_ref)], atol=TOLERANCE)
    return coh


@pytest.mark.parametrize(
    ("options", "is_int"),
    [
        ({}, True),
        ({"shp_test": "ad", "estimator": "emi", "min_shp": 9, "window": (9, 5)}, True),
        ({"shp_test": "tlog", "alpha": 0.01, "min_shp": 9, "window": (7, 5)}, False),
    ],
)
def test_matches_reference(xy_session, options, is_int):
    stack = synthetic_stack(6, 0.7)
    names = write_stack(xy_session, "s", stack, is_int=is_int)
    args = dict(options)
    if "window" in args:
        args["window"] = "{},{}".format(*args["window"])
    proc = run_module(
        xy_session, input=",".join(names), output="coh", shp_count="shp", **args
    )
    assert proc.returncode == 0, proc.stderr
    assert_matches(xy_session, stack, "coh", "shp", **options)


@pytest.mark.parametrize("estimator", ["evd", "emi"])
def test_bias_correction(xy_session, estimator):
    # With few looks the corrected |T| can be indefinite (EMI then falls back
    # from the Cholesky inverse to the pseudo-inverse), and zeroed coherences
    # can split the dates into unlinked groups (undefined, NULL).
    stack = synthetic_stack(6, 0.3, seed=3)
    names = write_stack(xy_session, "s", stack)
    proc = run_module(
        xy_session,
        "b",
        input=",".join(names),
        output="coh",
        shp_count="shp",
        estimator=estimator,
        window="5,3",
        min_shp=6,
    )
    assert proc.returncode == 0, proc.stderr
    assert_matches(
        xy_session,
        stack,
        "coh",
        "shp",
        bias_correction=True,
        estimator=estimator,
        window=(5, 3),
        min_shp=6,
    )
    coh, count = read_map(xy_session, "coh"), read_map(xy_session, "shp")
    assert np.any(np.isnan(coh) & (count >= 6))


def test_strips(xy_session):
    # A tiny memory budget forces several strips.
    stack = synthetic_stack(8, 0.6, rows=30, cols=400, seed=4)
    names = write_stack(xy_session, "s", stack)
    proc = run_module(
        xy_session,
        "-verbose",
        input=",".join(names),
        output="coh",
        shp_count="shp",
        window="7,5",
        min_shp=10,
        memory=1,
    )
    assert proc.returncode == 0, proc.stderr
    strip_rows = int(re.search(r"Strips of (\d+) rows", proc.stderr).group(1))
    assert strip_rows < 30
    assert_matches(xy_session, stack, "coh", "shp", window=(7, 5), min_shp=10)


@pytest.mark.parametrize("estimator", ["evd", "emi"])
def test_coherent_and_incoherent_stacks(xy_session, estimator):
    means = {}
    for g in (0.95, 0.0):
        stack = synthetic_stack(10, g, seed=5)
        names = write_stack(xy_session, "g{}".format(int(g * 100)), stack)
        output = "coh{}".format(int(g * 100))
        proc = run_module(
            xy_session, input=",".join(names), output=output, estimator=estimator
        )
        assert proc.returncode == 0, proc.stderr
        means[g] = np.nanmean(read_map(xy_session, output))
    assert means[0.95] > 0.9
    assert means[0.0] < 0.5


def test_nulls_and_zero_samples(xy_session):
    stack = synthetic_stack(5, 0.8, seed=6)
    stack[2, 10:13, 4:9] = np.nan
    stack[4, 20, 20] = 0
    names = write_stack(xy_session, "s", stack)
    proc = run_module(
        xy_session, input=",".join(names), output="coh", shp_count="shp", min_shp=5
    )
    assert proc.returncode == 0, proc.stderr
    coh = assert_matches(xy_session, stack, "coh", "shp", min_shp=5)
    count = read_map(xy_session, "shp")
    assert np.isnan(coh[10:13, 4:9]).all()
    assert np.isnan(count[10:13, 4:9]).all()
    assert np.isnan(coh[20, 20])
    assert np.isnan(count[20, 20])
    assert np.isfinite(coh[25, 5])


def test_region_subset(xy_session):
    stack = synthetic_stack(5, 0.8, seed=7)
    names = write_stack(xy_session, "s", stack)
    gs.run_command("g.region", n=25, s=5, w=3, e=20, env=xy_session.env)
    proc = run_module(xy_session, input=",".join(names), output="coh", min_shp=5)
    assert proc.returncode == 0, proc.stderr
    info = gs.raster_info("coh", env=xy_session.env)
    assert (int(info["rows"]), int(info["cols"])) == (20, 17)
    # Windows are cut at the region edges, as for any GRASS module.
    expected, _count = reference_for(stack[:, 5:25, 3:20], min_shp=5)
    coh = read_map(xy_session, "coh")
    assert np.allclose(coh, expected, atol=TOLERANCE, equal_nan=True)


def test_metadata(xy_session):
    stack = synthetic_stack(4, 0.8, seed=8)
    names = write_stack(xy_session, "s", stack)
    proc = run_module(
        xy_session,
        input=",".join(reversed(names)),
        output="coh",
        shp_count="shp",
        min_shp=4,
    )
    assert proc.returncode == 0, proc.stderr
    info = gs.raster_info("coh", env=xy_session.env)
    assert info["semantic_label"] == "S1_VV_TEMPORAL_COHERENCE"
    assert (
        info["title"].strip('"') == "Temporal coherence of the 4-date IW1 VV SLC stack"
    )
    assert (
        gs.raster_info("shp", env=xy_session.env)["semantic_label"] == "S1_VV_SHP_COUNT"
    )
    stamp = gs.read_command("r.timestamp", map="coh", env=xy_session.env).strip()
    assert stamp == "12 Jan 2023 14:25:09 / 17 Feb 2023 14:25:11"
    env = gs.gisenv(env=xy_session.env)
    path = os.path.join(
        env["GISDBASE"],
        env["LOCATION_NAME"],
        env["MAPSET"],
        "cell_misc",
        "coh",
        "description.json",
    )
    with open(path) as fd:
        meta = json.load(fd)
    # Dates are sorted chronologically whatever the input order.
    assert [e["input"] for e in meta["epochs"]] == names
    assert meta["stack"]["swath"] == "IW1"
    assert meta["temporal_coherence"]["min_shp"] == 4


def test_missing_metadata_warns(xy_session):
    stack = synthetic_stack(3, 0.8, seed=9)
    names = write_stack(xy_session, "s", stack, metadata=False)
    proc = run_module(xy_session, input=",".join(names), output="coh")
    assert proc.returncode == 0, proc.stderr
    assert "consistency not checked" in " ".join(proc.stderr.split())


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"swath": "IW2"}, "sub-swath"),
        ({"polarization": "VH"}, "polarization"),
        ({"relative_orbit": "57"}, "relative orbit"),
        ({"calibration": "sigma0"}, "calibration"),
    ],
)
def test_inconsistent_stack_fails(xy_session, override, message):
    stack = synthetic_stack(3, 0.8, seed=10)
    names = write_stack(xy_session, "s", stack[:2])
    write_epoch(xy_session, "odd", stack[2], 30, **override)
    proc = run_module(xy_session, input=",".join(names + ["odd"]), output="coh")
    assert proc.returncode != 0
    assert message in " ".join(proc.stderr.split())


def test_same_date_fails(xy_session):
    stack = synthetic_stack(3, 0.8, seed=11)
    names = write_stack(xy_session, "s", stack[:2])
    write_epoch(xy_session, "again", stack[2], 12)
    proc = run_module(xy_session, input=",".join(names + ["again"]), output="coh")
    assert proc.returncode != 0
    assert "same acquisition time" in " ".join(proc.stderr.split())


def test_grid_mismatch_fails(xy_session):
    stack = synthetic_stack(3, 0.8, seed=12)
    names = write_stack(xy_session, "s", stack[:2])
    write_epoch(xy_session, "small", stack[2][:, :-1], 30)
    proc = run_module(xy_session, input=",".join(names + ["small"]), output="coh")
    assert proc.returncode != 0
    assert "coregistered" in " ".join(proc.stderr.split())


def test_region_resolution_fails(xy_session):
    names = write_stack(xy_session, "s", synthetic_stack(3, 0.8, seed=13))
    gs.run_command("g.region", res=2, env=xy_session.env)
    proc = run_module(xy_session, input=",".join(names), output="coh")
    assert proc.returncode != 0
    assert "resolution" in " ".join(proc.stderr.split())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window": "20,7"}, "odd"),
        ({"window": "3,3", "min_shp": 20}, "fewer pixels"),
        ({"shp_test": "ad", "alpha": 0.5}, "alpha"),
        ({"input_count": 2}, "at least 3"),
        ({"missing": True}, "not found"),
    ],
)
def test_option_failures(xy_session, kwargs, message):
    names = write_stack(xy_session, "s", synthetic_stack(3, 0.8, seed=14))
    kwargs = dict(kwargs)
    if kwargs.pop("input_count", None):
        names = names[:2]
    if kwargs.pop("missing", None):
        names.append("nothing")
    proc = run_module(xy_session, input=",".join(names), output="coh", **kwargs)
    assert proc.returncode != 0
    assert message in " ".join(proc.stderr.split())


def pair_list(n, mode):
    if mode == "all":
        return [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
    return [(i, i + 1) for i in range(n - 1)]


def pair_name(prefix, i, j):
    day = [(T0 + timedelta(days=12 * k)).strftime("%Y%m%d") for k in (i, j)]
    return "{}_{}_{}".format(prefix, *day)


def assert_pairs(session, prefix, mode, pairs_ref, tolerance=TOLERANCE):
    n = pairs_ref.shape[0]
    for i, j in pair_list(n, mode):
        got = read_map(session, pair_name(prefix, i, j))
        want = pairs_ref[i, j]
        assert np.array_equal(np.isnan(got), np.isnan(want))
        assert np.allclose(got[~np.isnan(got)], want[~np.isnan(want)], atol=tolerance)


@pytest.mark.parametrize("mode", ["consecutive", "all"])
def test_pairs(xy_session, mode):
    stack = synthetic_stack(5, 0.7, seed=20)
    names = write_stack(xy_session, "s", stack)
    proc = run_module(
        xy_session,
        input=",".join(names),
        output="coh",
        pairs="p",
        pairs_mode=mode,
        min_shp=5,
    )
    assert proc.returncode == 0, proc.stderr
    coh_ref, _count, pairs_ref = reference.stack_coherence(
        stack, min_shp=5, with_pairs=True
    )
    assert_pairs(xy_session, "p", mode, pairs_ref)
    maps = gs.list_strings("raster", pattern="p_*", env=xy_session.env)
    assert len(maps) == len(pair_list(5, mode))
    name = pair_name("p", 1, 2)
    info = gs.raster_info(name, env=xy_session.env)
    assert info["semantic_label"] == "S1_VV_COHERENCE"
    stamp = gs.read_command("r.timestamp", map=name, env=xy_session.env).strip()
    assert stamp == "24 Jan 2023 14:25:09 / 5 Feb 2023 14:25:11"
    env = gs.gisenv(env=xy_session.env)
    path = os.path.join(
        env["GISDBASE"],
        env["LOCATION_NAME"],
        env["MAPSET"],
        "cell_misc",
        name,
        "description.json",
    )
    with open(path) as fd:
        meta = json.load(fd)
    assert [e["input"] for e in meta["epochs"]] == names[1:3]
    assert meta["temporal_coherence"]["dates"] == 5


def test_pairs_defined_when_dates_unlinked(xy_session):
    # Few looks with bias correction: the temporal coherence is NULL where
    # zeroed coherences unlink the dates, the pair coherences are not.
    stack = synthetic_stack(6, 0.3, seed=3)
    names = write_stack(xy_session, "s", stack)
    proc = run_module(
        xy_session,
        "b",
        input=",".join(names),
        output="coh",
        pairs="p",
        window="5,3",
        min_shp=6,
    )
    assert proc.returncode == 0, proc.stderr
    _coh, count, pairs_ref = reference.stack_coherence(
        stack, window=(5, 3), min_shp=6, bias_correction=True, with_pairs=True
    )
    assert_pairs(xy_session, "p", "consecutive", pairs_ref)
    coh = read_map(xy_session, "coh")
    first = read_map(xy_session, pair_name("p", 0, 1))
    assert np.any(np.isnan(coh) & np.isfinite(first))


def ramped_stack(geometry, heights, coherence=0.8, seed=21):
    """Stack carrying the geometric phase of the synthetic orbits, and it."""
    n = len(geometry.orbits)
    x = synthetic_stack(n, coherence, rows=geometry.rows, cols=geometry.cols, seed=seed)
    phase = geometry.phases(heights)
    return np.round(x * np.exp(-1j * phase)), phase


@pytest.mark.parametrize("ref", [0, 2])
def test_flat_earth_phase(xy_session, ref):
    geometry = geometry_stack(6, ref=ref)
    stack, phase = ramped_stack(geometry, np.zeros((ROWS, COLS)))
    names = write_stack(xy_session, "s", stack, geometry=geometry)
    common = {"input": ",".join(names), "pairs_mode": "all", "min_shp": 6}
    proc = run_module(
        xy_session,
        "f",
        output="coh",
        pairs="p",
        reference=names[ref],
        orbit="annotation",
        **common,
    )
    assert proc.returncode == 0, proc.stderr
    coh_ref, _count, pairs_ref = reference.stack_coherence(
        stack, min_shp=6, with_pairs=True, phase=phase
    )
    coh = read_map(xy_session, "coh")
    assert np.allclose(coh, coh_ref, atol=TOLERANCE, equal_nan=True)
    assert_pairs(xy_session, "p", "all", pairs_ref)
    meta_path = os.path.join(
        gs.gisenv(env=xy_session.env)["GISDBASE"],
        gs.gisenv(env=xy_session.env)["LOCATION_NAME"],
        "PERMANENT",
        "cell_misc",
        "coh",
        "description.json",
    )
    with open(meta_path) as fd:
        params = json.load(fd)["temporal_coherence"]
    assert (params["phase_reference"], params["orbit"]) == ("ellipsoid", "annotation")

    # Without the correction, the fringes decorrelate the windows.
    proc = run_module(xy_session, output="raw", pairs="r", **common)
    assert proc.returncode == 0, proc.stderr
    flat = read_map(xy_session, pair_name("p", 2, 3))
    raw = read_map(xy_session, pair_name("r", 2, 3))
    assert np.nanmean(flat) > np.nanmean(raw) + 0.1


def test_topographic_phase(xy_session):
    geometry = geometry_stack(6)
    y, x = np.mgrid[0:ROWS, 0:COLS]
    hill = 800.0 * np.exp(-((y - 15) ** 2 / 100.0 + (x - 12) ** 2 / 50.0))
    stack, phase = ramped_stack(geometry, hill)
    names = write_stack(xy_session, "s", stack, geometry=geometry)
    write_map(xy_session, "dem", hill, is_int=False)
    common = {"input": ",".join(names), "reference": names[0], "orbit": "annotation"}
    proc = run_module(
        xy_session, output="coh", elevation="dem", pairs="p", min_shp=6, **common
    )
    assert proc.returncode == 0, proc.stderr
    coh_ref, _count, pairs_ref = reference.stack_coherence(
        stack, min_shp=6, with_pairs=True, phase=phase
    )
    assert np.allclose(
        read_map(xy_session, "coh"), coh_ref, atol=TOLERANCE, equal_nan=True
    )
    assert_pairs(xy_session, "p", "consecutive", pairs_ref)

    # The flat-earth phase alone leaves the topographic fringes.
    proc = run_module(xy_session, "f", output="flat", pairs="f", min_shp=6, **common)
    assert proc.returncode == 0, proc.stderr
    hilltop = (slice(10, 21), slice(8, 17))
    assert (
        np.nanmean(read_map(xy_session, "coh")[hilltop])
        > np.nanmean(read_map(xy_session, "flat")[hilltop]) + 0.1
    )


def test_orbit_files(xy_session, tmp_path):
    geometry = geometry_stack(4)
    stack, _phase = ramped_stack(geometry, np.zeros((ROWS, COLS)))
    names = write_stack(xy_session, "s", stack, geometry=geometry)
    orbit_dir = tmp_path / "orbits"
    orbit_dir.mkdir()
    for k in range(4):
        (orbit_dir / geometry.eof_name(k)).write_text(geometry.eof(k))
    common = {"input": ",".join(names), "reference": names[0], "min_shp": 4}
    proc = run_module(xy_session, "f", output="eof", orbit_dir=orbit_dir, **common)
    assert proc.returncode == 0, proc.stderr
    proc = run_module(xy_session, "f", output="ann", orbit="annotation", **common)
    assert proc.returncode == 0, proc.stderr
    assert np.allclose(
        read_map(xy_session, "eof"),
        read_map(xy_session, "ann"),
        atol=1e-6,
        equal_nan=True,
    )


def step_reachable():
    import socket

    try:
        socket.create_connection(("step.esa.int", 80), timeout=5).close()
    except OSError:
        return False
    return True


@pytest.mark.skipif(not step_reachable(), reason="ESA STEP orbit mirror not reachable")
def test_orbit_download(xy_session, tmp_path):
    # Real precise orbits of S1A for the synthetic dates (January 2023).
    geometry = geometry_stack(3)
    stack, _phase = ramped_stack(geometry, np.zeros((ROWS, COLS)))
    names = write_stack(xy_session, "s", stack, geometry=geometry)
    orbit_dir = tmp_path / "orbits"
    proc = run_module(
        xy_session,
        "f",
        input=",".join(names),
        output="coh",
        reference=names[0],
        orbit_dir=orbit_dir,
        min_shp=3,
    )
    assert proc.returncode == 0, proc.stderr
    files = sorted(p.name for p in orbit_dir.iterdir())
    assert len(files) == 3
    assert all(
        f.startswith("S1A_OPER_AUX_POEORB_OPOD_") and f.endswith(".EOF") for f in files
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"flags": "f"}, "reference is required"),
        ({"flags": "f", "reference": "other"}, "not one of the inputs"),
        (
            {"flags": "f", "reference": 0, "orbit": "annotation", "geometry": False},
            "No orbit state vectors",
        ),
        ({"flags": "f", "reference": 0, "mission": "S1X"}, "No POEORB orbit file"),
    ],
)
def test_geometry_failures(xy_session, tmp_path, kwargs, message):
    kwargs = dict(kwargs)
    geometry = geometry_stack(3)
    stack, _phase = ramped_stack(geometry, np.zeros((ROWS, COLS)))
    use_geometry = kwargs.pop("geometry", True)
    mission = kwargs.pop("mission", None)
    names = []
    for k in range(3):
        extra = geometry.metadata(k) if use_geometry else None
        if mission:
            extra["swath"]["mission"] = mission
        name = "s_{:02d}_iw1_vv".format(k)
        write_epoch(xy_session, name, stack[k], 12 * k, extra=extra)
        names.append(name)
    gs.run_command("g.region", raster=names[0] + "_i", env=xy_session.env)
    flags = kwargs.pop("flags")
    if kwargs.get("reference") == 0:
        kwargs["reference"] = names[0]
    proc = run_module(
        xy_session,
        flags,
        input=",".join(names),
        output="coh",
        min_shp=3,
        orbit_dir=tmp_path / "empty",
        **kwargs,
    )
    assert proc.returncode != 0
    assert message in " ".join(proc.stderr.split())
