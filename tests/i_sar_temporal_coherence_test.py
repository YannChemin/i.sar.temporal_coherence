"""Tests of i.sar.temporal_coherence on synthetic coregistered SLC stacks."""

import json
import os

import numpy as np
import pytest
import reference
from conftest import (
    read_map,
    run_module,
    synthetic_stack,
    write_epoch,
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
    assert "Strips of 20 rows" in " ".join(proc.stderr.split())
    assert_matches(xy_session, stack, "coh", "shp", window=(7, 5), min_shp=10)


@pytest.mark.parametrize("estimator", ["evd", "emi"])
def test_coherent_and_incoherent_stacks(xy_session, estimator):
    means = {}
    for g in (0.95, 0.0):
        stack = synthetic_stack(10, g, seed=5)
        names = write_stack(xy_session, "g{}".format(int(g * 100)), stack)
        output = "coh{}".format(int(g * 100))
        proc = run_module(xy_session, input=",".join(names), output=output, estimator=estimator)
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
