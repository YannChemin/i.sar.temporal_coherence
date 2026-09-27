"""Fixtures for i.sar.temporal_coherence tests: synthetic coregistered SLC stacks.

A stack follows the distributed scatterer model: at each pixel
s_n = sqrt(g) z0 exp(j phi_n) + sqrt(1 - g) z_n with circular Gaussian
z0, z_n and a per-date phase phi_n smooth in space, so that g close to 1
gives a high and g = 0 a low temporal coherence. Every date is written as
the r.in.s1slc complex pair <basename>_i, <basename>_q with its
description.json metadata.
"""

import shutil
import json
import os
import subprocess
from datetime import datetime, timedelta

import numpy as np
import pytest

import grass.script as gs

MODULE = "i.sar.temporal_coherence"
T0 = datetime(2023, 1, 12, 14, 25, 9)
ROWS = 30
COLS = 24


def synthetic_stack(n, coherence, rows=ROWS, cols=COLS, seed=0):
    """Complex stack (n, rows, cols) of the distributed scatterer model."""
    rng = np.random.default_rng(seed)

    def gaussian(*shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(
            2
        )

    y, x = np.mgrid[0:rows, 0:cols]
    phase = np.stack([0.7 * k + 0.05 * k * (x + 0.5 * y) for k in range(n)])
    common = gaussian(rows, cols)
    stack = np.sqrt(coherence) * common * np.exp(1j * phase) + np.sqrt(
        1 - coherence
    ) * gaussian(n, rows, cols)
    # Scaled to 16-bit digital numbers like uncalibrated SLC samples.
    return np.round(stack * 300.0)


def write_map(session, name, array, is_int):
    path = session.tmp_dir / (name + ".bin")
    rows, cols = array.shape
    if is_int:
        data = np.where(np.isfinite(array), array, -2147483648).astype(np.int32)
    else:
        data = array.astype(np.float32)
    data.tofile(path)
    gs.run_command(
        "r.in.bin",
        flags="s" if is_int else "f",
        input=path,
        output=name,
        bytes=4,
        north=rows,
        south=0,
        east=cols,
        west=0,
        rows=rows,
        cols=cols,
        anull=-2147483648 if is_int else "nan",
        env=session.env,
    )


def write_epoch(session, basename, slc, day, metadata=True, is_int=True, **overrides):
    """Write one date as <basename>_i/_q with r.in.s1slc-like metadata."""
    write_map(session, basename + "_i", slc.real, is_int)
    write_map(session, basename + "_q", slc.imag, is_int)
    if not metadata:
        return
    start = T0 + timedelta(days=day)
    meta = {
        "product": {
            "product_name": "S1A_TEST_{}.SAFE".format(day),
            "relative_orbit_start": overrides.get("relative_orbit", "130"),
            "absolute_orbit_start": str(46752 + day),
        },
        "swath": {
            "mission": "S1A",
            "swath": overrides.get("swath", "IW1"),
            "polarization": overrides.get("polarization", "VV"),
            "pass": "ASCENDING",
        },
        "raster_geometry": {
            "rows": slc.shape[0],
            "cols": slc.shape[1],
            "first_line_time": start.isoformat(),
            "last_line_time": (start + timedelta(seconds=1.5)).isoformat(),
        },
        "measure": "i",
        "calibration": overrides.get("calibration", "none"),
        "absolute_calibration_constant": None,
        "thermal_noise_removed": False,
    }
    env = gs.gisenv(env=session.env)
    meta_dir = os.path.join(
        env["GISDBASE"],
        env["LOCATION_NAME"],
        env["MAPSET"],
        "cell_misc",
        basename + "_i",
    )
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "description.json"), "w") as fd:
        json.dump(meta, fd)


def write_stack(session, prefix, stack, **kwargs):
    """Write every date of the stack, 12 days apart; return the basenames."""
    names = []
    for k, slc in enumerate(stack):
        name = "{}_{:02d}_iw1_vv".format(prefix, k)
        write_epoch(session, name, slc, 12 * k, **kwargs)
        names.append(name)
    gs.run_command("g.region", raster=names[0] + "_i", env=session.env)
    return names


class Session:
    def __init__(self, session, tmp_dir):
        self.env = session.env
        self.tmp_dir = tmp_dir


@pytest.fixture
def xy_session(tmp_path):
    project = tmp_path / "xy"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        yield Session(session, tmp_path)


def module_binary(session):
    """Compiled module: $I_SAR_TEMPORAL_COHERENCE, else found on the PATH."""
    path = os.environ.get("I_SAR_TEMPORAL_COHERENCE") or shutil.which(
        MODULE, path=session.env.get("PATH")
    )
    if not path:
        pytest.fail(
            "{} not found: build it with make MODULE_TOPDIR=... and run the "
            "tests with that GRASS, or set I_SAR_TEMPORAL_COHERENCE".format(MODULE)
        )
    return path


def run_module(session, *flags, **kwargs):
    """Run the compiled module; return the completed process."""
    args = [module_binary(session)]
    args += ["-" + f for f in flags]
    args += ["{}={}".format(k, v) for k, v in kwargs.items()]
    return subprocess.run(args, env=session.env, capture_output=True, text=True, check=False)


def read_map(session, name):
    import grass.script.array as garray

    data = np.asarray(garray.array(name, null="nan", env=session.env), dtype=np.float64)
    # r.out.bin cannot write NaN into integer data: CELL nulls keep their value.
    data[data == -2147483648] = np.nan
    return data
