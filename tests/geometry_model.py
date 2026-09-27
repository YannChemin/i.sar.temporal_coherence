"""Synthetic acquisition geometry, independent of the module's C code.

Circular polar orbits in an Earth-fixed frame over longitude 0, heading
north and right-looking (towards +y), one per date, offset across track by
its perpendicular baseline. Orbits are evaluated analytically, the ground
point of a reference pixel is found by a vectorized Newton solution of the
range-Doppler equations, and every date sees it at the range of its own
zero-Doppler time. The SLC phase convention is -4 pi R / lambda.
"""

from datetime import timedelta

import numpy as np

C = 299792458.0
A = 6378137.0
B = 6356752.314245
WAVELENGTH = C / 5.405e9
RANGE_SAMPLING_RATE = 64.34523812571427e6
AZIMUTH_TIME_INTERVAL = 2.055556e-3
NEAR_RANGE = 850e3
RADIUS = A + 700e3
SPEED = 7500.0
LATITUDE = np.radians(25.0)


class Orbit:
    """Circular orbit crossing LATITUDE at time t0 (seconds), baseline y."""

    def __init__(self, t0, baseline):
        self.t0 = t0
        self.baseline = baseline
        self.omega = SPEED / RADIUS

    def state(self, t):
        t = np.asarray(t, dtype=float)
        theta = LATITUDE + self.omega * (t - self.t0)
        pos = np.stack(
            [
                RADIUS * np.cos(theta),
                np.full_like(theta, self.baseline),
                RADIUS * np.sin(theta),
            ],
            axis=-1,
        )
        vel = np.stack(
            [
                -RADIUS * self.omega * np.sin(theta),
                np.zeros_like(theta),
                RADIUS * self.omega * np.cos(theta),
            ],
            axis=-1,
        )
        return pos, vel

    def zero_doppler(self, p, t):
        """Zero-Doppler times of points p (..., 3) from the guesses t."""
        for _it in range(50):
            s, v = self.state(t)
            # Exact derivative of v . (p - s): a . (p - s) - |v|^2, with the
            # centripetal acceleration of the circular orbit.
            acc = -(self.omega**2) * (s - np.array([0.0, self.baseline, 0.0]))
            f = np.sum(v * (p - s), axis=-1)
            df = np.sum(acc * (p - s), axis=-1) - np.sum(v * v, axis=-1)
            step = f / df
            t = t - step
            if np.max(np.abs(step)) < 1e-12:
                break
        return t


def geolocate(orbit, t, rng, h):
    """Ground points (..., 3) at height h seen at times t and ranges rng."""
    t, rng, h = np.broadcast_arrays(
        np.asarray(t, float), np.asarray(rng, float), np.asarray(h, float)
    )
    s, v = orbit.state(t)
    a, b = A + h, B + h
    # Start below the satellite, shifted across track (+y) on the ground.
    rs = np.linalg.norm(s, axis=-1)
    p = s * (a / rs)[..., None]
    p[..., 1] += np.sqrt(np.maximum(rng**2 - (rs - a) ** 2, 0.0))
    for _it in range(100):
        d = p - s
        dist = np.linalg.norm(d, axis=-1)
        f = np.stack(
            [
                np.sum(v * d, axis=-1),
                dist - rng,
                (p[..., 0] ** 2 + p[..., 1] ** 2) / a**2 + p[..., 2] ** 2 / b**2 - 1.0,
            ],
            axis=-1,
        )
        jac = np.stack(
            [
                v,
                d / dist[..., None],
                np.stack(
                    [2 * p[..., 0] / a**2, 2 * p[..., 1] / a**2, 2 * p[..., 2] / b**2],
                    axis=-1,
                ),
            ],
            axis=-2,
        )
        step = np.linalg.solve(jac, -f[..., None])[..., 0]
        p = p + step
        if np.max(np.abs(step)) < 1e-7:
            break
    return p


class Stack:
    """Geometry of a stack: one orbit per date, reference date ref."""

    def __init__(self, first_line_times, baselines, ref, rows, cols, epoch0):
        """first_line_times: datetimes; epoch0: datetime of time 0."""
        self.epoch0 = epoch0
        self.times = [(t - epoch0).total_seconds() for t in first_line_times]
        self.first_line_times = first_line_times
        self.orbits = [
            Orbit(t0 + 0.5 * rows * AZIMUTH_TIME_INTERVAL, b)
            for t0, b in zip(self.times, baselines)
        ]
        self.ref = ref
        self.rows, self.cols = rows, cols
        self.tau0 = 2.0 * NEAR_RANGE / C

    def phases(self, heights):
        """Geometric phase 4 pi (R_k - R_ref) / lambda, shape (N, rows, cols).

        The SLC of date k carries exp(-i phase_k) relative to the reference.
        """
        line, sample = np.mgrid[0 : self.rows, 0 : self.cols].astype(float)
        t = self.times[self.ref] + line * AZIMUTH_TIME_INTERVAL
        rng = 0.5 * C * (self.tau0 + sample / RANGE_SAMPLING_RATE)
        p = geolocate(self.orbits[self.ref], t, rng, heights)
        ranges = []
        for orbit in self.orbits:
            tk = orbit.zero_doppler(p, t + orbit.t0 - self.orbits[self.ref].t0)
            s, _v = orbit.state(tk)
            ranges.append(np.linalg.norm(p - s, axis=-1))
        ranges = np.array(ranges)
        return 4.0 * np.pi / WAVELENGTH * (ranges - ranges[self.ref])

    def state_vectors(self, k, margin=60.0, step=10.0):
        """Annotation-like state vectors of date k around its acquisition."""
        t0 = self.times[k]
        out = []
        for t in np.arange(t0 - margin, t0 + margin + 1e-9, step):
            pos, vel = self.orbits[k].state(t)
            out.append(
                {
                    "time": (self.epoch0 + timedelta(seconds=float(t))).isoformat(
                        timespec="microseconds"
                    ),
                    "frame": "Earth Fixed",
                    "position": [float(x) for x in pos],
                    "velocity": [float(x) for x in vel],
                }
            )
        return out

    def metadata(self, k):
        """Extra metadata sections of date k (r.in.s1slc layout)."""
        start = self.first_line_times[k]
        stop = start + timedelta(seconds=(self.rows - 1) * AZIMUTH_TIME_INTERVAL)
        return {
            "swath": {
                "mission": "S1A",
                "start_time": start.isoformat(timespec="microseconds"),
                "stop_time": stop.isoformat(timespec="microseconds"),
                "wavelength": WAVELENGTH,
                "range_sampling_rate": RANGE_SAMPLING_RATE,
                "terrain_height": 0.0,
                "orbit_state_vectors": self.state_vectors(k),
            },
            "raster_geometry": {
                "first_line_time": start.isoformat(timespec="microseconds"),
                "last_line_time": stop.isoformat(timespec="microseconds"),
                "azimuth_time_interval": AZIMUTH_TIME_INTERVAL,
                "slant_range_time_first_sample": self.tau0,
            },
        }

    def eof(self, k, margin=600.0, step=10.0):
        """Text of a POEORB-like EOF file for date k."""
        t0 = self.times[k]
        osv = []
        for t in np.arange(t0 - margin, t0 + margin + 1e-9, step):
            pos, vel = self.orbits[k].state(t)
            utc = (self.epoch0 + timedelta(seconds=float(t))).isoformat(
                timespec="microseconds"
            )
            osv.append(
                "      <OSV>\n"
                f"        <TAI>TAI={utc}</TAI>\n"
                f"        <UTC>UTC={utc}</UTC>\n"
                f"        <UT1>UT1={utc}</UT1>\n"
                "        <Absolute_Orbit>+46742</Absolute_Orbit>\n"
                f'        <X unit="m">{pos[0]:.6f}</X>\n'
                f'        <Y unit="m">{pos[1]:.6f}</Y>\n'
                f'        <Z unit="m">{pos[2]:.6f}</Z>\n'
                f'        <VX unit="m/s">{vel[0]:.6f}</VX>\n'
                f'        <VY unit="m/s">{vel[1]:.6f}</VY>\n'
                f'        <VZ unit="m/s">{vel[2]:.6f}</VZ>\n'
                "        <Quality>NOMINAL</Quality>\n"
                "      </OSV>\n"
            )
        return (
            '<?xml version="1.0" ?>\n<Earth_Explorer_File>\n  <Data_Block type="xml">\n'
            f'    <List_of_OSVs count="{len(osv)}">\n'
            + "".join(osv)
            + "    </List_of_OSVs>\n  </Data_Block>\n</Earth_Explorer_File>\n"
        )

    def eof_name(self, k, kind="POEORB", margin=600.0):
        t0 = self.epoch0 + timedelta(seconds=self.times[k] - margin + 1)
        t1 = self.epoch0 + timedelta(seconds=self.times[k] + margin - 1)
        fmt = "%Y%m%dT%H%M%S"
        return f"S1A_OPER_AUX_{kind}_OPOD_20990101T000000_V{t0.strftime(fmt)}_{t1.strftime(fmt)}.EOF"
