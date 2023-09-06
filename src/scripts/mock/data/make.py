"""Simulate mock stream."""

import os
from pathlib import Path

import asdf
import astropy.units as u
import brutus.filters
import brutus.seds
import numpy as np
import shapely
from astropy.coordinates import Distance
from astropy.table import QTable, vstack
from scipy import stats
from shapely.affinity import affine_transform
from shapely.geometry import Point, Polygon
from shapely.ops import triangulate
from showyourwork.paths import user as Paths

import stream_ml.pytorch as sml

paths = Paths()

##############################################################################
# Parameters

try:
    snkmk = snakemake.params
except NameError:
    snkmk = {"seed": 10, "diagnostic_plots": True}

seed: int = snkmk["seed"]
rng = np.random.default_rng(seed)

N_BACKGROUND = 13_000
N_STREAM_MAX = 2_000  # going to be less b/c of gaps
stop_stream_phot = -1_000  # Limit the isochrone.

filters = brutus.filters.ps[:-2]  # g, r
nnfile = (paths.static / "brutus" / "nn_c3k.h5").resolve()
mistfile = (paths.static / "brutus" / "MIST_1.2_iso_vvcrit0.0.h5").resolve()

##############################################################################


def _sample_mags_in_iso_polygon(
    polygon: Polygon, n_pnts: int, *, rng: np.random.Generator
) -> np.ndarray:
    """Return list of k points chosen uniformly at random inside the polygon.

    From https://codereview.stackexchange.com/questions/69833/generate-sample-coordinates-inside-a-polygon
    """
    triangles = triangulate(polygon)
    areas = np.zeros(len(triangles))
    transforms = np.zeros((len(triangles), 6))
    for i, t in enumerate(triangles):
        areas[i] = t.area
        (x0, y0), (x1, y1), (x2, y2), _ = t.exterior.coords
        transforms[i, :] = [x1 - x0, x2 - x0, y2 - y0, y1 - y0, x0, y0]
    areas /= np.sum(areas)
    points = []

    # for transform in random.choices(transforms, weights=areas, k=n_pnts):
    for transform in rng.choice(transforms, p=areas, size=n_pnts, replace=True, axis=0):
        x, y = (rng.random() for _ in range(2))
        p = Point(1 - x, 1 - y) if x + y > 1 else Point(x, y)
        points.append(affine_transform(p, transform).xy)
    return np.array(points)


def sample_magnitudes_from_isochrone(
    abs_mags: u.Quantity,
    distmod: u.Quantity,
    n_pnts: int = 10_000,
    *,
    rng: np.random.Generator,
) -> u.Quantity:
    """Sample magnitudes from isochrone."""
    # TODO! better generation by parameterizing the isochrone by gamma in [0, 1]
    # then sample gamma to get the mean location on the isochrone
    # then add random noise with some covariance.
    iso_shape = shapely.LineString(np.c_[abs_mags[:, 0], abs_mags[:, 1]])  # g, r
    iso = iso_shape.buffer(0.2)
    mags = _sample_mags_in_iso_polygon(iso, n_pnts, rng=rng)[..., 0]
    mags = mags[shapely.contains_xy(iso, mags[:, 0], mags[:, 1])][: len(distmod)]
    return (mags * u.mag) + distmod


##############################################################################

names = ("phi1", "phi2", "distance", "distmod", "parallax", "g", "r", "g_err", "r_err")
stream_tbl = QTable(
    np.empty((N_STREAM_MAX, 9)),
    names=names,
    units=(u.deg, u.deg, u.pc, u.mag, u.mas, u.mag, u.mag, u.mag, u.mag),
    dtype=(float, float, float, float, float, float, float, float, float),
)

# ---- Astrometrics ----

stream_tbl["phi1"] = u.Quantity(rng.uniform(-20, 30, N_STREAM_MAX), u.deg)
stream_tbl["phi2"] = u.Quantity(
    (stream_tbl["phi1"].value ** 2 + rng.normal(0, 20, size=N_STREAM_MAX)) / 130,
    u.deg,
)

_p1unord = np.argsort(np.argsort(stream_tbl["phi1"]))
stream_tbl["distance"] = u.Quantity(
    (np.linspace(7, 15, N_STREAM_MAX)[_p1unord] + rng.normal(0, 0.25, N_STREAM_MAX))
    * 1000,
    u.pc,
)
stream_tbl["distmod"] = Distance(stream_tbl["distance"]).distmod
stream_tbl["parallax"] = stream_tbl["distance"].to(u.mas, equivalencies=u.parallax())

# ---- Photometrics ----

isochrone = brutus.seds.Isochrone(filters=filters, nnfile=nnfile, mistfile=mistfile)
isochrone_age = 12 * u.Gyr
isochrone_feh = -1.35
abs_mags_, *_ = isochrone.get_seds(
    eep=np.linspace(202, 600, 5000),  # EEP grid: MS to RGB
    apply_corr=True,
    feh=isochrone_feh,
    dist=10,
    loga=np.log10(isochrone_age.to_value(u.yr)),
)
stream_abs_mags = u.Quantity(abs_mags_[np.all(np.isfinite(abs_mags_), axis=1)], u.mag)
stream_cmd = sample_magnitudes_from_isochrone(
    stream_abs_mags[:stop_stream_phot, :], stream_tbl["distmod"][:, None], rng=rng
)
stream_tbl["g"] = stream_cmd[:, 0]
stream_tbl["g_err"] = 0 * u.mag
stream_tbl["r"] = stream_cmd[:, 1]
stream_tbl["r_err"] = 0 * u.mag

# ---- spatial variation ----

num_gaps = rng.integers(1, 4)
gap_centers = rng.uniform(-20, 30, num_gaps)  # [deg]
gap_widths = rng.uniform(2, 5, num_gaps)  # [deg]
gap_depths = rng.uniform(0.5, 1, num_gaps)  # [counts]

keep = np.ones(len(stream_tbl), dtype=bool)
for center, width, depth in zip(gap_centers, gap_widths, gap_depths, strict=True):
    distr = stats.norm(loc=center, scale=width)
    removes = distr.pdf(stream_tbl["phi1"].value) >= depth * distr.pdf(center)
    keep &= ~removes

stream_tbl = stream_tbl[keep]

# -------------------------------------------
# Background

bkg_tbl = QTable(
    np.empty((N_BACKGROUND, 9)),
    names=names,
    units=(u.deg, u.deg, u.pc, u.mag, u.mas, u.mag, u.mag, u.mag, u.mag),
    dtype=(float, float, float, float, float, float, float, float, float),
)

# ---- Astrometrics ----

bkg_tbl["phi1"] = u.Quantity(rng.uniform(-45, 50, size=N_BACKGROUND), u.deg)
bkg_tbl["phi2"] = u.Quantity(rng.uniform(-2, 10, size=N_BACKGROUND), u.deg)
bkg_tbl["distance"] = u.Quantity(rng.uniform(5, 17, N_BACKGROUND) * 1000, u.pc)
bkg_tbl["distmod"] = Distance(bkg_tbl["distance"]).distmod
bkg_tbl["parallax"] = bkg_tbl["distance"].to(u.mas, equivalencies=u.parallax())

# ---- Photometrics ----

root_cov = np.array([[1, 4], [4, 1]])  # entirely made up
cov_background = np.matmul(root_cov, root_cov.T)
background_cmd = rng.multivariate_normal(
    mean=np.array([20, 13]), cov=cov_background, size=N_BACKGROUND
)
bkg_tbl["g"] = u.Quantity(background_cmd[:, 0], u.mag)
bkg_tbl["g_err"] = (
    np.abs(np.asarray(rng.normal(0, scale=0.05, size=N_BACKGROUND))) * u.mag
)
bkg_tbl["r"] = u.Quantity(background_cmd[:, 1], u.mag)
bkg_tbl["r_err"] = (
    np.abs(np.asarray(rng.normal(0, scale=0.05, size=N_BACKGROUND))) * u.mag
)

# -------------------------------------------
# Together

# Combine tables
tot_table = vstack((stream_tbl, bkg_tbl))
tot_table["label"] = ["stream"] * len(stream_tbl) + ["background"] * len(bkg_tbl)

# Add g-r color
tot_table["g-r"] = tot_table["g"] - tot_table["r"]
tot_table["g-r_err"] = np.sqrt(tot_table["g_err"] ** 2 + tot_table["r_err"] ** 2)

# sort by phi1
tot_table.sort("phi1")

names = list(tot_table.colnames)
names.remove("label")
data = sml.Data.from_format(tot_table.as_array(names=names), fmt="numpy.structured")

# Fit scaling
scaler = sml.utils.StandardScaler.fit(data, names=data.names)

# -----------------------------------------------------------------------------

af = asdf.AsdfFile()
af["table"] = tot_table

# stream
af["n_stream"] = keep.sum()
af["stream_table"] = tot_table[tot_table["label"] == "stream"]
af["stream_abs_mags"] = stream_abs_mags
af["isochrone_age"] = isochrone_age
af["isochrone_feh"] = isochrone_feh

# background
af["n_background"] = N_BACKGROUND
af["bkg_table"] = tot_table[tot_table["label"] == "background"]

# off-stream selection
af["off_stream"] = (
    (data["phi2"] < -1)
    | (data["phi2"] > 7)
    | (data["phi1"] < -20)
    | (data["phi1"] > 30)
)

af["data"] = {"array": data.array, "names": data.names}
af["where"] = {"array": ~np.isnan(data.array), "names": data.names}
af["scaler"] = {"mean": scaler.mean, "scale": scaler.scale, "names": scaler.names}
af["coord_bounds"] = {
    k: (float(np.nanmin(data[k])), float(np.nanmax(data[k]))) for k in data.names
}

af.write_to(paths.data / "mock" / "data.asdf", all_array_storage="internal")
af.close()


# ----------
# Diagnostics

if snkmk["diagnostic_plots"]:
    nbpath = (Path(__file__).parent / "diagnostics.ipynb").as_posix()
    os.system(f"jupyter execute {nbpath}")  # noqa: S605
