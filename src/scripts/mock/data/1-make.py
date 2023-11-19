"""Simulate mock stream."""

import sys
from dataclasses import asdict
from typing import Any

import asdf
import astropy.units as u
import brutus.filters
import brutus.seds
import numpy as np
from astropy.coordinates import Distance
from astropy.table import QTable, vstack
from scipy import stats
from scipy.interpolate import CubicSpline
from showyourwork.paths import user as user_paths

import stream_ml.pytorch as sml

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.helper import isochrone_spline

##############################################################################
# Parameters

try:
    snkmk = snakemake.params
except NameError:
    # snkmk = {"seed": 9, "diagnostic_plots": True}
    snkmk = {"seed": 35, "diagnostic_plots": True}

seed: int = snkmk["seed"]
rng = np.random.default_rng(seed)

N_BACKGROUND = 13_000
N_STREAM_MAX = 2_000  # going to be less b/c of gaps
stop_stream_phot = -1_000  # Limit the isochrone.

filters = brutus.filters.ps[:-2]  # g, r
nnfile = (paths.static / "brutus" / "nn_c3k.h5").resolve()
mistfile = (paths.static / "brutus" / "MIST_1.2_iso_vvcrit0.0.h5").resolve()

##############################################################################


class Pal5MF(stats.rv_continuous):
    r"""Palomar 5 mass function.

    See https://iopscience.iop.org/article/10.1086/323916/pdf.

    .. math::

        \frac{dN}{dm} = m^{-1/2}
    """

    def __init__(self, mmin: float, mmax: float, **kwargs: Any) -> None:
        self.mmin = mmin
        self.mmax = mmax
        super().__init__(a=mmin, b=mmax, **kwargs)

    @property
    def _pdf_normalization(self) -> float:
        """Normalization for the PDF."""
        return 2 * (np.sqrt(self.mmax) - np.sqrt(self.mmin))

    def _pdf(self, mass: float) -> float:
        """Return the PDF."""
        dNdm = 1 / np.sqrt(mass)
        return dNdm / self._pdf_normalization

    def _cdf(self, mass: float) -> float:
        r"""Return the CDF(mass)."""
        return 2 * (np.sqrt(mass) - np.sqrt(self.mmin)) / self._pdf_normalization

    def _ppf(self, q: float) -> float:
        """Return the inverse CDF(q)."""
        return (q * self._pdf_normalization / 2 + np.sqrt(self.mmin)) ** 2


##############################################################################
# Stream

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

# Define the isochrone
isochrone = brutus.seds.Isochrone(filters=filters, nnfile=nnfile, mistfile=mistfile)
isochrone_age = 12 * u.Gyr
isochrone_feh = -1.35

# Sample the isochrone
abs_mags_, info1, _ = isochrone.get_seds(
    eep=np.linspace(202, 600, 5000),  # EEP grid: MS to RGB
    apply_corr=True,
    feh=isochrone_feh,
    dist=10,
    loga=np.log10(isochrone_age.to_value(u.yr)),
)

# Find the finite values
select_finite = np.all(np.isfinite(abs_mags_), axis=1)

# Get the mass
mass = info1["mass"][select_finite]
# And limit to the region of monotonicity
select_monotonic = slice(None, np.argmax(mass) + 1)
mass = mass[select_monotonic]

# Get the absolute magnitudes within the select region
stream_abs_mags = u.Quantity(abs_mags_[select_finite][select_monotonic][:, :2], u.mag)

# Mass range
mmin, mmax = mass.min(), mass.max()

# Cubic spline the isochrone as a function of gamma
mag_spline = isochrone_spline(stream_abs_mags.value, xp=np)
# Get the gamma values
gamma = mag_spline.x

# mass(gamma) spline
gamma_spline = CubicSpline(mass, gamma)

# Mass function distribution
p5mf = Pal5MF(mmin, mmax)

# Sample the mass function
mass_samples = p5mf.rvs(size=N_STREAM_MAX, random_state=rng)
gamma_samples = gamma_spline(mass_samples)

stream_cmd = (
    mag_spline(gamma_samples)
    + stats.norm.rvs(size=(len(mass_samples), 2), scale=0.2, random_state=rng)
) * u.mag
# Boost to the distance modulus
stream_cmd += stream_tbl["distmod"][:, None]

# Add to the table
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

##############################################################################
# Save

af = asdf.AsdfFile()
af["table"] = tot_table

# stream
af["n_stream"] = keep.sum()
af["stream_table"] = tot_table[tot_table["label"] == "stream"]
af["stream_abs_mags"] = stream_abs_mags
af["isochrone_age"] = isochrone_age
af["isochrone_feh"] = isochrone_feh
af["gamma_mass"] = np.c_[gamma, mass]

# background
af["n_background"] = N_BACKGROUND
af["bkg_table"] = tot_table[tot_table["label"] == "background"]

# off-stream selection
af["off_stream"] = (
    (data["phi2"] < stream_tbl["phi2"].value.min())
    | (data["phi2"] > stream_tbl["phi2"].value.max())
    | (data["phi1"] < stream_tbl["phi1"].value.min())
    | (data["phi1"] > stream_tbl["phi1"].value.max())
)

af["data"] = {"array": data.array, "names": data.names}
af["where"] = {"array": ~np.isnan(data.array), "names": data.names}
af["scaler"] = asdict(scaler)
af["coord_bounds"] = {
    k: (float(np.nanmin(data[k])), float(np.nanmax(data[k]))) for k in data.names
}

af.write_to(paths.data / "mock" / "data.asdf", all_array_storage="internal")
af.close()
