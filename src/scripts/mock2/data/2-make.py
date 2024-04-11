"""Simulate mock stream."""

import itertools
import sys
from dataclasses import asdict
from typing import Any

import asdf
import astropy.coordinates as apyc
import astropy.units as u
import brutus.filters
import brutus.seds
import gala.dynamics as gd
import gala.potential as gp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable, vstack
from emulation_model import (
    make_astro_data_model,
    make_astro_error_model,
    make_phot_data_model,
    make_phot_error_model,
)
from gala.units import galactic
from scipy import stats
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline
from scipy.stats import norm as guassian
from showyourwork.paths import user as user_paths

import stream_mapper.pytorch as sml

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
    snkmk = {"seed": 35, "diagnostic_plots": True}

diagnostic_plots = paths.scripts / "mock2" / "_diagnostics"
diagnostic_plots.mkdir(parents=True, exist_ok=True)

seed: int = snkmk["seed"]
rng = np.random.default_rng(seed)

N_BACKGROUND = 30_000
# N_STREAM_MAX = 2_000  # going to be less b/c of gaps
stop_stream_phot = -1_000  # Limit the isochrone.

filters = brutus.filters.ps[:-2]  # g, r
nnfile = (paths.static / "brutus" / "nn_c3k.h5").resolve()
mistfile = (paths.static / "brutus" / "MIST_1.2_iso_vvcrit0.0.h5").resolve()

##############################################################################

# ============================================================================
# Background

bkg_tbl = QTable()

# ---- Astrometrics ----

key = jr.PRNGKey(90)
key, subkey = jr.split(key)

flow_astro_data = make_astro_data_model(paths.data / "mock2" / "astro_data_flow.eqx")
data_samples = flow_astro_data.sample(subkey, (N_BACKGROUND + 500,))

key, subkey = jr.split(key)
flow_astro_error = make_astro_error_model(paths.data / "mock2" / "astro_error_flow.eqx")
data_error_samples = flow_astro_error.sample(subkey, (len(data_samples),))


data_samples = data_samples[data_samples[:, 3] > -25]
data_samples = data_samples[data_samples[:, 3] < 25]
data_samples = data_samples[:N_BACKGROUND]
data_error_samples = data_error_samples[:N_BACKGROUND]


bkg_tbl["phi1"] = data_samples[:, 0] * u.deg
bkg_tbl["phi1_err"] = (np.maximum(data_error_samples[:, 0], 36) * u.mas).to(u.deg)
bkg_tbl["phi2"] = data_samples[:, 1] * u.deg
bkg_tbl["phi2_err"] = (np.maximum(data_error_samples[:, 1], 36) * u.mas).to(u.deg)
bkg_tbl["parallax"] = data_samples[:, 2] * u.mas
bkg_tbl["parallax_err"] = np.abs(data_error_samples[:, 2]) / 20 * u.mas
bkg_tbl["distmod"] = apyc.Distance(parallax=bkg_tbl["parallax"]).distmod
bkg_tbl["pmphi1"] = data_samples[:, 3] * u.mas / u.yr
bkg_tbl["pmphi1_err"] = np.abs(data_error_samples[:, 3]) * u.mas / u.yr
bkg_tbl["pmphi2"] = data_samples[:, 4] * u.mas / u.yr
bkg_tbl["pmphi2_err"] = np.abs(data_error_samples[:, 4]) * u.mas / u.yr


# ---- Photometrics ----

flow_phot_data = make_phot_data_model(paths.data / "mock2" / "phot_data_flow.eqx")
flow_phot_error = make_phot_error_model(paths.data / "mock2" / "phot_error_flow.eqx")

key, subkey = jr.split(key)
phot_samples = flow_phot_data.sample(subkey, (N_BACKGROUND + 1_000,))
phot_samples = phot_samples[phot_samples[:, 0] > 10]
phot_samples = phot_samples[phot_samples[:, 0] < 22]
phot_samples = phot_samples[phot_samples[:, 1] > -0.1]
phot_samples = phot_samples[phot_samples[:, 1] < 1.3]
phot_samples = phot_samples[:N_BACKGROUND]

key, subkey = jr.split(key)
phot_errors = flow_phot_error.sample(subkey, (N_BACKGROUND,))

bkg_tbl["g"] = phot_samples[:, 0] * u.mag
bkg_tbl["g_err"] = phot_errors[:, 0] * u.mag
bkg_tbl["r"] = u.Quantity(phot_samples[:, 0] - phot_samples[:, 1], u.mag)
bkg_tbl["r_err"] = u.Quantity(
    np.sqrt(np.abs(phot_errors[:, 1] ** 2 - phot_errors[:, 0] ** 2)), u.mag
)


# ============================================================================
# Stream


class StreamMF(stats.rv_continuous):
    r"""A stream mass function.

    Inspired by Palomar 5, but applicable to many streams.

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


galactocentric_frame = apyc.Galactocentric()

w0_gd1_icrs = apyc.ICRS(
    ra=148.94 * u.deg,
    dec=36.16 * u.deg,
    distance=7.56 * u.kpc,
    pm_ra_cosdec=-8 * u.mas / u.yr,
    pm_dec=-6 * u.mas / u.yr,
    radial_velocity=0 * u.km / u.s,
)
w0_gd1_gcf = w0_gd1_icrs.transform_to(galactocentric_frame)

w0 = gd.PhaseSpacePosition(w0_gd1_gcf)

# Define the gravitational potential
potential = gp.MilkyWayPotential(units=galactic)

# Initialize the mockstream generator
stream_generator = gd.MockStreamGenerator(
    gd.FardalStreamDF(lead=True, trail=True, random_state=np.random.default_rng(42)),
    potential,
)

# Integrate the stream
stream, prog = stream_generator.run(
    w0,
    1e4 * u.Msun,
    t=np.linspace(0, -2, 1_000) * u.Gyr,
)

# Transform to ICRS
stream_icrs = stream.to_coord_frame(apyc.ICRS(), galactocentric_frame)
prog_icrs = prog.to_coord_frame(apyc.ICRS(), galactocentric_frame)

# Degrade for observation
_idx2step10 = np.sort(
    np.stack(
        (
            np.arange(0, len(stream_icrs), step=8),
            np.arange(1, len(stream_icrs), step=8),
        )
    ).reshape((-1,))
)
idx = np.zeros(len(stream_icrs), dtype=bool)
idx[_idx2step10] = True
idx &= stream_icrs.ra >= (w0_gd1_icrs.ra - 19 * u.deg)
idx &= stream_icrs.ra <= (w0_gd1_icrs.ra + 19 * u.deg)
idx &= stream_icrs.dec >= (w0_gd1_icrs.dec - 5 * u.deg)
idx &= stream_icrs.dec <= (w0_gd1_icrs.dec + 5 * u.deg)
stream_icrs_obs = stream_icrs[idx]

data_error_samples = flow_astro_error.sample(subkey, (len(stream_icrs_obs),))

# Need to add observational errors to the stream
phi2 = stream_icrs_obs.dec.view(u.Quantity)
phi2_err = np.abs(data_error_samples[:, 1]) * u.mas
phi2 = phi2 + guassian.rvs(scale=phi2_err.value, random_state=rng) * u.mas

parallax = stream_icrs_obs.distance.parallax << u.mas
parallax_err = np.abs(data_error_samples[:, 2]) / 20 * u.mas
noise = guassian.rvs(scale=parallax_err.value, random_state=rng) * u.mas
_parallax = parallax + noise
pmin = bkg_tbl["parallax"].min()
pmax = bkg_tbl["parallax"].max()
bad_fn = lambda p: (p < pmin) | (p > pmax)  # noqa: E731
while np.any(bad_fn(_parallax)):
    bad = bad_fn(_parallax)
    noise = guassian.rvs(scale=parallax_err.value[bad], random_state=rng) * u.mas
    _parallax[bad] = parallax[bad] + noise
parallax = _parallax

pmphi1 = stream_icrs_obs.pm_ra_cosdec
pmphi1_err = np.abs(data_error_samples[:, 3]) * u.mas / u.yr
pmphi1 = pmphi1 + guassian.rvs(scale=pmphi1_err.value, random_state=rng) * u.mas / u.yr

pmphi2 = stream_icrs_obs.pm_dec
pmphi2_err = np.abs(data_error_samples[:, 4]) * u.mas / u.yr
pmphi2 = pmphi2 + guassian.rvs(scale=pmphi2_err.value, random_state=rng) * u.mas / u.yr


stream_tbl = QTable()
stream_tbl["phi1"] = stream_icrs_obs.ra.view(u.Quantity)
stream_tbl["phi1_err"] = (np.maximum(data_error_samples[:, 0], 36) * u.mas).to(u.deg)
stream_tbl["phi2"] = phi2
stream_tbl["phi2_err"] = (np.maximum(phi2_err, 36 * u.mas)).to(u.deg)
stream_tbl["parallax"] = parallax
stream_tbl["parallax_err"] = parallax_err
stream_tbl["distmod"] = apyc.Distance(parallax=parallax).distmod
stream_tbl["pmphi1"] = pmphi1
stream_tbl["pmphi1_err"] = pmphi1_err
stream_tbl["pmphi2"] = pmphi2
stream_tbl["pmphi2_err"] = pmphi2_err


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
abs_mags_ = abs_mags_ - 1.8  # adjustment to get into Gaia range

# Find the finite values
select_finite = np.all(np.isfinite(abs_mags_), axis=1)

# Get the mass
mass = info1["mass"][select_finite]
# And limit to the region of monotonicity
select_monotonic = slice(None, np.argmax(mass) + 1)
mass = mass[select_monotonic]

# Get the absolute magnitudes within the select region
stream_abs_mags = u.Quantity(abs_mags_[select_finite][select_monotonic][:, :2], u.mag)

# Cubic spline the isochrone as a function of gamma
mag_spline = isochrone_spline(stream_abs_mags.value, xp=np)
# Get the gamma values
gamma = mag_spline.x

# mass(gamma) spline
gamma_spline = CubicSpline(mass, gamma)

# Mass function distribution
stream_mf = StreamMF(mass.min(), mass.max())

# Sample the mass function
mass_samples = stream_mf.rvs(size=len(stream_tbl), random_state=rng)
gamma_samples = gamma_spline(mass_samples)

stream_cmd = (
    mag_spline(gamma_samples)
    + guassian.rvs(scale=0.05, size=(len(mass_samples), 2), random_state=rng)
) * u.mag
# Boost to the distance modulus
stream_cmd += stream_icrs_obs.distance.distmod[:, None]

key, subkey = jr.split(key)
phot_errors = flow_phot_error.sample(subkey, (len(stream_cmd),))

# Add to the table
stream_tbl["g"] = stream_cmd[:, 0]
stream_tbl["g_err"] = phot_errors[:, 0] * u.mag
stream_tbl["r"] = stream_cmd[:, 1]
stream_tbl["r_err"] = (
    np.sqrt(np.abs(phot_errors[:, 1] ** 2 - phot_errors[:, 0] ** 2)) * u.mag
)


if snkmk["diagnostic_plots"]:
    fig, ax = plt.subplots(1, 1)

    ax.scatter(bkg_tbl["g"] - bkg_tbl["r"], bkg_tbl["g"], c="gray", alpha=0.1)
    ax.scatter(stream_tbl["g"] - stream_tbl["r"], stream_tbl["g"], c="tab:blue")

    fig.savefig(diagnostic_plots / "cmd.png")


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
af["n_stream"] = len(stream_tbl)
af["stream_table"] = tot_table[tot_table["label"] == "stream"]
af["true_stream"] = QTable(
    {
        "phi1": stream_icrs_obs.ra,
        "phi2": stream_icrs_obs.dec,
        "parallax": stream_icrs_obs.distance.parallax,
        "pmphi1": stream_icrs_obs.pm_ra_cosdec,
        "pmphi2": stream_icrs_obs.pm_dec,
    }
)
af["stream_abs_mags"] = stream_abs_mags
af["isochrone_age"] = isochrone_age
af["isochrone_feh"] = isochrone_feh
af["gamma_mass"] = np.c_[gamma, mass]

# background
af["n_background"] = N_BACKGROUND
af["bkg_table"] = tot_table[tot_table["label"] == "background"]

# off-stream selection
prog = potential.integrate_orbit(prog, t=np.linspace(0, -9, 2) * u.Myr)[-1]
prog_o = potential.integrate_orbit(prog, t=np.linspace(0, 19, 1_000) * u.Myr)
prog_o_icrs = prog_o.to_coord_frame(apyc.ICRS(), galactocentric_frame)
spline = InterpolatedUnivariateSpline(
    prog_o_icrs.ra.deg[::-1], prog_o_icrs.dec.deg[::-1], k=3
)
mean_phi2 = spline(data["phi1"])
af["off_stream"] = (data["phi2"] < mean_phi2 - 1.5) | (data["phi2"] > mean_phi2 + 1.5)

# Misc
af["data"] = {"array": data.array, "names": data.names}
af["where"] = {"array": ~np.isnan(data.array), "names": data.names}
af["scaler"] = asdict(scaler)
af["coord_bounds"] = {
    k: (float(np.nanmin(data[k])), float(np.nanmax(data[k]))) for k in data.names
}

af.write_to(paths.data / "mock2" / "data.asdf", all_array_storage="internal")
af.close()


# ===================================================================

if snkmk["diagnostic_plots"]:
    fig, ax = plt.subplots(1, 1)

    ax.scatter(
        tot_table["phi1"][af["off_stream"]],
        tot_table["phi2"][af["off_stream"]],
        c="tab:blue",
    )
    ax.scatter(
        tot_table["phi1"][~af["off_stream"]],
        tot_table["phi2"][~af["off_stream"]],
        c="tab:green",
    )
    plt.plot(data["phi1"], mean_phi2, c="k")
    fig.savefig(diagnostic_plots / "off_stream.png")


if snkmk["diagnostic_plots"]:
    colnames = (k for k in bkg_tbl.colnames if not k.endswith("_err"))

    dims = len(bkg_tbl.colnames)

    fig, axs = plt.subplots(dims, dims, figsize=(3 * dims, 3 * dims))

    for i, j in itertools.product(range(dims), repeat=2):
        ax = axs[i, j]
        if i < j:
            ax.axis("off")  # Upper triangle
            continue

        if i == j:
            col = bkg_tbl.colnames[i]
            _, bins, _ = ax.hist(
                bkg_tbl[col].value, bins=30, alpha=0.5, label="Background"
            )
            ax.hist(stream_tbl[col].value, bins=bins, alpha=0.5, label="Stream")
            ax.set_title(col)

        else:
            col_i, col_j = bkg_tbl.colnames[i], bkg_tbl.colnames[j]
            ax.scatter(
                bkg_tbl[col_j].value,
                bkg_tbl[col_i].value,
                s=0.1,
                alpha=0.5,
                label="Data",
                rasterized=True,
            )
            ax.scatter(
                stream_tbl[col_j].value,
                stream_tbl[col_i].value,
                s=0.1,
                alpha=0.5,
                label="Stream",
            )
            ax.set_xlabel(col_j)
            ax.set_ylabel(col_i)

    fig.tight_layout()
    fig.savefig(diagnostic_plots / "data.pdf")
