"""Combine data fields into one dataset."""

import contextlib
import shutil
import sys
from pathlib import Path
from typing import Any

import asdf
import astropy.units as u
import dustmaps.bayestar
import gala.coordinates as gc
import numpy as np
from astropy.coordinates import ICRS, Distance, SkyCoord
from astropy.table import QTable, vstack
from zero_point import zpt

sys.path.append(Path(__file__).parents[3].as_posix())
# isort: split

from scripts import paths
from scripts.pal5.frames import pal5_frame as frame

##############################################################################
# Parameters

SAVE_LOC = paths.data / "pal5" / "gaia_ps1_xm.asdf"

snkmk: dict[str, Any]
try:
    snkmk = dict(snakemake.params)
except NameError:
    snkmk = {"load_from_static": False, "save_to_static": False}


##############################################################################

if snkmk["load_from_static"]:
    shutil.copyfile(paths.static / "pal5" / "gaia_ps1_xm.asdf", SAVE_LOC)
    sys.exit(0)

# -----------------------------------------------------------------------------
# Initial combination

# NOTE: need to use the fixed polygons
af = asdf.open(paths.static / "pal5" / "gaia_ps1_xm_polygons.asdf", mode="r")

combined = vstack(af.search("polygon-*").nodes, metadata_conflicts="silent")

# correct metadata
combined.meta.pop("query")
combined.meta["frame"] = f"{frame.__class__.__module__}.{frame.__class__.__name__}"

# correct unit and data types
combined["ps1_g"].unit = u.mag

combined.write(SAVE_LOC, format="asdf")

del combined
af.close()

# -----------------------------------------------------------------------------

table = QTable.read(SAVE_LOC)

# ---------------------------------------
# Astrometrics

# Custom Frame Coordinates. Note that we don't need to specify the distance
# because NaNs can spread through the transformation.
c_pal5 = ICRS(
    ra=table["ra"],
    dec=table["dec"],
    pm_ra_cosdec=table["pmra"],
    pm_dec=table["pmdec"],
).transform_to(frame)

# Add to Table
table["phi1"] = c_pal5.phi1
table["phi2"] = c_pal5.phi2
table["pm_phi1"] = c_pal5.pm_phi1_cosphi2
table["pm_phi2"] = c_pal5.pm_phi2

# Transform covariance matrix
cov_icrs = np.empty((len(table), 2, 2))
cov_icrs[:, 0, 0] = table["pmra_error"].value ** 2
cov_icrs[:, 0, 1] = table["pmra_pmdec_corr"]
cov_icrs[:, 1, 0] = table["pmra_pmdec_corr"]
cov_icrs[:, 1, 1] = table["pmdec_error"].value ** 2

cov_pal5 = gc.transform_pm_cov(c_pal5.transform_to(ICRS()), cov_icrs, frame)

# TODO: what do negative errors mean?
table["phi1_error"] = np.hypot(table["ra_error"], table["dec_error"]) << u.deg
table["phi2_error"] = np.hypot(table["ra_error"], table["dec_error"]) << u.deg
table["pm_phi1_error"] = np.sqrt(np.abs(cov_pal5[:, 0, 0])) << (u.mas / u.yr)
table["pm_phi2_error"] = np.sqrt(np.abs(cov_pal5[:, 1, 1])) << (u.mas / u.yr)
table["pmphi1_pmphi2_corr"] = cov_pal5[:, 0, 1]

# TODO: remove this when asdf can serialize MaskedQuantity
for _m in (
    "ps1_g",
    "ps1_g_error",
    "ps1_r",
    "ps1_r_error",
    "ps1_i",
    "ps1_i_error",
    "ps1_z",
    "ps1_z_error",
    "ps1_y",
    "ps1_y_error",
    "ag_gspphot",
    "ebpminrp_gspphot",
):
    with contextlib.suppress(AttributeError):
        table[_m] = table[_m].unmasked


# ---------------------------------------
# Zero Point Correction

zpt.load_tables()
table["parallax_zpt"] = (
    zpt.get_zpt(
        table["gaia_g"],
        table["nu_eff"],
        table["pseudocolour"],
        table["ecl_lat"],
        table["astrometric_params_solved"],
    )
    * u.mas
)

table["parallax"] = table["parallax"] + table["parallax_zpt"]
table.meta["parallax"] = "parallax, zero point corrected."


# ---------------------------------------
# Photometrics

# Set the NaN distances to 20 kpc
c_icrs = SkyCoord(
    ra=table["ra"],
    dec=table["dec"],
    distance=Distance(parallax=table["parallax"] << u.arcsecond, allow_negative=True),
    pm_ra_cosdec=table["pmra"],
    pm_dec=table["pmdec"],
    frame="icrs",
)
c_icrs.data.distance[np.isnan(c_icrs.distance)] = 20 * u.kpc

dustmap = dustmaps.bayestar.BayestarQuery(
    map_fname=None, max_samples=None, version="bayestar2019"
)

table["E(B-V)"] = dustmap.query(c_icrs, mode="best")

# Add the 1-sigma errors (+1sigma - -1sigma) / 2
table["E(B-V)_error"] = (
    np.diff(dustmap.query(c_icrs, mode="percentile", pct=[15.87, 84.13])).flatten() / 2
) * u.mag

# numbers from https://ui.adsabs.harvard.edu/abs/2019ApJ...887...93G
table["g0"] = table["ps1_g"] - 3.158 * table["E(B-V)"] * u.mag
table["g0_error"] = np.hypot(table["ps1_g_error"], (3.158 * table["E(B-V)_error"]))

table["r0"] = table["ps1_r"] - 2.617 * table["E(B-V)"] * u.mag
table["r0_error"] = np.hypot(table["ps1_r_error"], (2.617 * table["E(B-V)_error"]))

table["i0"] = table["ps1_i"] - 1.971 * table["E(B-V)"] * u.mag
table["i0_error"] = np.hypot(table["ps1_i_error"], (1.971 * table["E(B-V)_error"]))

table["z0"] = table["ps1_z"] - 1.549 * table["E(B-V)"] * u.mag
table["z0_error"] = np.hypot(table["ps1_z_error"], (1.549 * table["E(B-V)_error"]))

table["y0"] = table["ps1_y"] - 1.263 * table["E(B-V)"] * u.mag
table["y0_error"] = np.hypot(table["ps1_y_error"], (1.263 * table["E(B-V)_error"]))

# Metadata
table.meta["E(B-V)"] = "Bayestar 2019 exctinction"
table.meta["g0"] = "extinction-corrected g_mean_psf_mag"
table.meta["r0"] = "extinction-corrected r_mean_psf_mag"
table.meta["i0"] = "extinction-corrected i_mean_psf_mag"
table.meta["z0"] = "extinction-corrected z_mean_psf_mag"
table.meta["y0"] = "extinction-corrected y_mean_psf_mag"

# Add colors
table["g0-r0"] = table["g0"] - table["r0"]
table["g0-r0_error"] = np.sqrt(table["g0_error"] ** 2 + table["r0_error"] ** 2)

# -----------------------------------------------------------------------------

# Sort by phi1
table = table[np.argsort(table["phi1"])]

# Save
table.write(SAVE_LOC, format="asdf")

if snkmk["save_to_static"]:
    shutil.copyfile(SAVE_LOC, paths.static / "pal5" / "gaia_ps1_xm.asdf")
