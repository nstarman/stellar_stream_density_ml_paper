"""Setup."""

import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from matplotlib import pyplot as plt

sys.path.append(Path(__file__).parents[3].as_posix())
# isort: split

from scripts import paths
from scripts.pal5.frames import pal5_frame as frame

##############################################################################
# PARAMETERS

SAVE_LOC = paths.data / "pal5" / "masks.asdf"

try:
    snkmk = snakemake.params
except NameError:
    snkmk = {"load_from_static": False, "save_to_static": False}


if snkmk["load_from_static"]:
    shutil.copyfile(paths.static / "pal5" / "masks.asdf", SAVE_LOC)

    sys.exit(0)


##############################################################################
# Read tables

# Gaia Data
table = QTable.read(paths.data / "pal5" / "gaia_ps1_xm.asdf")

c_pal5_icrs = SkyCoord(
    ra=table["ra"], dec=table["dec"], pm_ra_cosdec=table["pmra"], pm_dec=table["pmdec"]
)
c_pal5 = c_pal5_icrs.transform_to(frame)


##############################################################################
# Masks

masks_table = QTable()

# =============================================================================
# M5

M5 = SkyCoord.from_name("messier 5")
masks_table["M5"] = ~(M5.separation(c_pal5_icrs) < 0.8 * u.deg)


# =============================================================================
# Other Thing

masks_table["things"] = ~(
    (c_pal5.pm_phi1_cosphi2.value > -1)
    & (c_pal5.pm_phi1_cosphi2.value < 2)
    & (c_pal5.pm_phi2.value > -1)
    & (c_pal5.pm_phi2.value < 1)
)


# =============================================================================
# Proper motion

# Mask edges
pm_edges = QTable(
    rows=[["tight_icrs", *(-3.5, -2) * u.mas / u.yr, *(-3.5, -2) * u.mas / u.yr]],
    names=("label", "pm_phi1_min", "pm_phi1_max", "pm_phi2_min", "pm_phi2_max"),
    dtype=(str, float, float, float, float),
    units=(None, u.mas / u.yr, u.mas / u.yr, u.mas / u.yr, u.mas / u.yr),
    meta={
        "pm_phi1_min": r"$\mu_{\phi_1}\cos{\phi_2}$, not reflex corrected",
        "pm_phi2_min": r"$\mu_{\phi_2}$, not reflex corrected",
    },
)
pm_edges.add_index("label", unique=True)


pm_tight = pm_edges.loc["tight_icrs"]
masks_table["pm_tight_icrs"] = (
    (table["pmra"] > pm_tight["pm_phi1_min"])
    & (table["pmra"] < pm_tight["pm_phi1_max"])
    & (table["pmdec"] > pm_tight["pm_phi2_min"])
    & (table["pmdec"] < pm_tight["pm_phi2_max"])
)

# =============================================================================

masks_table.write(SAVE_LOC)

if snkmk["save_to_static"]:
    shutil.copyfile(SAVE_LOC, paths.static / "pal5" / "masks.asdf")


# =============================================================================
# Diagnostic plot

fig = plt.figure(figsize=(15, 7))
gs = mpl.gridspec.GridSpec(4, 2, width_ratios=[1, 3])
ax1 = fig.add_subplot(gs[1:-1, 0])
ax2 = fig.add_subplot(gs[0:2, 1])
ax3 = fig.add_subplot(gs[2:, 1])

_mask = masks_table["M5"] & masks_table["things"]
ax1.hist2d(
    table["pmra"][_mask].value,
    table["pmdec"][_mask].value,
    bins=(np.linspace(-10, 10, 128), np.linspace(-10, 10, 128)),
    cmap="Greys",
    norm=mpl.colors.LogNorm(),
)


def _sel_patch(row: Any, k1: str, k2: str, **kwargs: Any) -> Any:
    """Add a Rectangular patch to the plot."""
    rec = mpl.patches.Rectangle(
        (row[k1 + "_min"].value, row[k2 + "_min"].value),
        row[k1 + "_max"].value - row[k1 + "_min"].value,
        row[k2 + "_max"].value - row[k2 + "_min"].value,
        **kwargs
    )
    rec.set_facecolor((*rec.get_facecolor()[:-1], 0.05))

    return rec


ax1.add_patch(_sel_patch(pm_edges.loc["tight_icrs"], "pm_phi1", "pm_phi2", color="red"))

ax2.plot(
    c_pal5.phi1[masks_table["pm_tight_icrs"]],
    table["pmdec"][masks_table["pm_tight_icrs"]],
    c="black",
    marker=",",
    linestyle="none",
    alpha=1,
)
ax2.set_ylabel(r"$\mu_{\phi_1}^*$ [deg]")

ax3.plot(
    c_pal5.phi1[masks_table["pm_tight_icrs"]],
    c_pal5.phi2[masks_table["pm_tight_icrs"]],
    c="black",
    marker=",",
    linestyle="none",
    alpha=1,
)
ax3.set_xlabel(r"$\phi_1$ [deg]")
ax3.set_ylabel(r"$\phi_2$ [deg]")

fig.tight_layout()
fig.savefig(paths.figures / "pal5" / "diagnostic" / "masks.png", dpi=300)
