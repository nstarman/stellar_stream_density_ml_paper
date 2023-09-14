"""Setup."""

import shutil
import sys
from typing import Any

import matplotlib as mpl
import matplotlib.path as mpath
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from matplotlib import pyplot as plt
from showyourwork.paths import user as user_paths

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.as_posix())
# isort: split

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
# Off-stream selection
# Applying this mask to the data table will remove the off-stream region.

footprint = np.load(paths.data / "pal5" / "footprint.npz")["footprint"]

masks_table["off_stream"] = mpath.Path(footprint.T, readonly=True).contains_points(
    np.c_[table["phi1"].to_value("deg"), table["phi2"].to_value("deg")]
)

# =============================================================================
# M5
# Applying this mask to the data table will remove the M5 stars.

M5 = SkyCoord.from_name("messier 5")
masks_table["M5"] = ~(M5.separation(c_pal5_icrs) < 0.8 * u.deg)


# =============================================================================
# Other Thing
# Applying this mask to the data table will remove the other thing.

masks_table["things"] = ~(
    (c_pal5.pm_phi1_cosphi2.value > -1)
    & (c_pal5.pm_phi1_cosphi2.value < 2)
    & (c_pal5.pm_phi2.value > -1)
    & (c_pal5.pm_phi2.value < 1)
)


# =============================================================================
# Proper motion
# Applying this mask to the data table will remove the stars outside the
# proper motion box.

pm_edges = QTable.read(paths.data / "pal5" / "pm_edges.ecsv")
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