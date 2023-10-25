"""Plot the data masks."""

import sys
from typing import Any

import asdf
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable, Row
from showyourwork.paths import user as user_paths

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split


###############################################################################
# Load stuff

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

table = QTable.read(paths.data / "gd1" / "gaia_ps1_xm.asdf")
masks_table = QTable.read(paths.data / "gd1" / "masks.asdf")

pm_edges = QTable.read(paths.data / "gd1" / "pm_edges.ecsv")
pm_edges.add_index("label", unique=True)

with asdf.open(
    paths.data / "gd1" / "isochrone.asdf", lazy_load=False, copy_arrays=True
) as af:
    isochrone = af["isochrone_medium"]


###############################################################################

_mask = (
    # position cuts
    (table["phi1"] > -100 * u.deg)
    & (table["phi1"] < 20 * u.deg)
    & (table["phi2"] > -4 * u.deg)
    & (table["phi2"] < 2 * u.deg)
)


def _sel_patch(row: Row, k1: str, k2: str, **kwargs: Any) -> mpl.patches.Rectangle:
    """Add a Rectangular patch to the plot."""
    rec = mpl.patches.Rectangle(
        (row[k1 + "_min"].value, row[k2 + "_min"].value),
        row[k1 + "_max"].value - row[k1 + "_min"].value,
        row[k2 + "_max"].value - row[k2 + "_min"].value,
        **kwargs
    )
    rec.set_facecolor((*rec.get_facecolor()[:-1], 0.1))

    return rec


fig = plt.figure(layout="constrained", figsize=(11, 6))
outer_grid = fig.add_gridspec(2, 1, wspace=0, hspace=0, height_ratios=[1, 1.75])

gs0 = outer_grid[0, :].subgridspec(1, 4, width_ratios=[1, 3, 3, 1])
gs1 = outer_grid[1, :].subgridspec(2, 2)


# -----------------------------------------------------------------------------
# PM space

ax00 = fig.add_subplot(
    gs0[0, 1],
    xlabel=r"$\mu_{\phi_1}$ [mas yr$^{-1}$]",
    ylabel=r"$\mu_{\phi_2}$ [mas yr$^{-1}$]",
    rasterization_zorder=0,
)
ax00.hist2d(
    table["pm_phi1"].value[_mask],
    table["pm_phi2"].value[_mask],
    bins=(np.linspace(-22, 2, 128), np.linspace(-9, 4, 128)),
    cmap="Greys",
    norm=mpl.colors.LogNorm(),
    zorder=-10,
)
ax00.add_patch(
    _sel_patch(pm_edges.loc["medium"], "pm_phi1", "pm_phi2", color="tab:blue")
)
ax00.set_axisbelow(False)

# -----------------------------------------------------------------------------
# Phot space

ax10 = fig.add_subplot(
    gs0[0, 2],
    xlabel=r"$g_0 - i_0$ [mag]",
    ylabel=r"$g_0$ [mag]",
    rasterization_zorder=0,
)
_mask = _mask & (
    ((table["g0"] > 0) & (table["i0"] > 0))
    & ((table["phi2"] > -2.5 * u.deg) & (table["phi2"] < 1 * u.deg))
)
ax10.hist2d(
    (table["g0"] - table["i0"]).value[_mask & masks_table["pm_medium"]],
    table["g0"].value[_mask & masks_table["pm_medium"]],
    bins=100,
    norm=mpl.colors.LogNorm(),
    cmap="Greys",
    zorder=-10,
)

# Isochrone
ax10.plot(*isochrone.T, c="tab:blue", label="w=0.3", zorder=-5)

ax10.set(xlim=(-1, 3), ylim=(24, 12))  # has to be after the hist2d
ax10.legend(loc="upper left")
ax10.set_axisbelow(False)

# -----------------------------------------------------------------------------
# Combined Selection Tight

sel = (
    masks_table["pm_medium"]
    & masks_table["phot_medium"]
    & (table["parallax"] > -0.5 * u.mas)
    & _mask
)

# Phi2
ax30 = fig.add_subplot(
    gs1[0, 0], ylabel=r"$\phi_2$ [$\degree$]", rasterization_zorder=0, xticklabels=[]
)
ax30.hist2d(
    table["phi1"].value[sel],
    table["phi2"].value[sel],
    cmap="Blues",
    density=True,
    bins=100,
    zorder=-10,
)
ax30.set_axisbelow(False)

# Parallax
ax31 = fig.add_subplot(gs1[0, 1], ylabel=r"$\varpi$ [mas]", xticklabels=[])
ax31.hist2d(
    table["phi1"].value[sel],
    table["parallax"].value[sel],
    cmap="Blues",
    density=True,
    bins=100,
    rasterized=True,
)
ax31.set_axisbelow(False)

# PM-Phi1
ax32 = fig.add_subplot(
    gs1[1, 0], xlabel=r"$\phi_1$", ylabel=r"$\mu_{\phi_1}^*$ [mas yr$^{-1}$]"
)
ax32.hist2d(
    table["phi1"].value[sel],
    table["pm_phi1"].value[sel],
    cmap="Blues",
    density=True,
    bins=100,
    rasterized=True,
)
ax32.set_axisbelow(False)

# PM-Phi2
ax33 = fig.add_subplot(
    gs1[1, 1], xlabel=r"$\phi_1$", ylabel=r"$\mu_{\phi_2}$ [mas yr$^{-1}$]"
)
ax33.hist2d(
    table["phi1"].value[sel],
    table["pm_phi2"].value[sel],
    cmap="Blues",
    density=True,
    bins=100,
    rasterized=True,
)
ax33.set_axisbelow(False)

# -----------------------------------------------------------------------------

fig.savefig(paths.figures / "gd1" / "data_selection.pdf")
