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
    iso_medium = af["isochrone_medium"]


###############################################################################

_mask = (
    # position cuts
    (table["phi1"] > -80 * u.deg)
    & (table["phi1"] < 10 * u.deg)
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


fig = plt.figure(layout="constrained", figsize=(11, 10))
subfigs = fig.subfigures(2, 1, hspace=0.05)

gs0 = mpl.gridspec.GridSpec(2, 2, width_ratios=[1, 3], figure=subfigs[0])
gs1 = mpl.gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1], figure=subfigs[1])


# -----------------------------------------------------------------------------
# PM space

ax00 = subfigs[0].add_subplot(gs0[0, 0], rasterization_zorder=100)
ax00.hist2d(
    table["pm_phi1"].value[_mask],
    table["pm_phi2"].value[_mask],
    bins=(np.linspace(-22, 2, 128), np.linspace(-9, 4, 128)),
    cmap="Greys",
    norm=mpl.colors.LogNorm(),
    zorder=-10,
)
ax00.add_patch(_sel_patch(pm_edges.loc["loose"], "pm_phi1", "pm_phi2", color="y"))
ax00.add_patch(
    _sel_patch(pm_edges.loc["medium"], "pm_phi1", "pm_phi2", color="tab:red")
)
ax00.set(
    xlabel=r"$\mu_{\phi_1}$ [mas yr$^{-1}$]", ylabel=r"$\mu_{\phi_2}$ [mas yr$^{-1}$]"
)

# -----------------------------------------------------------------------------
# PM selection

ax01 = subfigs[0].add_subplot(
    gs0[0, 1],
    xlabel=r"$\phi_1$ [deg]",
    ylabel=r"$\phi_2$ [deg]",
    xlim=(-100, 30),
    ylim=(-9, 5),
    rasterization_zorder=0,
)
ax01.plot(
    table["phi1"][masks_table["pm_loose"] ^ masks_table["pm_tight"]],
    table["phi2"][masks_table["pm_loose"] ^ masks_table["pm_tight"]],
    c="y",
    marker=",",
    linestyle="none",
    alpha=0.25,
    zorder=-10,
)
ax01.plot(
    table["phi1"][masks_table["pm_tight"]],
    table["phi2"][masks_table["pm_tight"]],
    c="tab:red",
    marker=",",
    linestyle="none",
    alpha=1,
    zorder=-10,
)

# -----------------------------------------------------------------------------
# Phot space

ax10 = subfigs[0].add_subplot(
    gs0[1, 0],
    xlabel=r"$g_0 - i_0$ [mag]",
    ylabel=r"$g_0$ [mag]",
    xlim=(-1, 3),
    ylim=(24, 12),
    rasterization_zorder=0,
)
_mask = _mask & (
    ((table["g0"] > 0) & (table["i0"] > 0))
    & ((table["phi2"] > -2.5 * u.deg) & (table["phi2"] < 1 * u.deg))
)
ax10.hist2d(
    (table["g0"] - table["i0"]).value[_mask & masks_table["pm_tight"]],
    table["g0"].value[_mask & masks_table["pm_tight"]],
    bins=100,
    label="GD-1",
    norm=mpl.colors.LogNorm(),
    cmap="Greys",
    zorder=-10,
)

# Isochrone
ax10.plot(*iso_medium.T, c="tab:blue", label="iso 0.3")

ax10.legend(loc="upper left")

# -----------------------------------------------------------------------------
# Phot selection

ax11 = subfigs[0].add_subplot(
    gs0[1, 1],
    xlabel=r"$\phi_1$ [deg]",
    ylabel=r"$\phi_2$ [deg]",
    rasterization_zorder=0,
)
ax11.hist2d(
    table["phi1"][masks_table["phot_medium"]].value,
    table["phi2"][masks_table["phot_medium"]].value,
    cmap="Blues",
    density=True,
    bins=200,
    alpha=0.5,
    norm=mpl.colors.LogNorm(),
    zorder=-10,
)
ax11.hist2d(
    table["phi1"][masks_table["phot_tight"]].value,
    table["phi2"][masks_table["phot_tight"]].value,
    cmap="Blues",
    density=True,
    bins=200,
    norm=mpl.colors.LogNorm(),
    zorder=-5,
)

# -----------------------------------------------------------------------------
# Combined Selection Tight

subfigs[1].suptitle("Applying Selections")

sel = (
    masks_table["pm_tight"]
    & masks_table["phot_medium"]
    & (table["parallax"] > -0.5 * u.mas)
)

# Phi2
ax30 = subfigs[1].add_subplot(gs1[0, 0], ylabel=r"$\phi_2$ [$\degree$]")
ax30.hist2d(
    table["phi1"].value[sel],
    table["phi2"].value[sel],
    cmap="Purples",
    density=True,
    bins=100,
    rasterized=True,
)

# Parallax
ax31 = subfigs[1].add_subplot(gs1[0, 1], ylabel=r"$\varpi$ [mas]")
ax31.hist2d(
    table["phi1"].value[sel],
    table["parallax"].value[sel],
    cmap="Purples",
    density=True,
    bins=100,
    rasterized=True,
)

# PM-Phi1
ax32 = subfigs[1].add_subplot(gs1[0, 2], ylabel=r"$\mu_{\phi_1}^*$ [mas yr$^{-1}$]")
ax32.hist2d(
    table["phi1"].value[sel],
    table["pm_phi1"].value[sel],
    cmap="Purples",
    density=True,
    bins=100,
    rasterized=True,
)

# PM-Phi2
ax33 = subfigs[1].add_subplot(gs1[0, 3], ylabel=r"$\mu_{\phi_2}$ [mas yr$^{-1}$]")
ax33.hist2d(
    table["phi1"].value[sel],
    table["pm_phi2"].value[sel],
    cmap="Purples",
    density=True,
    bins=100,
    rasterized=True,
)

for ax in (ax30, ax31, ax32, ax33):
    ax.xaxis.set_ticklabels([])


# -----------------------------------------------------------------------------
# Combined Selection

sel = (
    masks_table["pm_loose"]
    & masks_table["phot_medium"]
    & (table["parallax"] > -0.5 * u.mas)
)

# Phi2
ax40 = subfigs[1].add_subplot(
    gs1[1, 0], xlabel=r"$\phi_1$ [$\degree$]", ylabel=r"$\phi_2$ [$\degree$]"
)
ax40.hist2d(
    table["phi1"].value[sel],
    table["phi2"].value[sel],
    cmap="Greens",
    density=True,
    bins=100,
    rasterized=True,
)

# Parallax
ax41 = subfigs[1].add_subplot(
    gs1[1, 1], xlabel=r"$\phi_1$ [$\degree$]", ylabel=r"$\varpi$ [mas]"
)
ax41.hist2d(
    table["phi1"].value[sel],
    table["parallax"].value[sel],
    cmap="Greens",
    density=True,
    bins=100,
    rasterized=True,
)

# PM-Phi1
ax42 = subfigs[1].add_subplot(
    gs1[1, 2],
    xlabel=r"$\phi_1$ [$\degree$]",
    ylabel=r"$\mu_{\phi_1}^*$ [mas yr$^{-1}$]",
)
ax42.hist2d(
    table["phi1"].value[sel],
    table["pm_phi1"].value[sel],
    cmap="Greens",
    density=True,
    bins=100,
    rasterized=True,
)

# PM-Phi2
ax43 = subfigs[1].add_subplot(
    gs1[1, 3], xlabel=r"$\phi_1$ [$\degree$]", ylabel=r"$\mu_{\phi_2}$ [mas yr$^{-1}$]"
)
ax43.hist2d(
    table["phi1"].value[sel],
    table["pm_phi2"].value[sel],
    cmap="Greens",
    density=True,
    bins=100,
    rasterized=True,
)

for ax in (ax40, ax41, ax42, ax43):
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks([-100, -75, -50, -25, 0, 25])
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%0d"))

# -----------------------------------------------------------------------------

fig.savefig(paths.figures / "gd1" / "data_selection.pdf")
