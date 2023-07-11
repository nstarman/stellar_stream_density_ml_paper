"""Plot the data masks."""

import sys

import asdf
import astropy.units as u
import galstreams
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sg
import shapely.ops as so
from astropy.table import QTable
from matplotlib.colors import LogNorm
from scipy.interpolate import InterpolatedUnivariateSpline
from showyourwork.paths import user as user_paths

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.gd1.frames import gd1_frame
from scripts.helper import make_path

###############################################################################
# Load stuff

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

# Load data
table = QTable.read(paths.data / "gd1" / "gaia_ps1_xm.asdf")
masks_table = QTable.read(paths.data / "gd1" / "masks.asdf")

pm_edges = QTable.read(paths.data / "gd1" / "pm_edges.ecsv")

# Load isochrone
with asdf.open(
    paths.data / "gd1" / "isochrone.asdf", lazy_load=False, copy_arrays=True
) as af:
    isochrone = af["isochrone_medium"]

# Load galstreams
mws = galstreams.MWStreams()
gd1 = mws["GD-1-I21"]
gd1_sc = gd1.track.transform_to(gd1_frame)[::100]

# spline of lit tracks
spline_pmphi1 = InterpolatedUnivariateSpline(
    gd1_sc.phi1.value, gd1_sc.pm_phi1_cosphi2.value
)
spline_pmphi2 = InterpolatedUnivariateSpline(gd1_sc.phi1.value, gd1_sc.pm_phi2.value)

# Lit tracks
_lit1_kw = {"c": "k", "ls": "--", "alpha": 0.6}

###############################################################################


path_pm = make_path(pm_edges["pm_phi1_cosphi2_medium"], pm_edges["pm_phi2_medium"])

_mask = (
    # position cuts
    (table["phi1"] > -100 * u.deg)
    & (table["phi1"] < 20 * u.deg)
    & (table["phi2"] > -4 * u.deg)
    & (table["phi2"] < 2 * u.deg)
)


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
    axisbelow=False,
)
ax00.hist2d(
    table["pm_phi1"].value[_mask],
    table["pm_phi2"].value[_mask],
    bins=(np.linspace(-22, 2, 128), np.linspace(-9, 4, 128)),
    cmap="Greys",
    norm=LogNorm(),
    zorder=-10,
)

# Lit
ax00.plot(
    gd1_sc.pm_phi1_cosphi2.value,
    gd1_sc.pm_phi2.value,
    **_lit1_kw,
    label="Ibata+21",
)

# Patch over selection
split = len(pm_edges) // 2
# Bottom left
x1s = pm_edges["pm_phi1_cosphi2_tight"][:split].value
y1s = pm_edges["pm_phi2_tight"][:split].value
# Top right
x2s = pm_edges["pm_phi1_cosphi2_tight"][split:-1].value
y2s = pm_edges["pm_phi2_tight"][split:-1].value

xs, ys = so.unary_union(
    [sg.box(x1, y1, x2, y2) for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s, strict=True)]
).exterior.xy
ax00.fill(xs, ys, alpha=0.4, fc="tab:blue", ec="none")
ax00.plot(x1s, y1s, c="tab:blue", lw=1, zorder=-5)
ax00.plot(x2s, y2s, c="tab:blue", lw=1, zorder=-5)

ax00.legend(loc="lower left", fontsize=8)

# -----------------------------------------------------------------------------
# Phot space

ax10 = fig.add_subplot(
    gs0[0, 2],
    xlabel=r"$g_0 - i_0$ [mag]",
    ylabel=r"$g_0$ [mag]",
    rasterization_zorder=0,
    axisbelow=False,
)
_mask = _mask & (
    ((table["g0"] > 0) & (table["i0"] > 0))
    & ((table["phi2"] > -2.5 * u.deg) & (table["phi2"] < 1 * u.deg))
)
ax10.hist2d(
    (table["g0"] - table["i0"]).value[_mask & masks_table["pm_medium"]],
    table["g0"].value[_mask & masks_table["pm_medium"]],
    bins=100,
    norm=LogNorm(),
    cmap="Greys",
    zorder=-10,
    range=((-0.5, 2.5), (12, 22)),
)

# Isochrone
ax10.plot(*isochrone.T, c="tab:blue", label="w=0.3", zorder=-5)

ax10.invert_yaxis()

# -----------------------------------------------------------------------------
# Combined Selection

sel = (
    masks_table["pm_medium"]
    & masks_table["phot_medium"]
    & (table["parallax"] > -0.5 * u.mas)
    # plot-related cuts
    & (table["phi2"] > -4 * u.deg)
    & (table["phi2"] < 2 * u.deg)
)

# Phi2
ax30 = fig.add_subplot(
    gs1[0, 0],
    ylabel=r"$\phi_2$ [$\degree$]",
    rasterization_zorder=0,
    xticklabels=[],
    axisbelow=False,
)
ax30.hist2d(
    table["phi1"].value[sel],
    table["phi2"].value[sel],
    cmap="Blues",
    norm=LogNorm(),
    density=True,
    bins=100,
    zorder=-10,
)
# add text to the top left corner of the plot saying log-density
t = ax30.text(
    0.05,
    0.95,
    "log-density",
    transform=ax30.transAxes,
    fontsize=12,
    verticalalignment="top",
)
t.set_bbox({"facecolor": "white", "alpha": 0.75, "edgecolor": "gray"})

# Parallax
ax31 = fig.add_subplot(gs1[0, 1], ylabel=r"$\varpi$ [mas]", xticklabels=[])
ax31.hist2d(
    table["phi1"].value[sel],
    table["parallax"].value[sel],
    cmap="Blues",
    norm=LogNorm(),
    density=True,
    bins=100,
    rasterized=True,
)
ax31.set_axisbelow(False)

# PM-Phi1
ax32 = fig.add_subplot(
    gs1[1, 0],
    xlabel=r"$\phi_1$",
    ylabel=r"$\mu_{\phi_1}^*$-GD-1 [mas yr$^{-1}$]",
    axisbelow=False,
)
ax32.hist2d(
    table["phi1"].value[sel],
    table["pm_phi1"].value[sel] - spline_pmphi1(table["phi1"].value[sel]),
    cmap="Blues",
    norm=LogNorm(),
    density=True,
    bins=100,
    rasterized=True,
)

# PM-Phi2
ax33 = fig.add_subplot(
    gs1[1, 1],
    xlabel=r"$\phi_1$",
    ylabel=r"$\mu_{\phi_2}$-GD-1 [mas yr$^{-1}$]",
    axisbelow=False,
)
ax33.hist2d(
    table["phi1"].value[sel],
    table["pm_phi2"].value[sel] - spline_pmphi2(table["phi1"].value[sel]),
    cmap="Blues",
    norm=LogNorm(),
    density=True,
    bins=100,
    rasterized=True,
)

# -----------------------------------------------------------------------------

fig.savefig(paths.figures / "gd1" / "data_selection.pdf")
