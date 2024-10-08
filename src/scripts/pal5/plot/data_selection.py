"""Train photometry background flow."""

import sys
from typing import Any

import asdf
import astropy.units as u
import galstreams
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable, Row
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
from showyourwork.paths import user as user_paths

import stream_mapper.pytorch as sml

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.pal5.datasets import data, off_stream

###############################################################################
# Prep

# Munge the data
data = data.astype(np.ndarray)

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

table = QTable.read(paths.data / "pal5" / "gaia_ps1_xm.asdf")
masks_table = QTable.read(paths.data / "pal5" / "masks.asdf")

# Load PM edges
pm_edges = QTable.read(paths.data / "pal5" / "pm_edges.ecsv")
pm_edges.add_index("label", unique=True)

# Load complete mask
with asdf.open(
    paths.data / "pal5" / "info.asdf", lazy_load=False, copy_arrays=True
) as af:
    mask = af["mask"]

# Isochrone
with asdf.open(
    paths.data / "pal5" / "isochrone.asdf", lazy_load=False, copy_arrays=True
) as af:
    isochrone_data = sml.Data(**af["isochrone_data"])
    isochrone_15 = af["isochrone_15"]

# Apply phi1 mask
table = table[masks_table["phi1_subset"]]
mask = mask[masks_table["phi1_subset"]]
masks_table = masks_table[masks_table["phi1_subset"]]  # has to be last

# shortcut bounds
p1min = table["phi1"].min().value
p1max = table["phi1"].max().value
p2min = table["phi2"].min().value
p2max = table["phi2"].max().value

# Lit tracks
_lit1_kw = {"c": "k", "ls": "--", "alpha": 0.6}
_lit2_kw = {"c": "k", "ls": ":", "alpha": 0.6}

# galstreams
allstreams = galstreams.MWStreams(implement_Off=True)
pal5I21 = allstreams["Pal5-I21"].track
pal5PW19 = allstreams["Pal5-PW19"].track


# On/off stream selection
bw_method = 0.05
background_kde = gaussian_kde(
    np.c_[data["g"][off_stream], data["r"][off_stream]].T,
    bw_method=bw_method,
)
positions = np.c_[data["g"][~off_stream], data["r"][~off_stream]]
stream_kde = gaussian_kde(positions.T, bw_method=bw_method)


def _sel_patch(row: Row, k1: str, k2: str, **kwargs: Any) -> mpl.patches.Rectangle:
    """Add a Rectangular patch to the plot."""
    rec = mpl.patches.Rectangle(
        (row[k1 + "_min"].value, row[k2 + "_min"].value),
        row[k1 + "_max"].value - row[k1 + "_min"].value,
        row[k2 + "_max"].value - row[k2 + "_min"].value,
        **kwargs,
    )
    rec.set_facecolor((*rec.get_facecolor()[:-1], 0.1))

    return rec


###############################################################################

fig = plt.figure(layout="constrained", figsize=(11, 6))
outer_grid = fig.add_gridspec(2, 1, wspace=0, hspace=0, height_ratios=[1, 1.75])

gs0 = outer_grid[0, :].subgridspec(1, 4, width_ratios=[1, 3, 3, 1])
gs1 = outer_grid[1, :].subgridspec(2, 2)


# -----------------------------------------------------------------------------
# PM space

ax00 = fig.add_subplot(
    gs0[0, 1],
    xlabel=r"$\mu_{\alpha}^*$ [mas yr$^{-1}$]",
    ylabel=r"$\mu_{\delta}$ [mas yr$^{-1}$]",
    rasterization_zorder=100,
)

mask_ = (table["phi2"] > -4 * u.deg) & (table["phi2"] < 10 * u.deg)

pm_phi1 = table["pmra"].value[mask_]
pm_phi2 = table["pmdec"].value[mask_]
ax00.hist2d(
    pm_phi1,
    pm_phi2,
    bins=(np.linspace(-5, 0, 128), np.linspace(-6, 0, 128)),
    cmap="Greys",
    norm=LogNorm(),
    zorder=-10,
)
ax00.add_patch(
    _sel_patch(pm_edges.loc["med_icrs"], "pm_ra", "pm_dec", color="tab:blue")
)
ax00.plot(
    pal5I21.pm_ra_cosdec.value, pal5I21.pm_dec.value, **_lit1_kw, label="Ibata+21"
)
ax00.plot(pal5PW19.pm_ra_cosdec.value, pal5PW19.pm_dec.value, **_lit2_kw, label="PW+19")

ax00.legend(loc="lower left", fontsize=8)

# -----------------------------------------------------------------------------
# Phot space

ax10 = fig.add_subplot(
    gs0[0, 2],
    xlabel=r"$g_0 - r_0$ [mag]",
    ylabel=r"$g_0$ [mag]",
    rasterization_zorder=0,
)
mask_ = (
    mask_
    & (
        ((table["g0"] > 0) & (table["r0"] > 0))
        & ((table["phi2"] > -2.5 * u.deg) & (table["phi2"] < 1 * u.deg))
    )
    & masks_table["M5"]
    & masks_table["things"]
)
# plot the data
ax10.hist2d(
    (table["g0"] - table["r0"]).value[mask_ & masks_table["pm_med_icrs"]],
    table["g0"].value[mask_ & masks_table["pm_med_icrs"]],
    bins=100,
    label="GD-1",
    norm=LogNorm(),
    cmap="Greys",
    zorder=-10,
    range=((-0.125, 1.25), (14, 21)),
)
# add the on-stream selection
alpha = stream_kde(positions.T) - background_kde(positions.T)
alpha[alpha < 0] = 0
alpha[alpha > 1] = 1
ax10.scatter(
    data["g"][~off_stream] - data["r"][~off_stream],
    data["g"][~off_stream],
    s=1,
    alpha=0.05 + 0.95 * alpha,
    label="on-off",
    c="tab:green",
    zorder=-5,
)

# Add the buffered isochrone
ax10.plot(
    isochrone_15[:, 0] - isochrone_15[:, 1],
    isochrone_15[:, 0],
    c="tab:blue",
    label="isochrone",
    zorder=-5,
)
ax10.invert_yaxis()

# -----------------------------------------------------------------------------
# Combined Selection

sel = (
    masks_table["pm_med_icrs"]
    & masks_table["phot_15"]
    & (table["parallax"] > -0.5 * u.mas)
)

# Phi2
ax30 = fig.add_subplot(
    gs1[0, 0],
    xlabel=r"$\phi_1$ [deg]",
    ylabel=r"$\phi_2$ [$\degree$]",
    rasterization_zorder=0,
    axisbelow=False,
)
ax30.hist2d(
    table["phi1"].value[sel],
    table["phi2"].value[sel],
    cmap="Blues",
    density=True,
    norm=LogNorm(),
    bins=100,
    zorder=-10,
    label="off-stream",
)
# on-stream selection
ax30.plot(
    data["phi1"][~off_stream],
    data["phi2"][~off_stream],
    ls="none",
    marker=",",
    ms=1e-2,
    color="tab:green",
    alpha=0.5,
    zorder=-5,
    label="`on'-stream",
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
ax30.legend(loc="upper right", fontsize=12)

# Parallax
ax31 = fig.add_subplot(
    gs1[0, 1],
    xlabel=r"$\phi_1$ [deg]",
    ylabel=r"$\varpi$ [mas]",
    rasterization_zorder=0,
)
ax31.hist2d(
    table["phi1"].value[sel],
    table["parallax"].value[sel],
    cmap="Blues",
    density=True,
    norm=LogNorm(),
    bins=100,
    zorder=-10,
)
ax31.set_axisbelow(False)

# PM-Phi1
ax32 = fig.add_subplot(
    gs1[1, 0],
    xlabel=r"$\phi_1$ [deg]",
    ylabel=r"$\mu_{\phi_1}^*$ [mas yr$^{-1}$]",
    rasterization_zorder=0,
)
ax32.hist2d(
    table["phi1"].value[sel],
    table["pm_phi1"].value[sel],
    cmap="Blues",
    density=True,
    norm=LogNorm(),
    bins=100,
    zorder=-10,
)
ax32.set_axisbelow(False)

# PM-Phi2
ax33 = fig.add_subplot(
    gs1[1, 1],
    xlabel=r"$\phi_1$ [deg]",
    ylabel=r"$\mu_{\phi_2}$ [mas yr$^{-1}$]",
    rasterization_zorder=0,
)
ax33.hist2d(
    table["phi1"].value[sel],
    table["pm_phi2"].value[sel],
    cmap="Blues",
    density=True,
    norm=LogNorm(),
    bins=100,
    zorder=-10,
)
ax33.set_axisbelow(False)

# -----------------------------------------------------------------------------

fig.savefig(paths.figures / "pal5" / "data_selection.pdf")
