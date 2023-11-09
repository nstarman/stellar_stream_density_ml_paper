"""Plot the trained GD1 model."""

from __future__ import annotations

import copy as pycopy
import sys

import astropy.units as u
import galstreams
import matplotlib.pyplot as plt
import numpy as np
import torch as xp
from astropy.coordinates import Distance
from astropy.table import QTable
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from scipy.interpolate import InterpolatedUnivariateSpline
from showyourwork.paths import user as user_paths
from tqdm import tqdm

from stream_ml.core import WEIGHT_NAME, Params

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.gd1.datasets import data
from scripts.gd1.frames import gd1_frame as frame
from scripts.gd1.model import make_model
from scripts.helper import (
    color_by_probable_member,
    manually_set_dropout,
    p2alpha,
    recursive_iterate,
)
from scripts.mpl_colormaps import stream_cmap1 as cmap_stream
from scripts.mpl_colormaps import stream_cmap2 as cmap_spur

# =============================================================================
# Load data

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

# Load control points
stream_cp = QTable.read(paths.data / "gd1" / "control_points_stream.ecsv")
spur_cp = QTable.read(paths.data / "gd1" / "control_points_spur.ecsv")
distance_cp = QTable.read(paths.data / "gd1" / "control_points_distance.ecsv")

# Load model
model = make_model()
model = pycopy.deepcopy(model)
model.load_state_dict(xp.load(paths.data / "gd1" / "models" / "model_11700.pt"))
model = model.eval()

# Load results from 4-likelihoods.py
lik_tbl = QTable.read(paths.data / "gd1" / "membership_likelhoods.ecsv")
stream_prob = np.array(lik_tbl["stream (50%)"])
stream_wgts = np.array(lik_tbl["stream.ln-weight"])

spur_prob = np.array(lik_tbl["spur (50%)"])
spur_wgt = np.array(lik_tbl["spur.ln-weight"])

allstream_prob = np.array(lik_tbl["allstream (50%)"])

# galstreams
allstreams = galstreams.MWStreams(implement_Off=True)
gd1I21 = allstreams["GD-1-I21"].track.transform_to(frame)
gd1PB18 = allstreams["GD-1-PB18"].track.transform_to(frame)


spline_pmphi1 = InterpolatedUnivariateSpline(
    gd1I21[::100].phi1.value, gd1I21[::100].pm_phi1_cosphi2.value
)
spline_pmphi2 = InterpolatedUnivariateSpline(
    gd1I21[::100].phi1.value, gd1I21[::100].pm_phi2.value
)


# =============================================================================
# Likelihood

# Sorter for plotting
psort = np.argsort(allstream_prob)

# Foreground
is_gd1 = (allstream_prob > 0.75)[psort]
is_strm = (stream_prob > 0.75)[psort]
is_spur = (spur_prob > 0.75)[psort]

# Also evaluate the model with dropout on
with xp.no_grad():
    # turn dropout on
    model = model.train()
    manually_set_dropout(model, 0.15)

    # evaluate the model
    ldmpars = [model.unpack_params(model(data)) for i in tqdm(range(100))]

    # mpars
    dmpars = Params(recursive_iterate(ldmpars, ldmpars[0], reduction=lambda x: x))
    mpars = Params(
        recursive_iterate(
            ldmpars, ldmpars[0], reduction=lambda x: np.percentile(x, 50, axis=1)
        ),
    )

    # turn dropout back off
    manually_set_dropout(model, 0)
    model = model.eval()


_is_strm = (stream_prob > 0.75) & (mpars[(f"stream.{WEIGHT_NAME}",)] > -5)
stream_range = (np.min(data["phi1"][_is_strm].numpy()) <= data["phi1"]) & (
    data["phi1"] <= np.max(data["phi1"][_is_strm].numpy())
)
_is_spur = (spur_prob > 0.75) & (mpars[(f"spur.{WEIGHT_NAME}",)] > -5)
spur_range = (np.min(data["phi1"][_is_spur].numpy()) <= data["phi1"]) & (
    data["phi1"] <= -25
)


##############################################################################
# Make Figure

fig = plt.figure(figsize=(11, 15))

gs = GridSpec(
    7,
    1,
    figure=fig,
    height_ratios=(2, 3, 6.5, 6.5, 6.5, 6.5, 6.5),
    hspace=0.15,
    left=0.07,
    right=0.98,
    top=0.965,
    bottom=0.03,
)

# Settings
colors = color_by_probable_member(
    (stream_prob[psort], cmap_stream), (spur_prob[psort], cmap_spur)
)
alphas = p2alpha(allstream_prob[psort])
sizes = 1 + stream_prob[psort]  # range [1, 2]
xlims = (data["phi1"].min(), 10)

_stream_kw = {"ls": "none", "marker": ",", "color": cmap_stream(0.75), "alpha": 0.05}
_spur_kw = {"ls": "none", "marker": ",", "color": cmap_spur(0.75), "alpha": 0.05}
_bounds_kw = {"c": "gray", "ls": "-", "lw": 2, "alpha": 0.8}
_lit1_kw = {"c": "k", "ls": "--", "alpha": 0.6}
_lit2_kw = {"c": "k", "ls": ":", "alpha": 0.6}

# ---------------------------------------------------------------------------
# Colormap

gs0 = gs[0, :].subgridspec(2, 1, height_ratios=(1, 1), hspace=0.15)

# Stream
ax00 = fig.add_subplot(gs0[0, :])
cbar = fig.colorbar(
    ScalarMappable(cmap=cmap_stream), cax=ax00, orientation="horizontal"
)
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")
cbar.ax.text(0.5, 0.5, "Stream Probability", ha="center", va="center", fontsize=14)

# Spur
ax01 = fig.add_subplot(gs0[1, :])
cbar = fig.colorbar(ScalarMappable(cmap=cmap_spur), cax=ax01, orientation="horizontal")
cbar.ax.xaxis.set_ticks([])
cbar.ax.xaxis.set_label_position("bottom")
cbar.ax.text(0.5, 0.5, "Spur Probability", ha="center", va="center", fontsize=14)


# ---------------------------------------------------------------------------
# Weight plot

ax1 = fig.add_subplot(
    gs[1, :],
    xlim=xlims,
    xticklabels=[],
    ylabel=r"$\ln f_{\rm stream}$",
    ylim=(-6, 0),
    rasterization_zorder=0,
)

# Upper and lower bounds
_bounds = model.params[(f"stream.{WEIGHT_NAME}",)].bounds
ax1.axhline(_bounds.lower[0], **_bounds_kw)
ax1.axhline(_bounds.upper[0], **_bounds_kw)

# Stream
f1 = ax1.fill_between(
    data["phi1"],
    np.percentile(stream_wgts, 5, axis=1),
    np.percentile(stream_wgts, 95, axis=1),
    color=cmap_stream(0.99),
    alpha=0.25,
    where=stream_range,
    zorder=-10,
)
(l1,) = ax1.plot(
    data["phi1"][stream_range],
    np.percentile(stream_wgts, 50, axis=1)[stream_range],
    c=cmap_stream(0.99),
    ls="--",
    lw=2,
    label="Mean",
    zorder=-9,
)

# Spur
f2 = ax1.fill_between(
    data["phi1"],
    np.percentile(spur_wgt, 5, axis=1),
    np.percentile(spur_wgt, 95, axis=1),
    color=cmap_spur(0.99),
    alpha=0.25,
    where=spur_range,
    zorder=-10,
)
(l2,) = ax1.plot(
    data["phi1"][spur_range],
    np.percentile(spur_wgt, 50, axis=1)[spur_range],
    c=cmap_spur(0.99),
    ls="--",
    lw=2,
    zorder=-9,
)

ax1.legend(
    [[(f1, l1), (f2, l2)]],
    [r"Models"],
    numpoints=1,
    handler_map={list: HandlerTuple(ndivide=None)},
    loc="upper left",
)

# ---------------------------------------------------------------------------
# Phi2 - variance

gs2 = gs[2].subgridspec(2, 1, height_ratios=(1, 3), wspace=0.0, hspace=0.0)
ax20 = fig.add_subplot(
    gs2[0, :],
    xlim=xlims,
    xticklabels=[],
    ylabel=r"$\ln\sigma_{\phi_2}$",
    aspect="auto",
    rasterization_zorder=0,
)

# Model (stream)
ln_sigma = dmpars["stream.astrometric.phi2", "ln-sigma"]
ax20.fill_between(
    data["phi1"],
    np.percentile(ln_sigma, 5, axis=1),
    np.percentile(ln_sigma, 95, axis=1),
    color=cmap_stream(0.99),
    alpha=0.1,
    where=stream_range,
    zorder=-10,
)
ax20.scatter(
    data["phi1"][stream_range],
    np.percentile(ln_sigma, 50, axis=1)[stream_range],
    s=1,
    color=cmap_stream(0.99),
    zorder=-9,
)

# Model (spur)
ln_sigma = dmpars["spur.astrometric.phi2", "ln-sigma"]
ax20.fill_between(
    data["phi1"],
    np.percentile(ln_sigma, 5, axis=1),
    np.percentile(ln_sigma, 95, axis=1),
    color=cmap_spur(0.99),
    alpha=0.1,
    where=spur_range,
    zorder=-10,
)
ax20.scatter(
    data["phi1"][spur_range],
    np.percentile(ln_sigma, 50, axis=1)[spur_range],
    s=1,
    color=cmap_spur(0.99),
    zorder=-9,
)

# ---------------------------------------------------------------------------
# Phi2

ax21 = fig.add_subplot(
    gs2[1, :],
    xlabel="",
    ylabel=r"$\phi_2$ [deg]",
    xlim=xlims,
    ylim=(data["phi2"].min(), data["phi2"].max()),
    xticklabels=[],
    rasterization_zorder=0,
)

# Stream control points
ax21.errorbar(
    stream_cp["phi1"],
    stream_cp["phi2"],
    yerr=stream_cp["w_phi2"],
    fmt="o",
    color="k",
    capthick=3,
    elinewidth=3,
    capsize=3,
    alpha=0.4,
    zorder=-21,
)
p1 = ax21.errorbar(
    stream_cp["phi1"],
    stream_cp["phi2"],
    yerr=stream_cp["w_phi2"],
    fmt=".",
    c=cmap_stream(0.99),
    capsize=2,
    alpha=0.4,
    zorder=-20,
    label="Stream Control Points",
)
# Spur control points
ax21.errorbar(
    spur_cp["phi1"],
    spur_cp["phi2"],
    yerr=spur_cp["w_phi2"],
    fmt="o",
    color="k",
    capthick=3,
    elinewidth=3,
    capsize=3,
    alpha=0.4,
    zorder=-21,
)
p2 = ax21.errorbar(
    spur_cp["phi1"],
    spur_cp["phi2"],
    yerr=spur_cp["w_phi2"],
    fmt=".",
    c=cmap_spur(0.99),
    capsize=2,
    alpha=0.4,
    zorder=-20,
    label="Spur Control Points",
)

# Background Data
ax21.scatter(
    data["phi1"][psort][~is_gd1],
    data["phi2"][psort][~is_gd1],
    c=colors[~is_gd1],
    alpha=alphas[~is_gd1],
    s=sizes[psort][~is_gd1],
    zorder=-10,
)

# Literature
(l1,) = ax21.plot(gd1I21.phi1.degree, gd1I21.phi2.degree, **_lit1_kw, label="Ibata+21")
(l2,) = ax21.plot(gd1PB18.phi1.degree, gd1PB18.phi2.degree, **_lit2_kw, label=r"P&B18")

# Model (stream)
mpstrm = mpars.get_prefixed("stream.astrometric")
f10 = ax21.fill_between(
    data["phi1"],
    np.percentile(dmpars["stream.astrometric.phi2", "mu"], 5, axis=1),
    np.percentile(dmpars["stream.astrometric.phi2", "mu"], 95, axis=1),
    color=cmap_stream(0.99),
    alpha=0.1,
    where=stream_range,
    zorder=-9.5,
)
f11 = ax21.fill_between(
    data["phi1"],
    (mpstrm["phi2", "mu"] - np.exp(mpstrm["phi2", "ln-sigma"])),
    (mpstrm["phi2", "mu"] + np.exp(mpstrm["phi2", "ln-sigma"])),
    color=cmap_stream(0.99),
    alpha=0.25,
    where=stream_range,
    zorder=-9,
)

# Model (spur)
mpspur = mpars.get_prefixed("spur.astrometric")
f20 = ax21.fill_between(
    data["phi1"],
    np.percentile(dmpars["spur.astrometric.phi2", "mu"], 5, axis=1),
    np.percentile(dmpars["spur.astrometric.phi2", "mu"], 95, axis=1),
    color=cmap_spur(0.99),
    alpha=0.1,
    where=spur_range,
    zorder=-6.5,
)
f21 = ax21.fill_between(
    data["phi1"],
    (mpspur["phi2", "mu"] - np.exp(mpspur["phi2", "ln-sigma"])),
    (mpspur["phi2", "mu"] + np.exp(mpspur["phi2", "ln-sigma"])),
    color=cmap_spur(0.99),
    alpha=0.25,
    where=spur_range,
    zorder=-6,
)

# Data: allstream errors, then allstream data)
d1 = ax21.errorbar(
    data["phi1"][psort][is_strm],
    data["phi2"][psort][is_strm],
    xerr=data["phi1_err"][psort][is_strm],
    yerr=data["phi2_err"][psort][is_strm],
    **_stream_kw,
    zorder=-8,
)
ax21.errorbar(
    data["phi1"][psort][is_spur],
    data["phi2"][psort][is_spur],
    xerr=data["phi1_err"][psort][is_spur],
    yerr=data["phi2_err"][psort][is_spur],
    **_spur_kw,
    zorder=-8,
)
ax21.scatter(
    data["phi1"][psort][is_gd1],
    data["phi2"][psort][is_gd1],
    c=colors[is_gd1],
    alpha=alphas[is_gd1],
    s=sizes[psort][is_gd1],
    zorder=-7,
)

# Legend
legend_elements_data = [
    Line2D(
        [0],
        [0],
        marker="o",
        markeredgecolor="none",
        linestyle="none",
        markerfacecolor=cmap_stream(0.01),
        markersize=7,
    ),
    (
        d1,
        Line2D(
            [0],
            [0],
            marker="o",
            markeredgecolor="none",
            linestyle="none",
            markerfacecolor=cmap_stream(0.99),
            markersize=7,
        ),
    ),
]
legend = ax21.legend(
    [legend_elements_data, [p1, p2], [(f10, f11), (f20, f21)]],
    ["Data", "Control points", "Models"],
    numpoints=1,
    ncols=3,
    handler_map={list: HandlerTuple(ndivide=None)},
    loc="upper left",
)
ax21.add_artist(legend)

legend = plt.legend(
    [l1, l2], [l1.get_label(), l2.get_label()], numpoints=1, ncols=2, loc="lower left"
)
ax21.add_artist(legend)
ax21.locator_params(axis="x", nbins=4)


# ---------------------------------------------------------------------------
# Parallax - variance

gs3 = gs[3].subgridspec(2, 1, height_ratios=(1, 3), wspace=0.0, hspace=0.0)
ax30 = fig.add_subplot(
    gs3[0, :],
    xlim=xlims,
    xticklabels=[],
    ylabel=r"$\ln\sigma_{\varpi}$",
    aspect="auto",
    rasterization_zorder=0,
)

# Model (stream)
ln_sigma = dmpars["stream.astrometric.plx", "ln-sigma"]
ax30.fill_between(
    data["phi1"],
    np.percentile(ln_sigma, 5, axis=1),
    np.percentile(ln_sigma, 95, axis=1),
    color=cmap_stream(0.99),
    alpha=0.1,
    where=stream_range,
    zorder=-9.5,
)
ax30.scatter(
    data["phi1"][stream_range],
    np.percentile(ln_sigma, 50, axis=1)[stream_range],
    s=1,
    color=cmap_stream(0.99),
    zorder=-9,
)

# Model (spur)
ln_sigma = dmpars["spur.astrometric.plx", "ln-sigma"]
ax30.fill_between(
    data["phi1"],
    np.percentile(ln_sigma, 5, axis=1),
    np.percentile(ln_sigma, 95, axis=1),
    color=cmap_spur(0.99),
    alpha=0.1,
    where=spur_range,
    zorder=-7.5,
)
ax30.scatter(
    data["phi1"][spur_range],
    np.percentile(ln_sigma, 50, axis=1)[spur_range],
    s=1,
    color=cmap_spur(0.99),
    zorder=-7,
)

for tick in ax30.get_yticklabels():
    tick.set_verticalalignment("bottom")


# ---------------------------------------------------------------------------
# Parallax

ax31 = fig.add_subplot(
    gs3[1, :],
    xlabel="",
    ylabel=r"$\varpi$ [mas]",
    xlim=xlims,
    xticklabels=[],
    rasterization_zorder=0,
)

# Stream & Spur control points
ax31.errorbar(
    distance_cp["phi1"],
    distance_cp["parallax"],
    yerr=distance_cp["w_parallax"],
    fmt="o",
    color="k",
    capthick=3,
    elinewidth=3,
    capsize=3,
    alpha=0.4,
    zorder=-21,
)
p1 = ax31.errorbar(
    distance_cp["phi1"],
    distance_cp["parallax"],
    yerr=distance_cp["w_parallax"],
    fmt=".",
    c=cmap_stream(0.99),
    capsize=2,
    alpha=0.4,
    zorder=-20,
    label="Stream Control Points",
)

# Data background
ax31.scatter(
    data["phi1"][psort][~is_gd1],
    data["plx"][psort][~is_gd1],
    c=colors[~is_gd1],
    alpha=alphas[~is_gd1],
    s=sizes[psort][~is_gd1],
    zorder=-15,
)

# Literature
(l1,) = ax31.plot(
    gd1I21.phi1.degree, gd1I21.distance.parallax.value, **_lit1_kw, label="Ibata+21"
)
(l2,) = ax31.plot(
    gd1PB18.phi1.degree, gd1PB18.distance.parallax.value, **_lit2_kw, label="PW+19"
)


# Model (stream)
ax31.fill_between(
    data["phi1"],
    np.percentile(dmpars["stream.astrometric.plx", "mu"], 5, axis=1),
    np.percentile(dmpars["stream.astrometric.plx", "mu"], 95, axis=1),
    color=cmap_stream(0.99),
    alpha=0.1,
    where=stream_range,
    zorder=-11,
)
ax31.fill_between(
    data["phi1"],
    (mpstrm["plx", "mu"] - np.exp(mpstrm["plx", "ln-sigma"])),
    (mpstrm["plx", "mu"] + np.exp(mpstrm["plx", "ln-sigma"])),
    color=cmap_stream(0.99),
    alpha=0.25,
    where=stream_range,
    zorder=-10,
)

# Model (spur)
ax31.fill_between(
    data["phi1"],
    np.percentile(dmpars["spur.astrometric.plx", "mu"], 5, axis=1),
    np.percentile(dmpars["spur.astrometric.plx", "mu"], 95, axis=1),
    color=cmap_spur(0.99),
    alpha=0.1,
    where=spur_range,
    zorder=-9,
)
ax31.fill_between(
    data["phi1"],
    (mpspur["plx", "mu"] - np.exp(mpspur["plx", "ln-sigma"])),
    (mpspur["plx", "mu"] + np.exp(mpspur["plx", "ln-sigma"])),
    color=cmap_spur(0.99),
    alpha=0.25,
    where=spur_range,
    zorder=-8,
)

# Data allstream errors, then allstream data)
d1 = ax31.errorbar(
    data["phi1"][psort][is_strm],
    data["plx"][psort][is_strm],
    xerr=data["phi1_err"][psort][is_strm],
    yerr=data["plx_err"][psort][is_strm],
    **_stream_kw,
    zorder=-14,
)
ax31.errorbar(
    data["phi1"][psort][is_spur],
    data["plx"][psort][is_spur],
    xerr=data["phi1_err"][psort][is_spur],
    yerr=data["plx_err"][psort][is_spur],
    **_spur_kw,
    zorder=-14,
)
ax31.scatter(
    data["phi1"][psort][is_gd1],
    data["plx"][psort][is_gd1],
    c=colors[is_gd1],
    alpha=alphas[is_gd1],
    s=sizes[psort][is_gd1],
    zorder=-5,
)

ax31.set_ylim(data["plx"].min(), data["plx"].max())

# ---------------------------------------------------------------------------
# PM-Phi1 - variance

gs4 = gs[4].subgridspec(2, 1, height_ratios=(1, 3), wspace=0.0, hspace=0.0)
ax40 = fig.add_subplot(
    gs4[0, :],
    xlim=xlims,
    xticklabels=[],
    ylabel=r"$\ln\sigma_{\varpi}$",
    aspect="auto",
    rasterization_zorder=0,
)

# Model (stream)
ln_sigma = dmpars["stream.astrometric.pmphi1", "ln-sigma"]
ax40.fill_between(
    data["phi1"],
    np.percentile(ln_sigma, 5, axis=1),
    np.percentile(ln_sigma, 95, axis=1),
    color=cmap_stream(0.99),
    alpha=0.1,
    where=stream_range,
    zorder=-10,
)
ax40.scatter(
    data["phi1"][stream_range],
    np.percentile(ln_sigma, 50, axis=1)[stream_range],
    s=1,
    color=cmap_stream(0.99),
    zorder=-9,
)

# Model (spur)
ln_sigma = dmpars["spur.astrometric.pmphi1", "ln-sigma"]
ax40.fill_between(
    data["phi1"],
    np.percentile(ln_sigma, 5, axis=1),
    np.percentile(ln_sigma, 95, axis=1),
    color=cmap_spur(0.99),
    alpha=0.1,
    where=spur_range,
    zorder=-7,
)
ax40.scatter(
    data["phi1"][spur_range],
    np.percentile(ln_sigma, 50, axis=1)[spur_range],
    s=1,
    color=cmap_spur(0.99),
    zorder=-6,
)

for tick in ax40.get_yticklabels():
    tick.set_verticalalignment("bottom")


# ---------------------------------------------------------------------------
# PM-Phi1

track_pmphi1 = spline_pmphi1(data["phi1"])

ax41 = fig.add_subplot(
    gs4[1, :],
    xlabel="",
    ylabel=r"$\Delta\mu_{\phi_1}^*$ [mas yr$^{-1}$]",
    xlim=xlims,
    ylim=((data["pmphi1"] - track_pmphi1).min(), (data["pmphi1"] - track_pmphi1).max()),
    xticklabels=[],
    rasterization_zorder=0,
)

# Stream control points
ax41.errorbar(
    stream_cp["phi1"].value,
    stream_cp["pm_phi1"].value - spline_pmphi1(stream_cp["phi1"].value),
    yerr=stream_cp["w_pm_phi1"].value,
    fmt="o",
    color="k",
    capthick=3,
    elinewidth=3,
    capsize=3,
    alpha=0.4,
    zorder=-21,
)
p1 = ax41.errorbar(
    stream_cp["phi1"].value,
    stream_cp["pm_phi1"].value - spline_pmphi1(stream_cp["phi1"].value),
    yerr=stream_cp["w_pm_phi1"].value,
    fmt=".",
    c=cmap_stream(0.99),
    capsize=2,
    alpha=0.4,
    zorder=-20,
    label="Stream Control Points",
)

# Spur control points
ax41.errorbar(
    spur_cp["phi1"].value,
    spur_cp["pm_phi1"].value - spline_pmphi1(spur_cp["phi1"].value),
    yerr=spur_cp["w_pm_phi1"].value,
    fmt="o",
    color="k",
    capthick=3,
    elinewidth=3,
    capsize=3,
    alpha=0.4,
    zorder=-21,
)
p2 = ax41.errorbar(
    spur_cp["phi1"],
    spur_cp["pm_phi1"].value - spline_pmphi1(spur_cp["phi1"].value),
    yerr=spur_cp["w_pm_phi1"].value,
    fmt=".",
    c=cmap_spur(0.99),
    capsize=2,
    alpha=0.4,
    zorder=-19,
    label="Spur Control Points",
)

# Data: background
ax41.scatter(
    data["phi1"][psort][~is_gd1],
    data["pmphi1"][psort][~is_gd1] - track_pmphi1[psort][~is_gd1],
    c=colors[~is_gd1],
    alpha=alphas[~is_gd1],
    s=sizes[psort][~is_gd1],
    zorder=-15,
)

# Literature
(l1,) = ax41.plot(
    gd1I21.phi1.degree,
    gd1I21.pm_phi1_cosphi2.value - spline_pmphi1(gd1I21.phi1.degree),
    **_lit1_kw,
    label="Ibata+21",
)
(l2,) = ax41.plot(
    gd1PB18.phi1.degree,
    gd1PB18.pm_phi1_cosphi2.value - spline_pmphi1(gd1PB18.phi1.degree),
    **_lit2_kw,
    label="PW+19",
)

# Model (stream)
ax41.fill_between(
    data["phi1"],
    np.percentile(dmpars["stream.astrometric.pmphi1", "mu"], 5, axis=1) - track_pmphi1,
    np.percentile(dmpars["stream.astrometric.pmphi1", "mu"], 95, axis=1) - track_pmphi1,
    color=cmap_stream(0.99),
    alpha=0.1,
    where=stream_range,
    zorder=-10,
)
ax41.fill_between(
    data["phi1"],
    (mpstrm["pmphi1", "mu"] - np.exp(mpstrm["pmphi1", "ln-sigma"])) - track_pmphi1,
    (mpstrm["pmphi1", "mu"] + np.exp(mpstrm["pmphi1", "ln-sigma"])) - track_pmphi1,
    color=cmap_stream(0.99),
    alpha=0.25,
    where=stream_range,
    zorder=-8,
)

# Model (spur)
ax41.fill_between(
    data["phi1"],
    np.percentile(dmpars["spur.astrometric.pmphi1", "mu"], 5, axis=1) - track_pmphi1,
    np.percentile(dmpars["spur.astrometric.pmphi1", "mu"], 95, axis=1) - track_pmphi1,
    color=cmap_spur(0.99),
    alpha=0.1,
    where=spur_range,
    zorder=-9,
)
ax41.fill_between(
    data["phi1"],
    (mpspur["pmphi1", "mu"] - np.exp(mpspur["pmphi1", "ln-sigma"])) - track_pmphi1,
    (mpspur["pmphi1", "mu"] + np.exp(mpspur["pmphi1", "ln-sigma"])) - track_pmphi1,
    color=cmap_spur(0.99),
    alpha=0.25,
    where=spur_range,
    zorder=-7,
)

# Data: allstream errors, then allstream data
ax41.errorbar(
    data["phi1"][psort][is_strm],
    data["pmphi1"][psort][is_strm] - track_pmphi1[psort][is_strm],
    xerr=data["phi1_err"][psort][is_strm],
    yerr=data["pmphi1_err"][psort][is_strm],
    **_stream_kw,
    zorder=-14,
)
ax41.errorbar(
    data["phi1"][psort][is_spur],
    data["pmphi1"][psort][is_spur] - track_pmphi1[psort][is_spur],
    xerr=data["phi1_err"][psort][is_spur],
    yerr=data["pmphi1_err"][psort][is_spur],
    **_spur_kw,
    zorder=-14,
)
ax41.scatter(
    data["phi1"][psort][is_gd1],
    data["pmphi1"][psort][is_gd1] - track_pmphi1[psort][is_gd1],
    c=colors[is_gd1],
    alpha=alphas[is_gd1],
    s=sizes[psort][is_gd1],
    zorder=-5,
)


# ---------------------------------------------------------------------------
# PM-Phi2 - variance

gs5 = gs[5].subgridspec(2, 1, height_ratios=(1, 3), wspace=0.0, hspace=0.0)
ax50 = fig.add_subplot(
    gs5[0, :],
    xlim=xlims,
    xticklabels=[],
    ylabel=r"$\ln\sigma_{\varpi}$",
    aspect="auto",
    rasterization_zorder=0,
)

# Model (stream)
ln_sigma = dmpars["stream.astrometric.pmphi2", "ln-sigma"]
ax50.fill_between(
    data["phi1"],
    np.percentile(ln_sigma, 5, axis=1),
    np.percentile(ln_sigma, 95, axis=1),
    color=cmap_stream(0.99),
    alpha=0.1,
    where=stream_range,
    zorder=-10,
)
ax50.scatter(
    data["phi1"][stream_range],
    np.percentile(ln_sigma, 50, axis=1)[stream_range],
    s=1,
    color=cmap_stream(0.99),
    zorder=-8,
)

# Model (spur)
ln_sigma = dmpars["spur.astrometric.pmphi2", "ln-sigma"]
ax50.fill_between(
    data["phi1"],
    np.percentile(ln_sigma, 5, axis=1),
    np.percentile(ln_sigma, 95, axis=1),
    color=cmap_spur(0.99),
    alpha=0.1,
    where=spur_range,
    zorder=-9,
)
ax50.scatter(
    data["phi1"][spur_range],
    np.percentile(ln_sigma, 50, axis=1)[spur_range],
    s=1,
    color=cmap_spur(0.99),
    zorder=-7,
)

for tick in ax50.get_yticklabels():
    tick.set_verticalalignment("bottom")

# ---------------------------------------------------------------------------
# PM-Phi2

ax51 = fig.add_subplot(
    gs5[1, :],
    ylabel=r"$\Delta\mu_{\phi_2}$ [mas yr$^{-1}$]",
    xlim=xlims,
    rasterization_zorder=0,
    xticklabels=[],
)

track_pmphi2 = spline_pmphi2(data["phi1"])

# Data: background
ax51.scatter(
    data["phi1"][psort][~is_gd1],
    data["pmphi2"][psort][~is_gd1] - track_pmphi2[psort][~is_gd1],
    c=colors[~is_gd1],
    alpha=alphas[~is_gd1],
    s=sizes[psort][~is_gd1],
    zorder=-11,
)

# Literature
(l1,) = ax51.plot(
    gd1I21.phi1.degree,
    gd1I21.pm_phi2.value - spline_pmphi2(gd1I21.phi1.degree),
    **_lit1_kw,
    label="Ibata+21",
)
(l2,) = ax51.plot(
    gd1PB18.phi1.degree,
    gd1PB18.pm_phi2.value - spline_pmphi2(gd1PB18.phi1.degree),
    **_lit2_kw,
    label="PW+19",
)

# Model (stream)
ax51.fill_between(
    data["phi1"],
    np.percentile(dmpars["stream.astrometric.pmphi2", "mu"], 5, axis=1) - track_pmphi2,
    np.percentile(dmpars["stream.astrometric.pmphi2", "mu"], 95, axis=1) - track_pmphi2,
    color=cmap_stream(0.99),
    alpha=0.1,
    where=stream_range,
    zorder=-9,
)
ax51.fill_between(
    data["phi1"],
    (mpstrm["pmphi2", "mu"] - np.exp(mpstrm["pmphi2", "ln-sigma"])) - track_pmphi2,
    (mpstrm["pmphi2", "mu"] + np.exp(mpstrm["pmphi2", "ln-sigma"])) - track_pmphi2,
    color=cmap_stream(0.99),
    alpha=0.25,
    where=stream_range,
    zorder=-7,
)

# Model (stream)
ax51.fill_between(
    data["phi1"],
    np.percentile(dmpars["spur.astrometric.pmphi2", "mu"], 5, axis=1) - track_pmphi2,
    np.percentile(dmpars["spur.astrometric.pmphi2", "mu"], 95, axis=1) - track_pmphi2,
    color=cmap_spur(0.99),
    alpha=0.1,
    where=spur_range,
    zorder=-8,
)
ax51.fill_between(
    data["phi1"],
    (mpspur["pmphi2", "mu"] - np.exp(mpspur["pmphi2", "ln-sigma"])) - track_pmphi2,
    (mpspur["pmphi2", "mu"] + np.exp(mpspur["pmphi2", "ln-sigma"])) - track_pmphi2,
    color=cmap_spur(0.99),
    alpha=0.25,
    where=spur_range,
    zorder=-6,
)

# Data: allstream errors, then allstream data
d1 = ax51.errorbar(
    data["phi1"][psort][is_strm],
    data["pmphi2"][psort][is_strm] - track_pmphi2[psort][is_strm],
    xerr=data["phi1_err"][psort][is_strm],
    yerr=data["pmphi2_err"][psort][is_strm],
    **_stream_kw,
    zorder=-10,
)
ax51.errorbar(
    data["phi1"][psort][is_spur],
    data["pmphi2"][psort][is_spur] - track_pmphi2[psort][is_spur],
    xerr=data["phi1_err"][psort][is_spur],
    yerr=data["pmphi2_err"][psort][is_spur],
    **_spur_kw,
    zorder=-10,
)
ax51.scatter(
    data["phi1"][psort][is_gd1],
    data["pmphi2"][psort][is_gd1] - track_pmphi2[psort][is_gd1],
    c=colors[is_gd1],
    alpha=alphas[is_gd1],
    s=sizes[psort][is_gd1],
    zorder=-4,
)

ax51.set_ylim(
    (data["pmphi2"] - track_pmphi2).min(), (data["pmphi2"] - track_pmphi2).max()
)


# ---------------------------------------------------------------------------
# Distance - variance

gs6 = gs[6].subgridspec(2, 1, height_ratios=(1, 3), wspace=0.0, hspace=0.0)

ax60 = fig.add_subplot(
    gs6[0, :],
    xlim=xlims,
    xticklabels=[],
    ylabel=r"$\ln\sigma_{d}$",
    aspect="auto",
    rasterization_zorder=0,
)

# Model (stream)
ln_sigma = np.log(
    (
        Distance(
            distmod=dmpars["stream.photometric.distmod", "mu"]
            + np.exp(dmpars["stream.photometric.distmod", "ln-sigma"])
        )
        - Distance(
            distmod=dmpars["stream.photometric.distmod", "mu"]
            - np.exp(dmpars["stream.photometric.distmod", "ln-sigma"])
        )
    ).value
    / 2
)
ax60.fill_between(
    data["phi1"],
    np.percentile(ln_sigma, 5, axis=1),
    np.percentile(ln_sigma, 95, axis=1),
    color=cmap_stream(0.99),
    alpha=0.1,
    where=stream_range,
    zorder=-10,
)
ax60.scatter(
    data["phi1"][stream_range],
    np.percentile(ln_sigma, 50, axis=1)[stream_range],
    s=1,
    color=cmap_stream(0.99),
    zorder=-9,
)

# Model (spur)
ln_sigma = np.log(
    (
        Distance(
            distmod=dmpars["spur.photometric.distmod", "mu"]
            + np.exp(dmpars["spur.photometric.distmod", "ln-sigma"])
        )
        - Distance(
            distmod=dmpars["spur.photometric.distmod", "mu"]
            - np.exp(dmpars["spur.photometric.distmod", "ln-sigma"])
        )
    ).value
    / 2
)
ax60.fill_between(
    data["phi1"],
    np.percentile(ln_sigma, 5, axis=1),
    np.percentile(ln_sigma, 95, axis=1),
    color=cmap_spur(0.99),
    alpha=0.1,
    where=spur_range,
    zorder=-8,
)
ax60.scatter(
    data["phi1"][spur_range],
    np.percentile(ln_sigma, 50, axis=1)[spur_range],
    s=1,
    color=cmap_spur(0.99),
    zorder=-7,
)

for tick in ax60.get_yticklabels():
    tick.set_verticalalignment("bottom")

# ---------------------------------------------------------------------------
# Distance

ax61 = fig.add_subplot(
    gs6[1, :],
    xlabel=r"$\phi_1$ [deg]",
    ylabel=r"$d$ [kpc]",
    xlim=xlims,
    rasterization_zorder=0,
)

mu = Distance(distmod=distance_cp["distmod"])
sigma = Distance(
    distmod=(distance_cp["distmod"] + distance_cp["w_distmod"])
) - Distance(distmod=(distance_cp["distmod"] - distance_cp["w_distmod"]))

# Stream control points
ax61.errorbar(
    distance_cp["phi1"],
    mu.value,
    yerr=sigma.value,
    fmt="o",
    color="k",
    capthick=3,
    elinewidth=3,
    capsize=3,
    alpha=0.4,
    zorder=-21,
)
p1 = ax61.errorbar(
    distance_cp["phi1"],
    mu.value,
    yerr=sigma.value,
    fmt=".",
    c=cmap_stream(0.99),
    capsize=2,
    alpha=0.4,
    zorder=-20,
    label="Stream Control Points",
)

# Literature
(l1,) = ax61.plot(
    gd1I21.phi1.degree, gd1I21.distance.value, **_lit1_kw, label="Ibata+21"
)
(l2,) = ax61.plot(
    gd1PB18.phi1.degree, gd1PB18.distance.value, **_lit2_kw, label="PW+19"
)

# Model (stream)
mpstrm = mpars["stream.photometric.distmod"]
d1sm = Distance(distmod=(mpstrm["mu"] - np.exp(mpstrm["ln-sigma"])) * u.mag)
d1sp = Distance(distmod=(mpstrm["mu"] + np.exp(mpstrm["ln-sigma"])) * u.mag)

ax61.fill_between(
    data["phi1"],
    Distance(
        distmod=np.percentile(dmpars["stream.photometric.distmod", "mu"], 5, axis=1)
    ).value,
    Distance(
        distmod=np.percentile(dmpars["stream.photometric.distmod", "mu"], 95, axis=1)
    ).value,
    color=cmap_stream(0.99),
    alpha=0.1,
    where=stream_range,
    zorder=-12,
)
ax61.fill_between(
    data["phi1"],
    d1sm.to_value("kpc"),
    d1sp.to_value("kpc"),
    alpha=0.25,
    color=cmap_stream(0.99),
    where=stream_range,
    zorder=-11,
)

# Model (spur)
mpspur = mpars["spur.photometric.distmod"]
d1sm = Distance(distmod=(mpspur["mu"] - np.exp(mpspur["ln-sigma"])) * u.mag)
d1sp = Distance(distmod=(mpspur["mu"] + np.exp(mpspur["ln-sigma"])) * u.mag)

ax61.fill_between(
    data["phi1"],
    Distance(
        distmod=np.percentile(dmpars["spur.photometric.distmod", "mu"], 5, axis=1)
    ).value,
    Distance(
        distmod=np.percentile(dmpars["spur.photometric.distmod", "mu"], 95, axis=1)
    ).value,
    color=cmap_spur(0.99),
    alpha=0.1,
    where=spur_range,
    zorder=-12,
)
ax61.fill_between(
    data["phi1"],
    d1sm.to_value("kpc"),
    d1sp.to_value("kpc"),
    alpha=0.25,
    color=cmap_spur(0.99),
    where=spur_range,
    zorder=-11,
)

# ===========================================================================

fig.savefig(paths.figures / "gd1" / "results_full.pdf")
