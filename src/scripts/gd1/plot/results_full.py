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
from showyourwork.paths import user as user_paths

import stream_ml.pytorch as sml
from stream_ml.core import WEIGHT_NAME

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.gd1.datasets import data
from scripts.gd1.define_model import model
from scripts.gd1.frames import gd1_frame as frame
from scripts.helper import (
    color_by_probable_member,
    manually_set_dropout,
    p2alpha,
    recursive_iterate,
)
from scripts.mpl_colormaps import stream_cmap1 as cmap1
from scripts.mpl_colormaps import stream_cmap2 as cmap2

# =============================================================================
# Load data

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

# Load control points
stream_cp = QTable.read(paths.data / "gd1" / "control_points_stream.ecsv")
spur_cp = QTable.read(paths.data / "gd1" / "control_points_spur.ecsv")
distance_cp = QTable.read(paths.data / "gd1" / "control_points_distance.ecsv")

# Load model
model = pycopy.deepcopy(model)
model.load_state_dict(xp.load(paths.data / "gd1" / "model" / "model_0100.pt"))
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


# =============================================================================
# Likelihood

# Sorter for plotting
psort = np.argsort(allstream_prob)

# Foreground
is_strm = (allstream_prob > 0.6)[psort]


# Also evaluate the model with dropout on
with xp.no_grad():
    # turn dropout on
    model = model.train()
    manually_set_dropout(model, 0.15)
    # evaluate the model
    ldmpars = [model.unpack_params(model(data)) for i in range(100)]
    # mpars
    dmpars = sml.params.Params(
        recursive_iterate(ldmpars, ldmpars[0], reduction=lambda x: x)
    )
    mpars = sml.params.Params(recursive_iterate(ldmpars, ldmpars[0]))

    # turn dropout back off
    manually_set_dropout(model, 0)
    model = model.eval()


_is_strm = (stream_prob > 0.6) & (mpars[(f"stream.{WEIGHT_NAME}",)].numpy() > -5)
strm_range = (np.min(data["phi1"][_is_strm].numpy()) <= data["phi1"]) & (
    data["phi1"] <= np.max(data["phi1"][_is_strm].numpy())
)
_is_spur = (spur_prob > 0.6) & (mpars[(f"spur.{WEIGHT_NAME}",)].numpy() > -5)
spur_range = (np.min(data["phi1"][_is_spur].numpy()) <= data["phi1"]) & (
    data["phi1"] <= np.max(data["phi1"][_is_spur].numpy())
)


##############################################################################
# Make Figure

fig = plt.figure(figsize=(11, 15))

gs = GridSpec(
    7,
    1,
    figure=fig,
    height_ratios=(2, 3, 6.5, 6.5, 6.5, 6.5, 5),
    hspace=0.15,
    left=0.07,
    right=0.98,
    top=0.965,
    bottom=0.03,
)

# Settings
colors = color_by_probable_member(
    (stream_prob[psort], cmap1), (spur_prob[psort], cmap2)
)
alphas = p2alpha(allstream_prob[psort])
sizes = 1 + stream_prob[psort]  # range [1, 2]
xlims = (data["phi1"].min(), 10)

_stream_kw = {"ls": "none", "marker": ",", "color": cmap1(0.75), "alpha": 0.25}
_bounds_kw = {"c": "gray", "ls": "-", "lw": 2, "alpha": 0.8}
_lit1_kw = {"c": "k", "ls": "--", "alpha": 0.6}
_lit2_kw = {"c": "k", "ls": ":", "alpha": 0.6}

# ---------------------------------------------------------------------------
# Colormap

gs0 = gs[0, :].subgridspec(2, 1, height_ratios=(1, 1), hspace=0.1)

# Stream
ax00 = fig.add_subplot(gs0[0, :])
cbar = fig.colorbar(ScalarMappable(cmap=cmap1), cax=ax00, orientation="horizontal")
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")
cbar.ax.text(0.5, 0.5, "Stream Probability", ha="center", va="center", fontsize=14)

# Spur
ax01 = fig.add_subplot(gs0[1, :])
cbar = fig.colorbar(ScalarMappable(cmap=cmap2), cax=ax01, orientation="horizontal")
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
    ylim=(-7, 0),
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
    color=cmap1(0.99),
    alpha=0.25,
    where=strm_range,
)
(l1,) = ax1.plot(
    data["phi1"][strm_range],
    np.percentile(stream_wgts, 50, axis=1)[strm_range],
    c=cmap1(0.99),
    ls="--",
    lw=2,
    label="Mean",
)

# Spur
f2 = ax1.fill_between(
    data["phi1"],
    np.percentile(spur_wgt, 5, axis=1),
    np.percentile(spur_wgt, 95, axis=1),
    color=cmap2(0.99),
    alpha=0.25,
    where=spur_range,
)
(l2,) = ax1.plot(
    data["phi1"][spur_range],
    np.percentile(spur_wgt, 50, axis=1)[spur_range],
    c=cmap2(0.99),
    ls="--",
    lw=2,
)

ax1.legend(
    [(f1, f2), l1],
    [r"Models", l1.get_label()],
    numpoints=1,
    handler_map={tuple: HandlerTuple(ndivide=None)},
    loc="upper left",
)

# ---------------------------------------------------------------------------
# Phi2 - variance

gs2 = gs[2].subgridspec(2, 1, height_ratios=(1, 3), wspace=0.0, hspace=0.0)
ax20 = fig.add_subplot(
    gs2[0, :], xlim=xlims, xticklabels=[], ylabel=r"$\ln\sigma_{\phi_2}$", aspect="auto"
)

# Model (stream)
ln_sigma = dmpars["stream.astrometric.phi2", "ln-sigma"]
ax20.fill_between(
    data["phi1"],
    np.percentile(ln_sigma, 5, axis=1),
    np.percentile(ln_sigma, 95, axis=1),
    color=cmap1(0.99),
    alpha=0.25,
    where=strm_range,
)
ax20.scatter(
    data["phi1"][strm_range],
    np.percentile(ln_sigma, 50, axis=1)[strm_range],
    s=1,
    color=cmap1(0.99),
)

# Model (spur)
ln_sigma = dmpars["spur.astrometric.phi2", "ln-sigma"]
ax20.fill_between(
    data["phi1"],
    np.percentile(ln_sigma, 5, axis=1),
    np.percentile(ln_sigma, 95, axis=1),
    color=cmap2(0.99),
    alpha=0.25,
    where=spur_range,
)
ax20.scatter(
    data["phi1"][spur_range],
    np.percentile(ln_sigma, 50, axis=1)[spur_range],
    s=1,
    color=cmap2(0.99),
)

for tick in ax20.get_yticklabels():
    tick.set_verticalalignment("bottom")

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
    zorder=-21,
)
p1 = ax21.errorbar(
    stream_cp["phi1"],
    stream_cp["phi2"],
    yerr=stream_cp["w_phi2"],
    fmt=".",
    c=cmap1(0.99),
    capsize=2,
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
    zorder=-21,
)
p2 = ax21.errorbar(
    spur_cp["phi1"],
    spur_cp["phi2"],
    yerr=spur_cp["w_phi2"],
    fmt=".",
    c=cmap2(0.99),
    capsize=2,
    zorder=-20,
    label="Spur Control Points",
)

# Data (background, then stream errors, then stream data)
ax21.scatter(
    data["phi1"][psort][~is_strm],
    data["phi2"][psort][~is_strm],
    c=colors[~is_strm],
    alpha=alphas[~is_strm],
    s=sizes[psort][~is_strm],
    zorder=-10,
)
d1 = ax21.errorbar(
    data["phi1"][psort][is_strm],
    data["phi2"][psort][is_strm],
    xerr=data["phi1_err"][psort][is_strm],
    yerr=data["phi2_err"][psort][is_strm],
    **_stream_kw,
    zorder=-9,
)
ax21.scatter(
    data["phi1"][psort][is_strm],
    data["phi2"][psort][is_strm],
    c=colors[is_strm],
    alpha=alphas[is_strm],
    s=sizes[psort][is_strm],
    zorder=-8,
)

# Literature
(l1,) = ax21.plot(gd1I21.phi1.degree, gd1I21.phi2.degree, **_lit1_kw, label="Ibata+21")
(l2,) = ax21.plot(gd1PB18.phi1.degree, gd1PB18.phi2.degree, **_lit2_kw, label="PW+19")

# Model (stream)
mpstrm = mpars.get_prefixed("stream.astrometric")
f1 = ax21.fill_between(
    data["phi1"],
    (mpstrm["phi2", "mu"] - xp.exp(mpstrm["phi2", "ln-sigma"])),
    (mpstrm["phi2", "mu"] + xp.exp(mpstrm["phi2", "ln-sigma"])),
    color=cmap1(0.99),
    alpha=0.25,
    where=strm_range,
)

# Model (spur)
mpspur = mpars.get_prefixed("spur.astrometric")
f2 = ax21.fill_between(
    data["phi1"],
    (mpspur["phi2", "mu"] - xp.exp(mpspur["phi2", "ln-sigma"])),
    (mpspur["phi2", "mu"] + xp.exp(mpspur["phi2", "ln-sigma"])),
    color=cmap2(0.99),
    alpha=0.25,
    where=spur_range,
)

# Legend
legend_elements_data = [
    Line2D(
        [0],
        [0],
        marker="o",
        markeredgecolor="none",
        linestyle="none",
        markerfacecolor=cmap1(0.01),
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
            markerfacecolor=cmap1(0.99),
            markersize=7,
        ),
    ),
]
legend = ax21.legend(
    [legend_elements_data, [p1, p2], [f1, f2]],
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


# ---------------------------------------------------------------------------
# Parallax - variance

gs3 = gs[3].subgridspec(2, 1, height_ratios=(1, 3), wspace=0.0, hspace=0.0)
ax30 = fig.add_subplot(
    gs3[0, :], xlim=xlims, xticklabels=[], ylabel=r"$\ln\sigma_{\varpi}$", aspect="auto"
)

# Model (stream)
ln_sigma = dmpars["stream.astrometric.plx", "ln-sigma"]
ax30.fill_between(
    data["phi1"],
    np.percentile(ln_sigma, 5, axis=1),
    np.percentile(ln_sigma, 95, axis=1),
    color=cmap1(0.99),
    alpha=0.25,
    where=strm_range,
)
ax30.scatter(
    data["phi1"][strm_range],
    np.percentile(ln_sigma, 50, axis=1)[strm_range],
    s=1,
    color=cmap1(0.99),
)

# Model (spur)
ln_sigma = dmpars["spur.astrometric.plx", "ln-sigma"]
ax30.fill_between(
    data["phi1"],
    np.percentile(ln_sigma, 5, axis=1),
    np.percentile(ln_sigma, 95, axis=1),
    color=cmap2(0.99),
    alpha=0.25,
    where=spur_range,
)
ax30.scatter(
    data["phi1"][spur_range],
    np.percentile(ln_sigma, 50, axis=1)[spur_range],
    s=1,
    color=cmap2(0.99),
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

# Stream control points
ax31.errorbar(
    distance_cp["phi1"],
    distance_cp["parallax"],
    yerr=distance_cp["w_parallax"],
    fmt="o",
    color="k",
    capthick=3,
    elinewidth=3,
    capsize=3,
    zorder=-21,
)
p1 = ax31.errorbar(
    distance_cp["phi1"],
    distance_cp["parallax"],
    yerr=distance_cp["w_parallax"],
    fmt=".",
    c=cmap1(0.99),
    capsize=2,
    zorder=-20,
    label="Stream Control Points",
)
# Spur control points
ax31.errorbar(
    distance_cp["phi1"],
    distance_cp["parallax"],
    yerr=distance_cp["w_parallax"],
    fmt="o",
    color="k",
    capthick=3,
    elinewidth=3,
    capsize=3,
    zorder=-21,
)
p2 = ax31.errorbar(
    distance_cp["phi1"],
    distance_cp["parallax"],
    yerr=distance_cp["w_parallax"],
    fmt=".",
    c=cmap2(0.99),
    capsize=2,
    zorder=-20,
    label="Spur Control Points",
)

# Data
d1 = ax31.scatter(
    data["phi1"][psort], data["plx"][psort], c=colors, alpha=alphas, s=2, zorder=-10
)

# Model
f1 = ax31.fill_between(
    data["phi1"],
    (mpstrm["plx", "mu"] - xp.exp(mpstrm["plx", "ln-sigma"])),
    (mpstrm["plx", "mu"] + xp.exp(mpstrm["plx", "ln-sigma"])),
    color=cmap1(0.99),
    alpha=0.25,
    where=strm_range,
)
f2 = ax31.fill_between(
    data["phi1"],
    (mpspur["plx", "mu"] - xp.exp(mpspur["plx", "ln-sigma"])),
    (mpspur["plx", "mu"] + xp.exp(mpspur["plx", "ln-sigma"])),
    color=cmap2(0.99),
    alpha=0.25,
    where=spur_range,
)

# ---------------------------------------------------------------------------
# PM-Phi1 - variance

gs4 = gs[4].subgridspec(2, 1, height_ratios=(1, 3), wspace=0.0, hspace=0.0)
ax40 = fig.add_subplot(
    gs4[0, :], xlim=xlims, xticklabels=[], ylabel=r"$\ln\sigma_{\varpi}$", aspect="auto"
)

# Model (stream)
ln_sigma = dmpars["stream.astrometric.pmphi1", "ln-sigma"]
ax40.fill_between(
    data["phi1"],
    np.percentile(ln_sigma, 5, axis=1),
    np.percentile(ln_sigma, 95, axis=1),
    color=cmap1(0.99),
    alpha=0.25,
    where=strm_range,
)
ax40.scatter(
    data["phi1"][strm_range],
    np.percentile(ln_sigma, 50, axis=1)[strm_range],
    s=1,
    color=cmap1(0.99),
)

# Model (spur)
ln_sigma = dmpars["spur.astrometric.pmphi1", "ln-sigma"]
ax40.fill_between(
    data["phi1"],
    np.percentile(ln_sigma, 5, axis=1),
    np.percentile(ln_sigma, 95, axis=1),
    color=cmap2(0.99),
    alpha=0.25,
    where=spur_range,
)
ax40.scatter(
    data["phi1"][spur_range],
    np.percentile(ln_sigma, 50, axis=1)[spur_range],
    s=1,
    color=cmap2(0.99),
)

for tick in ax40.get_yticklabels():
    tick.set_verticalalignment("bottom")


# ---------------------------------------------------------------------------
# PM-Phi1

ax41 = fig.add_subplot(
    gs4[1, :],
    xlabel="",
    ylabel=r"$\mu_{\phi_1}^*$ [mas yr$^{-1}$]",
    xlim=xlims,
    ylim=(data["pmphi1"].min(), data["pmphi1"].max()),
    xticklabels=[],
    rasterization_zorder=0,
)

# Stream control points
ax41.errorbar(
    stream_cp["phi1"],
    stream_cp["pm_phi1"],
    yerr=stream_cp["w_pm_phi1"],
    fmt="o",
    color="k",
    capthick=3,
    elinewidth=3,
    capsize=3,
    zorder=-21,
)
p1 = ax41.errorbar(
    stream_cp["phi1"],
    stream_cp["pm_phi1"],
    yerr=stream_cp["w_pm_phi1"],
    fmt=".",
    c=cmap1(0.99),
    capsize=2,
    zorder=-20,
    label="Stream Control Points",
)
# Spur control points
ax41.errorbar(
    spur_cp["phi1"],
    spur_cp["pm_phi1"],
    yerr=spur_cp["w_pm_phi1"],
    fmt="o",
    color="k",
    capthick=3,
    elinewidth=3,
    capsize=3,
    zorder=-21,
)
p2 = ax41.errorbar(
    spur_cp["phi1"],
    spur_cp["pm_phi1"],
    yerr=spur_cp["w_pm_phi1"],
    fmt=".",
    c=cmap2(0.99),
    capsize=2,
    zorder=-20,
    label="Spur Control Points",
)

# Data
d1 = ax41.scatter(
    data["phi1"][psort], data["pmphi1"][psort], c=colors, alpha=alphas, s=2, zorder=-10
)
# Model (stream)
f1 = ax41.fill_between(
    data["phi1"],
    (mpstrm["pmphi1", "mu"] - xp.exp(mpstrm["pmphi1", "ln-sigma"])),
    (mpstrm["pmphi1", "mu"] + xp.exp(mpstrm["pmphi1", "ln-sigma"])),
    color=cmap1(0.99),
    alpha=0.25,
    where=strm_range,
)
# Model (spur)
f2 = ax41.fill_between(
    data["phi1"],
    (mpspur["pmphi1", "mu"] - xp.exp(mpspur["pmphi1", "ln-sigma"])),
    (mpspur["pmphi1", "mu"] + xp.exp(mpspur["pmphi1", "ln-sigma"])),
    color=cmap2(0.99),
    alpha=0.25,
    where=spur_range,
)


# ---------------------------------------------------------------------------
# PM-Phi2 - variance

gs5 = gs[5].subgridspec(2, 1, height_ratios=(1, 3), wspace=0.0, hspace=0.0)
ax50 = fig.add_subplot(
    gs5[0, :], xlim=xlims, xticklabels=[], ylabel=r"$\ln\sigma_{\varpi}$", aspect="auto"
)

# Model (stream)
ln_sigma = dmpars["stream.astrometric.pmphi2", "ln-sigma"]
ax50.fill_between(
    data["phi1"],
    np.percentile(ln_sigma, 5, axis=1),
    np.percentile(ln_sigma, 95, axis=1),
    color=cmap1(0.99),
    alpha=0.25,
    where=strm_range,
)
ax50.scatter(
    data["phi1"][strm_range],
    np.percentile(ln_sigma, 50, axis=1)[strm_range],
    s=1,
    color=cmap1(0.99),
)

# Model (spur)
ln_sigma = dmpars["spur.astrometric.pmphi2", "ln-sigma"]
ax50.fill_between(
    data["phi1"],
    np.percentile(ln_sigma, 5, axis=1),
    np.percentile(ln_sigma, 95, axis=1),
    color=cmap2(0.99),
    alpha=0.25,
    where=spur_range,
)
ax50.scatter(
    data["phi1"][spur_range],
    np.percentile(ln_sigma, 50, axis=1)[spur_range],
    s=1,
    color=cmap2(0.99),
)

for tick in ax50.get_yticklabels():
    tick.set_verticalalignment("bottom")

# ---------------------------------------------------------------------------
# PM-Phi2

ax51 = fig.add_subplot(
    gs5[1, :],
    ylabel=r"$\mu_{\phi_2}$ [mas yr$^{-1}$]",
    xlim=xlims,
    rasterization_zorder=0,
    xticklabels=[],
)

# Data
ax51.scatter(
    data["phi1"][psort], data["pmphi2"][psort], c=colors, alpha=alphas, s=2, zorder=-10
)
# Model (stream)
f1 = ax51.fill_between(
    data["phi1"],
    (mpstrm["pmphi2", "mu"] - xp.exp(mpstrm["pmphi2", "ln-sigma"])),
    (mpstrm["pmphi2", "mu"] + xp.exp(mpstrm["pmphi2", "ln-sigma"])),
    color=cmap1(0.99),
    alpha=0.25,
    where=strm_range,
)
# Model (stream)
f2 = ax51.fill_between(
    data["phi1"],
    (mpspur["pmphi2", "mu"] - xp.exp(mpspur["pmphi2", "ln-sigma"])),
    (mpspur["pmphi2", "mu"] + xp.exp(mpspur["pmphi2", "ln-sigma"])),
    color=cmap2(0.99),
    alpha=0.25,
    where=spur_range,
)


# ---------------------------------------------------------------------------
# Distance

ax6 = fig.add_subplot(
    gs[6, :], xlabel=r"$\phi_1$ [deg]", ylabel=r"$d$ [kpc]", xlim=xlims
)

# Model (stream)
mpstrm = mpars["stream.photometric.distmod"]
d2sm = Distance(distmod=(mpstrm["mu"] - 2 * xp.exp(mpstrm["ln-sigma"])) * u.mag)
d2sp = Distance(distmod=(mpstrm["mu"] + 2 * xp.exp(mpstrm["ln-sigma"])) * u.mag)
d1sm = Distance(distmod=(mpstrm["mu"] - xp.exp(mpstrm["ln-sigma"])) * u.mag)
d1sp = Distance(distmod=(mpstrm["mu"] + xp.exp(mpstrm["ln-sigma"])) * u.mag)

ax6.fill_between(
    data["phi1"],
    d2sm.to_value("kpc"),
    d2sp.to_value("kpc"),
    alpha=0.15,
    color=cmap1(0.99),
    where=strm_range,
)
f1 = ax6.fill_between(
    data["phi1"],
    d1sm.to_value("kpc"),
    d1sp.to_value("kpc"),
    alpha=0.25,
    color=cmap1(0.99),
    where=strm_range,
)

# Model (spur)
mpspur = mpars["spur.photometric.distmod"]
d2sm = Distance(distmod=(mpspur["mu"] - 2 * xp.exp(mpspur["ln-sigma"])) * u.mag)
d2sp = Distance(distmod=(mpspur["mu"] + 2 * xp.exp(mpspur["ln-sigma"])) * u.mag)
d1sm = Distance(distmod=(mpspur["mu"] - xp.exp(mpspur["ln-sigma"])) * u.mag)
d1sp = Distance(distmod=(mpspur["mu"] + xp.exp(mpspur["ln-sigma"])) * u.mag)

ax6.fill_between(
    data["phi1"],
    d2sm.to_value("kpc"),
    d2sp.to_value("kpc"),
    alpha=0.15,
    color=cmap2(0.99),
    where=spur_range,
)
f2 = ax6.fill_between(
    data["phi1"],
    d1sm.to_value("kpc"),
    d1sp.to_value("kpc"),
    alpha=0.25,
    color=cmap2(0.99),
    where=spur_range,
)

fig.savefig(paths.figures / "gd1" / "results_full.pdf")
