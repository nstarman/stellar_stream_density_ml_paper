"""Plot the trained pal5 model."""

from __future__ import annotations

import copy as pycopy
import sys

import galstreams
import matplotlib.pyplot as plt
import numpy as np
import torch as xp
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

from scripts.helper import manually_set_dropout, p2alpha, recursive_iterate
from scripts.mpl_colormaps import stream_cmap1 as cmap1
from scripts.pal5.datasets import data, masks
from scripts.pal5.define_model import model
from scripts.pal5.frames import pal5_frame as frame

# =============================================================================
# Load data

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

# Control points
stream_cp = QTable.read(paths.data / "pal5" / "control_points_stream.ecsv")

# Load model
model = pycopy.deepcopy(model)
model.load_state_dict(xp.load(paths.data / "pal5" / "model" / "model_10800.pt"))
model = model.eval()

# Load results from 4-likelihoods.py
lik_tbl = QTable.read(paths.data / "pal5" / "membership_likelhoods.ecsv")
stream_prob = np.array(lik_tbl["stream (50%)"])
stream_wgt = np.array(lik_tbl["stream.ln-weight"])

# Progenitor
progenitor_prob = np.zeros(len(masks))
progenitor_prob[~masks["Pal5"]] = 1

# galstreams
allstreams = galstreams.MWStreams(implement_Off=True)
pal5I21 = allstreams["Pal5-I21"].track.transform_to(frame)
pal5PW19 = allstreams["Pal5-PW19"].track.transform_to(frame)

# =============================================================================
# Likelihood

# Sorter for plotting
psort = np.argsort(stream_prob)

# Foreground
is_strm_ = stream_prob > 0.6
is_strm = is_strm_[psort]
strm_range = (data["phi1"][is_strm_].min() <= data["phi1"]) & (
    data["phi1"] <= data["phi1"][is_strm_].max()
)

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

##############################################################################
# Make Figure

fig = plt.figure(figsize=(11, 15))

gs = GridSpec(
    6,
    1,
    figure=fig,
    height_ratios=(1, 3, 6.5, 5, 6.5, 6.5),
    hspace=0.15,
    left=0.07,
    right=0.98,
    top=0.965,
    bottom=0.03,
)

# Settings
colors = cmap1(stream_prob[psort])
alphas = p2alpha(stream_prob[psort])
sizes = 1 + stream_prob[psort]  # range [1, 2]
xlims = (data["phi1"].min(), data["phi1"].max())

_stream_kw = {"ls": "none", "marker": ",", "color": cmap1(0.75), "alpha": 0.25}
_bounds_kw = {"c": "gray", "ls": "-", "lw": 2, "alpha": 0.8}
_lit1_kw = {"c": "k", "ls": "--", "alpha": 0.6}
_lit2_kw = {"c": "k", "ls": ":", "alpha": 0.6}

# ---------------------------------------------------------------------------
# Colormap

# Stream
ax00 = fig.add_subplot(gs[0, :])
cbar = fig.colorbar(ScalarMappable(cmap=cmap1), cax=ax00, orientation="horizontal")
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")
cbar.ax.text(0.5, 0.5, "Stream Probability", ha="center", va="center", fontsize=14)


# ---------------------------------------------------------------------------
# Weight plot

ax01 = fig.add_subplot(
    gs[1, :],
    xlim=xlims,
    xticklabels=[],
    ylabel=r"$\ln f_{\rm stream}$",
    ylim=(-6, 0),
    rasterization_zorder=0,
)

# Upper and lower bounds
_bounds = model.params[(f"stream.{WEIGHT_NAME}",)].bounds
ax01.axhline(_bounds.lower[0], **_bounds_kw)
ax01.axhline(_bounds.upper[0], **_bounds_kw)

# Stream
f1 = ax01.fill_between(
    data["phi1"],
    np.percentile(stream_wgt, 5, axis=1),
    np.percentile(stream_wgt, 95, axis=1),
    color=cmap1(0.99),
    alpha=0.25,
)
(l1,) = ax01.plot(
    data["phi1"],
    np.percentile(stream_wgt, 50, axis=1),
    c=cmap1(0.99),
    ls="--",
    lw=2,
    label="Mean",
)

ax01.legend([(f1, l1)], [r"Model"], numpoints=1, loc="upper left")


# ---------------------------------------------------------------------------
# Phi2 - variance

gs2 = gs[2].subgridspec(2, 1, height_ratios=(1, 3), wspace=0.0, hspace=0.0)
ax02 = fig.add_subplot(
    gs2[0, :], xlim=xlims, xticklabels=[], ylabel=r"$\ln\sigma_{\phi_2}$", aspect="auto"
)

ln_sigma = dmpars["stream.astrometric.phi2", "ln-sigma"]

# Model
ax02.fill_between(
    data["phi1"],
    np.percentile(ln_sigma, 5, axis=1),
    np.percentile(ln_sigma, 95, axis=1),
    color=cmap1(0.99),
    alpha=0.25,
    where=strm_range,
)
ax02.scatter(
    data["phi1"][strm_range],
    np.percentile(ln_sigma, 50, axis=1)[strm_range],
    s=1,
    color=cmap1(0.99),
)

for tick in ax02.get_yticklabels():
    tick.set_verticalalignment("bottom")
ax02.set_ylim(-3, 0)

# ---------------------------------------------------------------------------
# Phi2

mpa = mpars.get_prefixed("stream.astrometric")

ax03 = fig.add_subplot(
    gs2[1, :],
    xlabel="",
    xlim=xlims,
    xticklabels=[],
    ylabel=r"$\phi_2$ [deg]",
    ylim=(data["phi2"].min(), 6),
    rasterization_zorder=0,
    aspect="auto",
)

# Stream control points
p1 = ax03.errorbar(
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
p2 = ax03.errorbar(
    stream_cp["phi1"],
    stream_cp["phi2"],
    yerr=stream_cp["w_phi2"],
    fmt=".",
    c=cmap1(0.99),
    capsize=2,
    zorder=-20,
    label="Stream Control Points",
)

# Data (background, then stream errors, then stream data)
ax03.scatter(
    data["phi1"][psort][~is_strm],
    data["phi2"][psort][~is_strm],
    c=colors[~is_strm],
    alpha=alphas[~is_strm],
    s=sizes[psort][~is_strm],
    zorder=-10,
)
d1 = ax03.errorbar(
    data["phi1"][psort][is_strm],
    data["phi2"][psort][is_strm],
    xerr=data["phi1_err"][psort][is_strm],
    yerr=data["phi2_err"][psort][is_strm],
    **_stream_kw,
    zorder=-9,
)
ax03.scatter(
    data["phi1"][psort][is_strm],
    data["phi2"][psort][is_strm],
    c=colors[is_strm],
    alpha=alphas[is_strm],
    s=sizes[psort][is_strm],
    zorder=-8,
)

# Literature
(l1,) = ax03.plot(
    pal5I21.phi1.degree, pal5I21.phi2.degree, **_lit1_kw, label="Ibata+21"
)
(l2,) = ax03.plot(pal5PW19.phi1.degree, pal5PW19.phi2.degree, **_lit2_kw, label="PW+19")

# Model
f1 = ax03.fill_between(
    data["phi1"],
    (mpa["phi2", "mu"] - xp.exp(mpa["phi2", "ln-sigma"])),
    (mpa["phi2", "mu"] + xp.exp(mpa["phi2", "ln-sigma"])),
    color=cmap1(0.99),
    alpha=0.25,
    where=strm_range,
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
legend = plt.legend(
    [legend_elements_data, (p1, p2), f1],
    ["Data", "Guides", "Model"],
    numpoints=1,
    ncols=3,
    loc="upper left",
    handler_map={list: HandlerTuple(ndivide=None)},
)
ax03.add_artist(legend)

legend = plt.legend(
    [l1, l2], [l1.get_label(), l2.get_label()], numpoints=1, ncols=2, loc="lower left"
)
ax03.add_artist(legend)


# ---------------------------------------------------------------------------
# Parallax - variance

# ---------------------------------------------------------------------------
# Parallax

ax05 = fig.add_subplot(
    gs[3, :],
    xlabel="",
    ylabel=r"$\varpi$ [mas]",
    xlim=xlims,
    ylim=(max(data["plx"].min(), -1), data["plx"].max()),
    xticklabels=[],
    rasterization_zorder=0,
)

# Data
ax05.scatter(
    data["phi1"][psort][~is_strm],
    data["plx"][psort][~is_strm],
    c=colors[~is_strm],
    alpha=alphas[~is_strm],
    s=sizes[psort][~is_strm],
    zorder=-10,
)
ax05.errorbar(
    data["phi1"][psort][is_strm],
    data["plx"][psort][is_strm],
    xerr=data["phi1_err"][psort][is_strm],
    yerr=data["plx_err"][psort][is_strm],
    **_stream_kw,
    zorder=-9,
)
ax05.scatter(
    data["phi1"][psort][is_strm],
    data["plx"][psort][is_strm],
    c=colors[is_strm],
    alpha=alphas[is_strm],
    s=sizes[psort][is_strm],
    zorder=-8,
)

# Literature
ax05.plot(pal5I21.phi1.degree, pal5I21.distance.parallax, **_lit1_kw, label="Ibata+21")
ax05.plot(pal5PW19.phi1.degree, pal5PW19.distance.parallax, **_lit2_kw, label="PW+19")


# ---------------------------------------------------------------------------
# PM-Phi1 - variance

gs6 = gs[4].subgridspec(2, 1, height_ratios=(1, 3), wspace=0.0, hspace=0.0)
ax06 = fig.add_subplot(
    gs6[0, :],
    xlim=xlims,
    xticklabels=[],
    ylabel=r"$\ln\sigma_{\mu_{\phi_1}^*}$",
    aspect="auto",
)

ln_sigma = dmpars["stream.astrometric.pmphi1", "ln-sigma"]

# Model
ax06.fill_between(
    data["phi1"],
    (np.percentile(ln_sigma, 5, axis=1)),
    (np.percentile(ln_sigma, 95, axis=1)),
    color=cmap1(0.99),
    alpha=0.25,
    where=strm_range,
)
ax06.scatter(
    data["phi1"][strm_range],
    (np.percentile(ln_sigma, 50, axis=1)[strm_range]),
    s=1,
    color=cmap1(0.99),
)

for tick in ax06.get_yticklabels():
    tick.set_verticalalignment("bottom")


# ---------------------------------------------------------------------------
# PM-Phi1

ax07 = fig.add_subplot(
    gs6[1, :],
    xlabel="",
    xlim=xlims,
    ylabel=r"$\mu_{\phi_1}^*$ [mas yr$^{-1}$]",
    ylim=(data["pmphi1"].min(), data["pmphi1"].max()),
    rasterization_zorder=0,
    xticklabels=[],
    aspect="auto",
)

# Data
ax07.scatter(
    data["phi1"][psort][~is_strm],
    data["pmphi1"][psort][~is_strm],
    c=colors[~is_strm],
    alpha=alphas[~is_strm],
    s=sizes[psort][~is_strm],
    zorder=-10,
)
ax07.errorbar(
    data["phi1"][psort][is_strm],
    data["pmphi1"][psort][is_strm],
    xerr=data["phi1_err"][psort][is_strm],
    yerr=data["pmphi1_err"][psort][is_strm],
    **_stream_kw,
    zorder=-9,
)
ax07.scatter(
    data["phi1"][psort][is_strm],
    data["pmphi1"][psort][is_strm],
    c=colors[is_strm],
    alpha=alphas[is_strm],
    s=sizes[psort][is_strm],
    zorder=-8,
)

# Literature
ax07.plot(
    pal5I21.phi1.degree,
    pal5I21.pm_phi1_cosphi2.value,
    **_lit1_kw,
    label="Ibata+21",
)
ax07.plot(
    pal5PW19.phi1.degree,
    pal5PW19.pm_phi1_cosphi2.value,
    **_lit2_kw,
    label="PW+19",
)

# Model
ax07.fill_between(
    data["phi1"],
    (mpa["pmphi1", "mu"] - xp.exp(mpa["pmphi1", "ln-sigma"])),
    (mpa["pmphi1", "mu"] + xp.exp(mpa["pmphi1", "ln-sigma"])),
    color=cmap1(0.99),
    alpha=0.25,
    where=strm_range,
)

# ---------------------------------------------------------------------------
# PM-Phi2 - variance

gs8 = gs[5].subgridspec(2, 1, height_ratios=(1, 3), wspace=0.0, hspace=0.0)
ax08 = fig.add_subplot(
    gs8[0, :],
    xlim=xlims,
    xticklabels=[],
    ylabel=r"$ln\sigma_{\mu_{\phi_2}}$",
    aspect="auto",
)

ln_sigma = dmpars["stream.astrometric.pmphi2", "ln-sigma"]

# Model
ax08.fill_between(
    data["phi1"],
    (np.percentile(ln_sigma, 5, axis=1)),
    (np.percentile(ln_sigma, 95, axis=1)),
    color=cmap1(0.99),
    alpha=0.25,
    where=strm_range,
)
ax08.scatter(
    data["phi1"][strm_range],
    (np.percentile(ln_sigma, 50, axis=1))[strm_range],
    s=1,
    color=cmap1(0.99),
)

for tick in ax08.get_yticklabels():
    tick.set_verticalalignment("bottom")


# ---------------------------------------------------------------------------
# PM-Phi2

ax09 = fig.add_subplot(
    gs8[1, :],
    xlabel=r"$\phi_1$ [deg]",
    xlim=xlims,
    ylabel=r"$\mu_{\phi_2}$ [mas yr$^{-1}$]",
    ylim=(data["pmphi2"].min(), data["pmphi2"].max()),
    rasterization_zorder=0,
    aspect="auto",
)

# Data
ax09.scatter(
    data["phi1"][psort][~is_strm],
    data["pmphi2"][psort][~is_strm],
    c=colors[~is_strm],
    alpha=alphas[~is_strm],
    s=sizes[psort][~is_strm],
    zorder=-10,
)
ax09.errorbar(
    data["phi1"][psort][is_strm],
    data["pmphi2"][psort][is_strm],
    xerr=data["phi1_err"][psort][is_strm],
    yerr=data["pmphi2_err"][psort][is_strm],
    **_stream_kw,
    zorder=-9,
)
ax09.scatter(
    data["phi1"][psort][is_strm],
    data["pmphi2"][psort][is_strm],
    c=colors[is_strm],
    alpha=alphas[is_strm],
    s=sizes[psort][is_strm],
    zorder=-8,
)

# Literature
ax09.plot(pal5I21.phi1.degree, pal5I21.pm_phi2.value, **_lit1_kw, label="Ibata+21")
ax09.plot(pal5PW19.phi1.degree, pal5PW19.pm_phi2.value, **_lit2_kw, label="PW+19")

# Model
ax09.fill_between(
    data["phi1"],
    (mpa["pmphi2", "mu"] - xp.exp(mpa["pmphi2", "ln-sigma"])),
    (mpa["pmphi2", "mu"] + xp.exp(mpa["pmphi2", "ln-sigma"])),
    color=cmap1(0.99),
    alpha=0.25,
    where=strm_range,
)

fig.savefig(paths.figures / "pal5" / "results_full.pdf")
