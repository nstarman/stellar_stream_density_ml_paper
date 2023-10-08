"""Plot the trained pal5 model."""

from __future__ import annotations

import copy as pycopy
import sys
from dataclasses import replace

import galstreams
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch as xp
from astropy.table import QTable
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
from scripts.pal5.datasets import where as model_where
from scripts.pal5.define_model import model
from scripts.pal5.frames import pal5_frame as frame

# =============================================================================
# Load data

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

# Load model
model = pycopy.deepcopy(model)
model = replace(  # TODO: remove this
    model,
    net=sml.nn.sequential(
        data=1, hidden_features=16, layers=4, features=1, dropout=0.15
    ),
)
model.load_state_dict(xp.load(paths.data / "pal5" / "model" / "model_10600.pt"))
model = model.eval()

# Control points
stream_cp = QTable.read(paths.data / "pal5" / "control_points_stream.ecsv")

# Progenitor
progenitor_prob = np.zeros(len(masks))
progenitor_prob[~masks["Pal5"]] = 1

# galstreams
allstreams = galstreams.MWStreams(implement_Off=True)
pal5I21 = allstreams["Pal5-I21"].track.transform_to(frame)
pal5PW19 = allstreams["Pal5-PW19"].track.transform_to(frame)

# =============================================================================
# Likelihood

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

    # Likelihoods
    stream_lnlik = model.component_ln_posterior(
        "stream", mpars, data, where=model_where
    )
    bkg_lnlik = model.component_ln_posterior(
        "background", mpars, data, where=model_where
    )
    # tot_lnlik = model.ln_posterior(mpars, data, where=model_where)  # FIXME
    tot_lnlik = xp.logaddexp(stream_lnlik, bkg_lnlik)

    # turn dropout back off
    manually_set_dropout(model, 0)
    model = model.eval()

# Weight
stream_weight = mpars[(f"stream.{WEIGHT_NAME}",)]
where = stream_weight > -10  # everything has weight > 0

# Probabilities
bkg_prob = xp.exp(bkg_lnlik - tot_lnlik)
stream_prob = xp.exp(stream_lnlik - tot_lnlik)
stream_prob[~where] = 0
bkg_prob[~where] = 1

# Sorter for plotting
psort = np.argsort(stream_prob)

# Foreground
is_stream = stream_prob[psort] > 0.6

##############################################################################
# Make Figure

fig = plt.figure(figsize=(11, 15))

gs0 = GridSpec(
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

colors = cmap1(stream_prob[psort])
alphas = p2alpha(stream_prob[psort])
xlim = (data["phi1"].min(), data["phi1"].max())

_stream_kw = {"ls": "none", "marker": ",", "color": cmap1(0.75), "alpha": 0.25}
_bounds_kw = {"c": "gray", "ls": "-", "lw": 2, "alpha": 0.8}
_lit_kw = {"c": "k", "ls": "--", "alpha": 0.6}

# ---------------------------------------------------------------------------
# Colormap

# Stream probability
ax00 = fig.add_subplot(gs0[0, :])
cbar = fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap1), cax=ax00, orientation="horizontal"
)
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")
cbar.ax.text(0.5, 0.5, "Stream Probability", ha="center", va="center", fontsize=14)


# ---------------------------------------------------------------------------
# Weight plot

ax01 = fig.add_subplot(
    gs0[1, :],
    ylabel="Stream fraction",
    xlim=xlim,
    ylim=(1e-4, 1),
    xticklabels=[],
    yscale="log",
    rasterization_zorder=0,
)

# Upper and lower bounds
_bounds = model.params[(f"stream.{WEIGHT_NAME}",)].bounds
ax01.axhline(np.exp(_bounds.lower[0]), **_bounds_kw)
ax01.axhline(np.exp(_bounds.upper[0]), **_bounds_kw)

# # 15% dropout
f1 = ax01.fill_between(
    data["phi1"],
    np.exp(np.percentile(dmpars[(f"stream.{WEIGHT_NAME}",)], 5, axis=1)),
    np.exp(np.percentile(dmpars[(f"stream.{WEIGHT_NAME}",)], 95, axis=1)),
    color=cmap1(0.99),
    alpha=0.25,
)
# Mean
(l1,) = ax01.plot(
    data["phi1"],
    np.exp(dmpars["stream.ln-weight"].mean(1)),
    c="salmon",
    ls="--",
    lw=2,
    label="Mean",
)

ax01.legend([(f1, l1)], [r"Model "], numpoints=1, loc="upper left")


# ---------------------------------------------------------------------------
# Phi2 - variance

mpa = mpars.get_prefixed("stream.astrometric")

gs02 = gs0[2].subgridspec(2, 1, height_ratios=(1, 3), wspace=0.0, hspace=0.0)
ax02 = fig.add_subplot(
    gs02[0, :], xlim=xlim, xticklabels=[], ylabel=r"$\sigma_{\phi_2}$", aspect="auto"
)

# Model
ax02.fill_between(
    data["phi1"],
    np.exp(np.percentile(dmpars["stream.astrometric.phi2", "ln-sigma"], 5, axis=1)),
    np.exp(np.percentile(dmpars["stream.astrometric.phi2", "ln-sigma"], 95, axis=1)),
    color=cmap1(0.99),
    alpha=0.25,
    where=where,
)
ax02.scatter(
    data["phi1"][where],
    np.exp(mpa["phi2", "ln-sigma"][where]),
    s=1,
    c=cmap1(0.99),
)

for tick in ax02.get_yticklabels():
    tick.set_verticalalignment("bottom")

# ---------------------------------------------------------------------------
# Phi2

ax03 = fig.add_subplot(
    gs02[1, :],
    xlabel="",
    xlim=xlim,
    ylabel=r"$\phi_2$ [deg]",
    ylim=(data["phi2"].min(), data["phi2"].max()),
    rasterization_zorder=0,
    xticklabels=[],
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

# Data
ax03.scatter(
    data["phi1"][psort][~is_stream],
    data["phi2"][psort][~is_stream],
    c=colors[~is_stream],
    alpha=alphas[~is_stream],
    s=2,
    zorder=-10,
)
d1 = ax03.errorbar(
    data["phi1"][psort][is_stream],
    data["phi2"][psort][is_stream],
    xerr=data["phi1_err"][psort][is_stream],
    yerr=data["phi2_err"][psort][is_stream],
    **_stream_kw,
    zorder=-9,
)
ax03.scatter(
    data["phi1"][psort][is_stream],
    data["phi2"][psort][is_stream],
    c=colors[is_stream],
    alpha=alphas[is_stream],
    s=2,
    zorder=-8,
)

# Literature
l1 = ax03.plot(pal5I21.phi1.degree, pal5I21.phi2.degree, **_lit_kw, label="Ibata+21")
ax03.plot(pal5PW19.phi1.degree, pal5PW19.phi2.degree, **_lit_kw, label="PW+19")

# Model
f1 = ax03.fill_between(
    data["phi1"],
    (mpa["phi2", "mu"] - xp.exp(mpa["phi2", "ln-sigma"])),
    (mpa["phi2", "mu"] + xp.exp(mpa["phi2", "ln-sigma"])),
    color=cmap1(0.99),
    alpha=0.25,
    where=where,
)
# f2 = ax03.plot(
#     data["phi1"][where],
#     mpa["phi2", "mu"][where],
#     c="salmon",
#     ls="--",
#     lw=1,
#     label="Model (MLE)",
# )

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
ax03.legend(
    [legend_elements_data, l1, (p1, p2), f1],
    ["Data", r"Past", "Guides", "Model"],
    numpoints=1,
    ncol=(2, 2),
    loc="upper left",
    handler_map={list: HandlerTuple(ndivide=None)},
)

# ---------------------------------------------------------------------------
# Parallax - variance

# ---------------------------------------------------------------------------
# Parallax

ax05 = fig.add_subplot(
    gs0[3, :],
    xlabel="",
    ylabel=r"$\varpi$ [mas]",
    xlim=xlim,
    ylim=(max(data["plx"].min(), -1), data["plx"].max()),
    xticklabels=[],
    rasterization_zorder=0,
)

# Data
ax05.scatter(
    data["phi1"][psort][~is_stream],
    data["plx"][psort][~is_stream],
    c=colors[~is_stream],
    alpha=alphas[~is_stream],
    s=2,
    zorder=-10,
)
ax05.errorbar(
    data["phi1"][psort][is_stream],
    data["plx"][psort][is_stream],
    xerr=data["phi1_err"][psort][is_stream],
    yerr=data["plx_err"][psort][is_stream],
    **_stream_kw,
    zorder=-9,
)
ax05.scatter(
    data["phi1"][psort][is_stream],
    data["plx"][psort][is_stream],
    c=colors[is_stream],
    alpha=alphas[is_stream],
    s=2,
    zorder=-8,
)


# ---------------------------------------------------------------------------
# PM-Phi1 - variance

gs06 = gs0[4].subgridspec(2, 1, height_ratios=(1, 3), wspace=0.0, hspace=0.0)
ax06 = fig.add_subplot(
    gs06[0, :],
    xlim=xlim,
    xticklabels=[],
    ylabel=r"$\sigma_{\mu_{\phi_1}^*}$",
    aspect="auto",
)

# Model
ax06.fill_between(
    data["phi1"],
    np.exp(np.percentile(dmpars["stream.astrometric.pmphi1", "ln-sigma"], 5, axis=1)),
    np.exp(np.percentile(dmpars["stream.astrometric.pmphi1", "ln-sigma"], 95, axis=1)),
    color=cmap1(0.99),
    alpha=0.25,
    where=where,
)
ax06.scatter(
    data["phi1"][where],
    np.exp(mpa["pmphi1", "ln-sigma"][where]),
    s=1,
    c=cmap1(0.99),
)

for tick in ax06.get_yticklabels():
    tick.set_verticalalignment("bottom")


# ---------------------------------------------------------------------------
# PM-Phi1

ax07 = fig.add_subplot(
    gs06[1, :],
    xlabel="",
    xlim=xlim,
    ylabel=r"$\mu_{\phi_1}^*$ [mas yr$^{-1}$]",
    ylim=(data["pmphi1"].min(), data["pmphi1"].max()),
    rasterization_zorder=0,
    xticklabels=[],
    aspect="auto",
)

# Data
ax07.scatter(
    data["phi1"][psort][~is_stream],
    data["pmphi1"][psort][~is_stream],
    c=colors[~is_stream],
    alpha=alphas[~is_stream],
    s=2,
    zorder=-10,
)
ax07.errorbar(
    data["phi1"][psort][is_stream],
    data["pmphi1"][psort][is_stream],
    xerr=data["phi1_err"][psort][is_stream],
    yerr=data["pmphi1_err"][psort][is_stream],
    **_stream_kw,
    zorder=-9,
)
ax07.scatter(
    data["phi1"][psort][is_stream],
    data["pmphi1"][psort][is_stream],
    c=colors[is_stream],
    alpha=alphas[is_stream],
    s=2,
    zorder=-8,
)

# Literature
ax07.plot(
    pal5I21.phi1.degree,
    pal5I21.pm_phi1_cosphi2.value,
    **_lit_kw,
    label="Ibata+21",
)
ax07.plot(
    pal5PW19.phi1.degree,
    pal5PW19.pm_phi1_cosphi2.value,
    **_lit_kw,
    label="PW+19",
)

# Model
ax07.fill_between(
    data["phi1"][where],
    (mpa["pmphi1", "mu"] - xp.exp(mpa["pmphi1", "ln-sigma"]))[where],
    (mpa["pmphi1", "mu"] + xp.exp(mpa["pmphi1", "ln-sigma"]))[where],
    color=cmap1(0.99),
    alpha=0.25,
)

# ---------------------------------------------------------------------------
# PM-Phi2 - variance

gs08 = gs0[5].subgridspec(2, 1, height_ratios=(1, 3), wspace=0.0, hspace=0.0)
ax08 = fig.add_subplot(
    gs08[0, :],
    xlim=xlim,
    xticklabels=[],
    ylabel=r"$\sigma_{\mu_{\phi_2}}$",
    aspect="auto",
)

# Model
ax08.fill_between(
    data["phi1"],
    np.exp(np.percentile(dmpars["stream.astrometric.pmphi2", "ln-sigma"], 5, axis=1)),
    np.exp(np.percentile(dmpars["stream.astrometric.pmphi2", "ln-sigma"], 95, axis=1)),
    color=cmap1(0.99),
    alpha=0.25,
    where=where,
)
ax08.scatter(
    data["phi1"][where],
    np.exp(mpa["pmphi2", "ln-sigma"][where]),
    s=1,
    c=cmap1(0.99),
)

for tick in ax08.get_yticklabels():
    tick.set_verticalalignment("bottom")


# ---------------------------------------------------------------------------
# PM-Phi2

ax09 = fig.add_subplot(
    gs08[1, :],
    xlabel=r"$\phi_1$",
    xlim=xlim,
    ylabel=r"$\mu_{\phi_2}$ [mas yr$^{-1}$]",
    ylim=(data["pmphi2"].min(), data["pmphi2"].max()),
    rasterization_zorder=0,
    aspect="auto",
)

# Data
ax09.scatter(
    data["phi1"][psort][~is_stream],
    data["pmphi2"][psort][~is_stream],
    c=colors[~is_stream],
    alpha=alphas[~is_stream],
    s=2,
    zorder=-10,
)
ax09.errorbar(
    data["phi1"][psort][is_stream],
    data["pmphi2"][psort][is_stream],
    xerr=data["phi1_err"][psort][is_stream],
    yerr=data["pmphi2_err"][psort][is_stream],
    **_stream_kw,
    zorder=-9,
)
ax09.scatter(
    data["phi1"][psort][is_stream],
    data["pmphi2"][psort][is_stream],
    c=colors[is_stream],
    alpha=alphas[is_stream],
    s=2,
    zorder=-8,
)

# Literature
ax09.plot(pal5I21.phi1.degree, pal5I21.pm_phi2.value, **_lit_kw, label="Ibata+21")
ax09.plot(pal5PW19.phi1.degree, pal5PW19.pm_phi2.value, **_lit_kw, label="PW+19")

# Model
ax09.fill_between(
    data["phi1"][where],
    (mpa["pmphi2", "mu"] - xp.exp(mpa["pmphi2", "ln-sigma"]))[where],
    (mpa["pmphi2", "mu"] + xp.exp(mpa["pmphi2", "ln-sigma"]))[where],
    color=cmap1(0.99),
    alpha=0.25,
)

fig.savefig(paths.figures / "pal5" / "results_full.pdf")
