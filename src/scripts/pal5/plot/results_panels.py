"""Plot Phi1-binned panels of the trained model."""

import copy as pycopy
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch as xp
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from showyourwork.paths import user as user_paths

import stream_ml.pytorch as sml
import stream_ml.visualization as smlvis
from stream_ml.core import WEIGHT_NAME

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.helper import (
    detect_significant_changes_in_width,
    manually_set_dropout,
    p2alpha,
    recursive_iterate,
)
from scripts.mpl_colormaps import stream_cmap1 as cmap1
from scripts.pal5.datasets import data, where
from scripts.pal5.define_model import model

# =============================================================================

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

# Load model
model = pycopy.deepcopy(model)
model.load_state_dict(xp.load(paths.data / "pal5" / "model" / "model_10600.pt"))
model = model.eval()

# Evaluate model
with xp.no_grad():
    model = model.train()
    manually_set_dropout(model, 0.15)

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

    stream_lnlik = model.component_ln_posterior("stream", mpars, data, where=where)
    bkg_lnlik = model.component_ln_posterior("background", mpars, data, where=where)
    # tot_lnlik = model.ln_posterior(mpars, data, where=where)  # FIXME
    tot_lnlik = xp.logaddexp(stream_lnlik, bkg_lnlik)

    # turn dropout back off
    manually_set_dropout(model, 0)
    model = model.eval()


# Post-processing cleanup
clean = mpars[(f"stream.{WEIGHT_NAME}",)] > -10  # everything has weight > 0

indices = detect_significant_changes_in_width(
    data["phi1"],
    mpars,
    coords=[
        ("stream.astrometric.phi2", "ln-sigma"),
        ("stream.astrometric.pmphi1", "ln-sigma"),
        ("stream.astrometric.pmphi2", "ln-sigma"),
    ],
    threshold=3_000,
)
clean &= data["phi1"] > data["phi1"][indices[0]]

# Probabilities
bkg_prob = xp.exp(bkg_lnlik - tot_lnlik)
stream_prob = xp.exp(stream_lnlik - tot_lnlik)
#
stream_prob[~clean] = 0
bkg_prob[~clean] = 1

# Sorter for plotting
psort = np.argsort(stream_prob)

# Foreground
is_stream = stream_prob[psort] > 0.6


# =============================================================================
# Make Figure

fig = plt.figure(figsize=(11, 14))
gs = GridSpec(
    2,
    1,
    figure=fig,
    height_ratios=(5, 10),
    hspace=0.13,
    left=0.07,
    right=0.98,
    top=0.965,
    bottom=0.03,
)
gs0 = gs[0].subgridspec(3, 1, height_ratios=(1, 3, 5))

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
# Phi2

mpa = mpars.get_prefixed("stream.astrometric")

ax02 = fig.add_subplot(
    gs0[2, :],
    xlabel=r"$\phi_1$ [deg]",
    xlim=xlim,
    ylabel=r"$\phi_2$ [deg]",
    ylim=(data["phi2"].min(), data["phi2"].max()),
    rasterization_zorder=0,
    aspect="auto",
)

ax02.scatter(
    data["phi1"][psort][~is_stream],
    data["phi2"][psort][~is_stream],
    c=colors[~is_stream],
    alpha=alphas[~is_stream],
    s=2,
    zorder=-10,
)

f1 = ax02.fill_between(
    data["phi1"][clean],
    (mpa["phi2", "mu"] - xp.exp(mpa["phi2", "ln-sigma"]))[clean],
    (mpa["phi2", "mu"] + xp.exp(mpa["phi2", "ln-sigma"]))[clean],
    color="k",
    alpha=0.25,
    label="Model (MLE)",
    zorder=-9,
)

ax02.scatter(
    data["phi1"][psort][is_stream],
    data["phi2"][psort][is_stream],
    c=colors[is_stream],
    alpha=alphas[is_stream],
    s=2,
    zorder=-8,
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
    Line2D(
        [0],
        [0],
        marker="o",
        markeredgecolor="none",
        linestyle="none",
        markerfacecolor=cmap1(0.99),
        markersize=7,
    ),
]
legend = plt.legend(
    [legend_elements_data, f1],
    ["Data", "Model"],
    numpoints=1,
    ncol=(2, 2),
    loc="upper left",
    handler_map={list: HandlerTuple(ndivide=None)},
)
ax02.add_artist(legend)

# =============================================================================
# Slice plots

gs1 = gs[1].subgridspec(5, 5, height_ratios=(1, 1, 1, 1, 1.75), hspace=0.5, wspace=0.1)
# Legend
legend1 = plt.legend(
    handles=[
        mpl.patches.Patch(color=cmap1(0.01), label="Background"),
        mpl.patches.Patch(color=cmap1(0.99), label="Stream"),
    ],
    ncols=4,
    loc="upper right",
    bbox_to_anchor=(1, -0.17),
)
ax02.add_artist(legend1)

# Bin the data for plotting
bins = np.concatenate(
    (
        np.linspace(data["phi1"].min(), -0.5, num=3, endpoint=True),
        np.linspace(0.5, data["phi1"].max(), num=3, endpoint=True),
    )
)

which_bin = np.digitize(data["phi1"], bins[:-1])

ax110 = ax120 = ax130 = ax140 = ax150 = None

for i, b in enumerate(np.unique(which_bin)):
    sel = which_bin == b

    data_ = data[psort][sel[psort]]
    bkg_prob_ = bkg_prob[psort][sel[psort]]
    stream_prob_ = stream_prob[psort][sel[psort]]

    ws = np.stack((bkg_prob_, stream_prob_), axis=1)
    color = [cmap1(0.01), cmap1(0.99)]
    label = ["", "Stream Model (MLE)"]

    # ---------------------------------------------------------------------------
    # Phi2

    ax10i = fig.add_subplot(gs1[0, i], xlabel=r"$\phi_2$ [$\degree$]")
    ax10i.locator_params(axis="x", nbins=5)

    # Connect to top plot(s)
    for ax in (ax01, ax02):
        ax.axvline(bins[i], color="gray", ls="--", zorder=-200)
        ax.axvline(bins[i + 1], color="gray", ls="--", zorder=-200)
    smlvis._slices.connect_slices_to_top(  # noqa: SLF001
        fig, ax02, ax10i, left=bins[i], right=bins[i + 1], color="gray"
    )

    cphi2s = np.ones((sel.sum(), 2)) * data_["phi2"][:, None].numpy()
    ax10i.hist(
        cphi2s,
        bins=50,
        weights=ws,
        color=color,
        alpha=0.75,
        density=True,
        stacked=True,
        label=label,
    )

    if i == 0:
        ax10i.set_ylabel("frequency")
        ax100 = ax10i
    else:
        ax10i.set_yticklabels([])

    # ---------------------------------------------------------------------------
    # Parallax

    ax11i = fig.add_subplot(gs1[1, i], xlabel=r"$\varpi$ [mas]", xlim=(-2, 1.5))
    ax11i.locator_params(axis="x", nbins=5)

    cplxs = np.ones((sel.sum(), 2)) * data_["plx"][:, None].numpy()
    ax11i.hist(
        cplxs,
        bins=50,
        weights=ws,
        color=color,
        alpha=0.75,
        density=True,
        stacked=True,
        label=label,
    )

    if i == 0:
        ax11i.set_ylabel("frequency", labelpad=10)
        ax110 = ax11i
    else:
        ax11i.set_yticklabels([])

    # ---------------------------------------------------------------------------
    # PM-Phi1

    ax12i = fig.add_subplot(gs1[2, i], xlabel=r"$\mu_{\phi_1}^*$ [mas yr$^{-1}$]")
    ax12i.locator_params(axis="x", nbins=5)

    # Recovered
    cpmphi1s = np.ones((sel.sum(), 2)) * data_["pmphi1"][:, None].numpy()
    ax12i.hist(
        cpmphi1s,
        bins=50,
        weights=ws,
        color=color,
        alpha=0.75,
        density=True,
        stacked=True,
        label=label,
    )

    if i == 0:
        ax12i.set_ylabel("frequency")
        ax120 = ax12i
    else:
        ax12i.set_yticklabels([])

    # ---------------------------------------------------------------------------
    # PM-Phi2

    ax13i = fig.add_subplot(gs1[3, i], xlabel=r"$\mu_{\phi_2}$ [mas yr$^{-1}$]")
    ax13i.locator_params(axis="x", nbins=5)

    cpmphi2s = np.ones((sel.sum(), 2)) * data_["pmphi2"][:, None].numpy()
    ax13i.hist(
        cpmphi2s,
        bins=50,
        weights=ws,
        color=color,
        alpha=0.75,
        density=True,
        stacked=True,
        label=label,
    )

    if i == 0:
        ax13i.set_ylabel("frequency")
    else:
        ax13i.set_yticklabels([])

    # ---------------------------------------------------------------------------
    # Photometry

    ax14i = fig.add_subplot(
        gs1[4, i],
        xlabel=("g - r [mag]"),
        xlim=(0, 1),
        ylim=(21, 13),
        rasterization_zorder=20,
    )
    ax14i.set_xticks([0.1, 0.5, 0.9], ["0.1", "0.5", "0.9"])

    ax14i.scatter(data_["g"] - data_["r"], data_["g"], c=colors[sel[psort]], s=1)

    for label in ax14i.get_xticklabels():
        label.set_horizontalalignment("center")

    if i == 0:
        ax14i.set_ylabel("g [mag]")
    else:
        ax14i.set_yticklabels([])

fig.savefig(paths.figures / "pal5" / "results_panels.pdf")
