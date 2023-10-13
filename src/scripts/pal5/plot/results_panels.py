"""Plot Phi1-binned panels of the trained model."""

import copy as pycopy
import sys

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
import stream_ml.visualization as smlvis
from stream_ml.core import WEIGHT_NAME

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.helper import manually_set_dropout, p2alpha, recursive_iterate
from scripts.mpl_colormaps import stream_cmap1 as cmap1
from scripts.pal5.datasets import data
from scripts.pal5.define_model import model

# =============================================================================

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

# Load model
model = pycopy.deepcopy(model)
model.load_state_dict(xp.load(paths.data / "pal5" / "model" / "model_10800.pt"))
# model = model.eval()

# Load results from 4-likelihoods.py
lik_tbl = QTable.read(paths.data / "pal5" / "membership_likelhoods.ecsv")
stream_prob = np.array(lik_tbl["stream (50%)"])
bkg_prob = np.array(lik_tbl["bkg (50%)"])
stream_wgt = np.array(lik_tbl["stream.ln-weight"])

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
gs0 = gs[0].subgridspec(3, 1, height_ratios=(1, 2.5, 5))

colors = cmap1(stream_prob[psort])
alphas = p2alpha(stream_prob[psort])
xlim = (data["phi1"].min(), data["phi1"].max())

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
    ylabel=r"$\ln f_{\rm stream}$",
    xlim=xlim,
    ylim=(-6, 0),
    xticklabels=[],
    rasterization_zorder=0,
)

# Upper and lower bounds
_bounds_kw = {"c": "gray", "ls": "-", "lw": 2, "alpha": 0.8}
_bounds = model.params[(f"stream.{WEIGHT_NAME}",)].bounds
ax01.axhline(_bounds.lower[0], **_bounds_kw)
ax01.axhline(_bounds.upper[0], **_bounds_kw)

# 15% dropout
f1 = ax01.fill_between(
    data["phi1"],
    np.percentile(stream_wgt, 5, axis=1),
    np.percentile(stream_wgt, 95, axis=1),
    color=cmap1(0.99),
    alpha=0.25,
    where=strm_range,
)
# Mean
(l1,) = ax01.plot(
    data["phi1"][strm_range],
    np.percentile(stream_wgt, 50, axis=1)[strm_range],
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
    data["phi1"][psort][~is_strm],
    data["phi2"][psort][~is_strm],
    c=colors[~is_strm],
    alpha=alphas[~is_strm],
    s=2,
    zorder=-10,
)

ax02.scatter(
    data["phi1"][psort][is_strm],
    data["phi2"][psort][is_strm],
    c=colors[is_strm],
    alpha=alphas[is_strm],
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
    [legend_elements_data],
    ["Data"],
    numpoints=1,
    ncol=(2, 2),
    loc="upper left",
    handler_map={list: HandlerTuple(ndivide=None)},
)
ax02.add_artist(legend)

xlabel = ax02.xaxis.get_label()
xlabel.set_bbox({"facecolor": "white", "edgecolor": "white"})

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

ax100 = ax110 = ax120 = ax130 = ax140 = None

for i, b in enumerate(np.unique(which_bin)):
    in_bin = which_bin == b

    data_ = data[psort][in_bin[psort]]
    bkg_prob_ = bkg_prob[psort][in_bin[psort]]
    stream_prob_ = stream_prob[psort][in_bin[psort]]

    ws = np.stack((bkg_prob_, stream_prob_), axis=1)
    color = [cmap1(0.01), cmap1(0.99)]
    label = ["", "Stream Model (MLE)"]

    # ---------------------------------------------------------------------------
    # Phi2

    ax10i = fig.add_subplot(
        gs1[0, i],
        xlabel=r"$\phi_2$ [$\degree$]",
        sharex=ax100 if ax100 is not None else None,
    )
    ax10i.locator_params(axis="x", nbins=5)

    # Connect to top plot(s)
    for ax in (ax01, ax02):
        ax.axvline(bins[i], color="gray", ls="--", zorder=-200)
        ax.axvline(bins[i + 1], color="gray", ls="--", zorder=-200)
    smlvis._slices.connect_slices_to_top(  # noqa: SLF001
        fig, ax02, ax10i, left=bins[i], right=bins[i + 1], color="gray"
    )

    cphi2s = np.ones((in_bin.sum(), 2)) * data_["phi2"][:, None].numpy()
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
        ax10i.tick_params(labelleft=False)

    # ---------------------------------------------------------------------------
    # Parallax

    ax11i = fig.add_subplot(
        gs1[1, i],
        xlabel=r"$\varpi$ [mas]",
        xlim=(-2, 1.5),
        sharex=ax110 if ax110 is not None else None,
    )
    ax11i.locator_params(axis="x", nbins=5)

    cplxs = np.ones((in_bin.sum(), 2)) * data_["plx"][:, None].numpy()
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
        ax11i.tick_params(labelleft=False)

    # ---------------------------------------------------------------------------
    # PM-Phi1

    ax12i = fig.add_subplot(
        gs1[2, i],
        xlabel=r"$\mu_{\phi_1}^*$ [mas yr$^{-1}$]",
        sharex=ax120 if ax120 is not None else None,
    )
    ax12i.locator_params(axis="x", nbins=5)

    # Recovered
    cpmphi1s = np.ones((in_bin.sum(), 2)) * data_["pmphi1"][:, None].numpy()
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
        ax12i.tick_params(labelleft=False)

    # ---------------------------------------------------------------------------
    # PM-Phi2

    ax13i = fig.add_subplot(
        gs1[3, i],
        xlabel=r"$\mu_{\phi_2}$ [mas yr$^{-1}$]",
        sharex=ax130 if ax130 is not None else None,
    )
    ax13i.locator_params(axis="x", nbins=5)

    cpmphi2s = np.ones((in_bin.sum(), 2)) * data_["pmphi2"][:, None].numpy()
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
        ax13i.tick_params(labelleft=False)

    # ---------------------------------------------------------------------------
    # Photometry

    ax14i = fig.add_subplot(
        gs1[4, i],
        xlabel=("g - r [mag]"),
        xlim=(0, 1),
        ylim=(21, 13),
        rasterization_zorder=20,
        sharex=ax140 if ax140 is not None else None,
    )
    ax14i.set_xticks([0.1, 0.5, 0.9], ["0.1", "0.5", "0.9"])

    ax14i.scatter(data_["g"] - data_["r"], data_["g"], c=colors[in_bin[psort]], s=1)

    for label in ax14i.get_xticklabels():
        label.set_horizontalalignment("center")

    if i == 0:
        ax14i.set_ylabel("g [mag]")
    else:
        ax14i.tick_params(labelleft=False)

fig.savefig(paths.figures / "pal5" / "results_panels.pdf")
