"""Plot Phi1-binned panels of the trained model."""

import sys

import asdf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch as xp
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerTuple
from scipy import stats
from showyourwork.paths import user as user_paths

import stream_ml.visualization as smlvis
from stream_ml.core import Data
from stream_ml.visualization.background import (
    exponential_like_distribution as exp_distr,
)

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.gd1.datasets import data, where
from scripts.gd1.define_model import model
from scripts.helper import color_by_probable_member, manually_set_dropout, p2alpha
from scripts.mpl_colormaps import stream_cmap1 as cmap1
from scripts.mpl_colormaps import stream_cmap2 as cmap2

# =============================================================================

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

# isochrone data
with asdf.open(
    paths.data / "gd1" / "isochrone.asdf", "r", lazy_load=False, copy_arrays=True
) as af:
    isochrone_data = Data(**af["isochrone_data"])

# Load model
model.load_state_dict(xp.load(paths.data / "gd1" / "model.pt"))
model = model.eval()

# Evaluate model
with xp.no_grad():
    mpars = model.unpack_params(model(data))

    stream_lnlik = model.component_ln_posterior("stream", mpars, data, where=where)
    spur_lnlik = model.component_ln_posterior("spur", mpars, data, where=where)
    bkg_lnlik = model.component_ln_posterior("background", mpars, data, where=where)
    # tot_lnlik = model.ln_posterior(mpars, data, where=where)  # FIXME
    tot_lnlik = xp.logsumexp(xp.stack((stream_lnlik, spur_lnlik, bkg_lnlik), 1), 1)


stream_weight = mpars[("stream.weight",)]
stream_cutoff = stream_weight > 2e-2

spur_weight = mpars[("spur.weight",)]
spur_cutoff = spur_weight > 1e-2

bkg_prob = xp.exp(bkg_lnlik - tot_lnlik)
stream_prob = xp.exp(stream_lnlik - tot_lnlik)
spur_prob = xp.exp(spur_lnlik - tot_lnlik)
allstream_prob = xp.exp(xp.logaddexp(stream_lnlik, spur_lnlik) - tot_lnlik)

psort = np.argsort(allstream_prob)


# =============================================================================
# Make Figure

fig = plt.figure(constrained_layout="tight", figsize=(11, 13))
gs = GridSpec(2, 1, figure=fig, height_ratios=(5, 10), hspace=0.12)
gs0 = gs[0].subgridspec(4, 1, height_ratios=(1, 1, 3, 5))

colors = color_by_probable_member(
    (stream_prob[psort], cmap1), (spur_prob[psort], cmap2)
)
alphas = p2alpha(allstream_prob[psort])

# ---------------------------------------------------------------------------
# Colormap

# Stream probability
ax00 = fig.add_subplot(gs0[0, :])
cbar = fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap1),
    cax=ax00,
    orientation="horizontal",
    label="Stream Probability",
)
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")

# Spur probability
ax01 = fig.add_subplot(gs0[1, :])
cbar = fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap2),
    cax=ax01,
    orientation="horizontal",
    label="Spur Probability",
)
cbar.ax.xaxis.set_ticks([])
cbar.ax.xaxis.set_label_position("bottom")


# ---------------------------------------------------------------------------
# Weight plot

ax02 = fig.add_subplot(gs0[2, :])
ax02.set(ylabel="Stream fraction", ylim=(0, 0.3))
ax02.set_xticklabels([])

with xp.no_grad():
    manually_set_dropout(model, 0.15)
    _stream_weights = []
    _spur_weights = []
    for _ in range(25):
        _mpars = model.unpack_params(model(data))
        _stream_weights.append(_mpars["stream.weight",])
        _spur_weights.append(_mpars["spur.weight",])

    stream_weights = xp.stack(_stream_weights, 1)
    stream_weight_percentiles = np.c_[
        np.percentile(stream_weights, 5, axis=1),
        np.percentile(stream_weights, 95, axis=1),
    ]

    spur_weights = xp.stack(_spur_weights, 1)
    spur_weight_percentiles = np.c_[
        np.percentile(spur_weights, 5, axis=1),
        np.percentile(spur_weights, 95, axis=1),
    ]

    manually_set_dropout(model, 0)

f1 = ax02.fill_between(
    data["phi1"],
    stream_weight_percentiles[:, 0],
    stream_weight_percentiles[:, 1],
    color=cmap1(0.99),
    alpha=0.25,
)
(l1,) = ax02.plot(
    data["phi1"], stream_weights.mean(1), c="k", ls="--", lw=2, label="Model (MLE)"
)
f2 = ax02.fill_between(
    data["phi1"],
    spur_weight_percentiles[:, 0],
    spur_weight_percentiles[:, 1],
    color=cmap2(0.99),
    alpha=0.25,
)
(l2,) = ax02.plot(data["phi1"], spur_weights.mean(1), c="k", ls="--", lw=2)

ax02.legend(
    [(f1, f2), l1],
    [r"Models (15% dropout)", l1.get_label()],
    numpoints=1,
    handler_map={tuple: HandlerTuple(ndivide=None)},
    loc="upper left",
)

# ---------------------------------------------------------------------------
# Phi2

mpa = mpars.get_prefixed("stream.astrometric")

ax03 = fig.add_subplot(
    gs0[3, :],
    xlabel=r"$\phi_1$ [$\degree$]",
    ylabel=r"$\phi_2$ [$\degree$]",
    xticklabels=[],
    rasterization_zorder=0,
)

ax03.scatter(
    data["phi1"][psort], data["phi2"][psort], c=colors, alpha=alphas, s=2, zorder=-10
)
ax03.fill_between(
    data["phi1"][stream_cutoff],
    (mpa["phi2", "mu"] - xp.exp(mpa["phi2", "ln-sigma"]))[stream_cutoff],
    (mpa["phi2", "mu"] + xp.exp(mpa["phi2", "ln-sigma"]))[stream_cutoff],
    color="k",
    alpha=0.25,
    label="Model (MLE)",
)
ax03.legend(loc="upper left")

# =============================================================================
# Slice plots

gs1 = gs[1].subgridspec(5, 4, height_ratios=(1, 1, 1, 1, 2), hspace=0)

# Legend
legend1 = plt.legend(
    handles=[
        mpl.patches.Patch(color=cmap1(0.01), label="Background (MLE)"),
        mpl.lines.Line2D(
            [0], [0], color="k", lw=4, ls="-", label="Background Distribution"
        ),
        mpl.patches.Patch(color=cmap1(0.99), label="Stream (MLE)"),
        mpl.patches.Patch(color=cmap2(0.99), label="Spur (MLE)"),
    ],
    ncols=4,
    loc="upper right",
    bbox_to_anchor=(1, -0.23),
)
ax03.add_artist(legend1)

# Bin the data for plotting
bins = np.linspace(data["phi1"].min(), data["phi1"].max(), num=5, endpoint=True)
which_bin = np.digitize(data["phi1"], bins[:-1])

for i, b in enumerate(np.unique(which_bin)):
    sel = which_bin == b

    data_ = data[psort][sel[psort]]
    bkg_prob_ = bkg_prob[psort][sel[psort]]
    stream_prob_ = stream_prob[psort][sel[psort]]
    spur_prob_ = spur_prob[psort][sel[psort]]

    # ---------------------------------------------------------------------------
    # Phi2

    ax10i = fig.add_subplot(
        gs1[0, i],
        xlabel=r"$\phi_2$ [$\degree$]",  # sharex=ax100 if i > 0 else None
    )

    # Connect to top plot(s)
    for ax in (ax01, ax02, ax03):
        ax.axvline(bins[i], color="gray", ls="--", zorder=-200)
        ax.axvline(bins[i + 1], color="gray", ls="--", zorder=-200)
    smlvis._slices.connect_slices_to_top(  # noqa: SLF001
        fig, ax03, ax10i, left=bins[i], right=bins[i + 1], color="gray"
    )

    cphi2s = np.ones((sel.sum(), 3)) * data_["phi2"][:, None].numpy()
    ws = np.stack((bkg_prob_, stream_prob_, spur_prob_), axis=1)
    ax10i.hist(
        cphi2s,
        bins=50,
        weights=ws,
        color=[cmap1(0.01), cmap1(0.99), cmap2(0.99)],
        alpha=0.75,
        density=True,
        stacked=True,
        label=["", "Stream Model (MLE)", "Spur Model (MLE)"],
    )

    xmin, xmax = data["phi2"].min().numpy(), data["phi2"].max().numpy()
    x = np.linspace(xmin, xmax)
    bkg_wgt = mpars["background.weight",][sel].mean()
    m = mpars["background.astrometric.phi2pmphi1.phi2", "slope"][sel].mean()
    ax10i.plot(x, bkg_wgt * exp_distr(m, xmin, xmax).pdf(x), c="k")

    if i == 0:
        ax10i.set_ylabel("frequency")
        ax100 = ax10i
    else:
        ax10i.set_yticklabels([])

    # ---------------------------------------------------------------------------
    # Parallax

    ax11i = fig.add_subplot(
        gs1[1, i],
        xlabel=r"$\varpi$ [mas]",  # sharex=ax110 if i > 0 else None
    )

    cplxs = np.ones((sel.sum(), 3)) * data_["plx"][:, None].numpy()
    ws = np.stack((bkg_prob_, stream_prob_, spur_prob_), axis=1)
    ax11i.hist(
        cplxs,
        bins=50,
        weights=ws,
        color=[cmap1(0.01), cmap1(0.99), cmap2(0.99)],
        alpha=0.75,
        density=True,
        stacked=True,
        label=["", "Stream Model (MLE)", "Spur Model (MLE)"],
    )

    if i == 0:
        ax11i.set_ylabel("frequency")
        ax110 = ax11i
    else:
        ax11i.set_yticklabels([])

    # ---------------------------------------------------------------------------
    # PM-Phi1

    ax12i = fig.add_subplot(
        gs1[2, i],
        xlabel=r"$\mu_{\phi_1}^*$ [mas yr$^{-1}$]",
        # sharex=ax120 if i > 0 else None,
    )

    # Recovered
    cpmphi1s = np.ones((sel.sum(), 3)) * data_["pmphi1"][:, None].numpy()
    ws = np.stack((bkg_prob_, stream_prob_, spur_prob_), axis=1)
    ax12i.hist(
        cpmphi1s,
        bins=50,
        weights=ws,
        color=[cmap1(0.01), cmap1(0.99), cmap2(0.99)],
        alpha=0.75,
        density=True,
        stacked=True,
        label=["", "Stream Model (MLE)", "Spur Model (MLE)"],
    )

    xmin, xmax = data["pmphi1"].min().numpy(), data["pmphi1"].max().numpy()
    x = np.linspace(xmin, xmax)
    m = mpars["background.astrometric.phi2pmphi1.pmphi1", "slope"][sel].mean()
    ax12i.plot(x, bkg_wgt * exp_distr(m, xmin, xmax).pdf(x), c="k")

    if i == 0:
        ax12i.set_ylabel("frequency")
        ax120 = ax12i
    else:
        ax12i.set_yticklabels([])

    # ---------------------------------------------------------------------------
    # PM-Phi2

    ax13i = fig.add_subplot(gs1[3, i], xlabel=r"$\mu_{phi_2}$ [mas yr$^{-1}$]")
    ax13i.hist(
        np.ones((sel.sum(), 3)) * data_["pmphi2"][:, None].numpy(),
        bins=50,
        weights=np.stack((bkg_prob_, stream_prob_, spur_prob_), axis=1),
        color=[cmap1(0.01), cmap1(0.99), cmap2(0.99)],
        alpha=0.75,
        density=True,
        stacked=True,
        label=["", "Stream Model (MLE)", "Spur Model (MLE)"],
    )

    xmin, xmax = data["pmphi2"].min().numpy(), data["pmphi2"].max().numpy()
    x = np.linspace(xmin, xmax)
    mu = mpars["background.astrometric.pmphi2.pmphi2", "mu"][sel].mean()
    lnsigma = mpars["background.astrometric.pmphi2.pmphi2", "ln-sigma"][sel].mean()
    bounds = model["background"]["astrometric"]["pmphi2"].coord_bounds["pmphi2"]
    d = stats.norm(loc=mu, scale=xp.exp(lnsigma))
    ax13i.plot(x, bkg_wgt * d.pdf(x) / (d.cdf(bounds[1]) - d.cdf(bounds[0])), c="k")

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
        xticklabels=[],
        rasterization_zorder=20,
    )

    ax14i.scatter(
        data_["g"] - data_["r"],
        data_["g"],
        c=colors[sel[psort]],
        s=1,
    )
    ax14i.plot(
        isochrone_data["g"] - isochrone_data["r"],
        isochrone_data["g"]
        + mpars["stream.photometric.distmod", "mu"][sel].mean().numpy(),
        c="green",
        label="Isochrone",
    )

    if i == 0:
        ax14i.set_ylabel("g [mag]")
        ax14i.legend(loc="upper left")
    else:
        ax14i.set_yticklabels([])


fig.savefig(paths.figures / "gd1" / "results_panels.pdf")
