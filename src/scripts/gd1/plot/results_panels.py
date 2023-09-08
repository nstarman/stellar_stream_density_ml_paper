"""Plot GD1 Likelihoods."""

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch as xp
from matplotlib.gridspec import GridSpec
from showyourwork.paths import user as user_paths

import stream_ml.visualization as smlvis
from stream_ml.visualization.background import (
    exponential_like_distribution as exp_distr,
)

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.as_posix())
# isort: split

from scripts.gd1.datasets import data, where
from scripts.gd1.define_model import model
from scripts.helper import manually_set_dropout

# =============================================================================

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

# Load model
model.load_state_dict(xp.load(paths.data / "gd1" / "model.pt"))
model = model.eval()

# Evaluate model
with xp.no_grad():
    mpars = model.unpack_params(model(data))

    stream_lik = model.component_ln_posterior("stream", mpars, data, where=where)
    spur_lik = model.component_ln_posterior("spur", mpars, data, where=where)
    bkg_lik = model.component_ln_posterior("background", mpars, data, where=where)
    tot_lik = model.ln_posterior(mpars, data, where=where)


stream_weight = mpars[("stream.weight",)]
stream_cutoff = stream_weight > 2e-2

bkg_prob = xp.exp(bkg_lik - tot_lik)
stream_prob = xp.exp(stream_lik - tot_lik)
spur_prob = xp.exp(spur_lik - tot_lik)
allstream_prob = xp.exp(xp.logaddexp(stream_lik, spur_lik) - tot_lik)

psort = np.argsort(allstream_prob)
pmax = allstream_prob.max()
pmin = allstream_prob.min()


# =============================================================================
# Make Figure

fig = plt.figure(constrained_layout="tight", figsize=(11, 12))
gs = GridSpec(2, 1, figure=fig, height_ratios=(5, 10), hspace=0.07)
gs0 = gs[0].subgridspec(3, 1, height_ratios=(1, 3, 5))

cmap = plt.get_cmap()

# ---------------------------------------------------------------------------
# Colormap

ax00 = fig.add_subplot(gs0[0, :])
cbar = fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap),
    cax=ax00,
    orientation="horizontal",
    label="Stream Probability",
)
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")


# ---------------------------------------------------------------------------
# Weight plot

ax01 = fig.add_subplot(gs0[1, :])
ax01.set(ylabel="Stream fraction", ylim=(0, 0.3))
ax01.set_xticklabels([])

with xp.no_grad():
    manually_set_dropout(model, 0.15)
    weights = xp.stack(
        [model.unpack_params(model(data))["stream.weight",] for i in range(100)], 1
    )
    weight_percentiles = np.c_[
        np.percentile(weights, 5, axis=1), np.percentile(weights, 95, axis=1)
    ]
    manually_set_dropout(model, 0)
ax01.fill_between(
    data["phi1"],
    weight_percentiles[:, 0],
    weight_percentiles[:, 1],
    color="k",
    alpha=0.25,
    label=r"Model (15% dropout)",
)
ax01.plot(data["phi1"], stream_weight, c="k", ls="--", lw=2, label="Model (MLE)")
ax01.legend(loc="upper left")

# ---------------------------------------------------------------------------
# Phi2

mpa = mpars.get_prefixed("stream.astrometric")

ax02 = fig.add_subplot(gs0[2, :])
ax02.set_xticklabels([])
ax02.set(ylabel=r"$\phi_2$ [$\degree$]")

ax02.scatter(
    data["phi1"][psort],
    data["phi2"][psort],
    c=allstream_prob[psort],
    alpha=0.1 + (1 - 0.1) / (pmax - pmin) * (allstream_prob[psort] - pmin),
    s=2,
    zorder=-10,
)
ax02.set_rasterization_zorder(0)
ax02.fill_between(
    data["phi1"][stream_cutoff],
    (mpa["phi2", "mu"] - xp.exp(mpa["phi2", "ln-sigma"]))[stream_cutoff],
    (mpa["phi2", "mu"] + xp.exp(mpa["phi2", "ln-sigma"]))[stream_cutoff],
    color="k",
    alpha=0.25,
    label="Model (MLE)",
)
ax02.legend(loc="upper left")

# =============================================================================
# Slice plots

gs1 = gs[1].subgridspec(5, 4, height_ratios=(1, 1, 1, 1, 2), hspace=0)

# Legend
legend1 = plt.legend(
    handles=[
        mpl.patches.Patch(color=cmap(0.01), label="Background (MLE)"),
        mpl.lines.Line2D(
            [0], [0], color="k", lw=4, ls=":", label="Background Distribution"
        ),
        mpl.patches.Patch(color=cmap(0.99), label="Stream (MLE)"),
        mpl.patches.Patch(color="tab:olive", label="Spur (MLE)"),
    ],
    ncols=4,
    loc="upper right",
    bbox_to_anchor=(1, -0.05),
)
ax02.add_artist(legend1)

# Bin the data for plotting
bins = np.linspace(data["phi1"].min(), data["phi1"].max(), num=5, endpoint=True)
which_bin = np.digitize(data["phi1"], bins[:-1])

for i, b in enumerate(np.unique(which_bin)):
    sel = which_bin == b

    data_ = data[sel]
    bkg_prob_ = bkg_prob[sel]
    stream_prob_ = stream_prob[sel]
    spur_prob_ = spur_prob[sel]

    # ---------------------------------------------------------------------------
    # Phi2

    ax11i = fig.add_subplot(gs1[0, i])

    # Connect to top plot(s)
    for ax in (ax01, ax02):
        ax.axvline(bins[i], color="gray", ls="--", zorder=-200)
        ax.axvline(bins[i + 1], color="gray", ls="--", zorder=-200)
    smlvis._slices.connect_slices_to_top(  # noqa: SLF001
        fig, ax02, ax11i, left=bins[i], right=bins[i + 1], color="gray"
    )

    cphi2s = np.ones((sel.sum(), 3)) * data_["phi2"][:, None].numpy()
    ws = np.stack((bkg_prob_, stream_prob_, spur_prob_), axis=1)
    ax11i.hist(
        cphi2s,
        bins=50,
        weights=ws,
        color=[cmap(0.01), cmap(0.99), "tab:olive"],
        alpha=0.75,
        density=True,
        stacked=True,
        label=["", "Stream Model (MLE)", "Spur Model (MLE)"],
    )

    xmin, xmax = data["phi2"].min().numpy(), data["phi2"].max().numpy()
    x = np.linspace(xmin, xmax)
    bkg_wgt = mpars["background.weight",][sel].mean()
    m = mpars["background.astrometric.phi2", "slope"][sel].mean()
    ax11i.plot(x, bkg_wgt * exp_distr(m, xmin, xmax).pdf(x), c="k")

    ax11i.set_xlabel(r"$\phi_2$ [$\degree$]")
    if i == 0:
        ax11i.set_ylabel("frequency")

    # ---------------------------------------------------------------------------
    # Parallax

    ax11i = fig.add_subplot(gs1[1, i])

    # Connect to top plot(s)
    for ax in (ax01, ax02):
        ax.axvline(bins[i], color="gray", ls="--", zorder=-200)
        ax.axvline(bins[i + 1], color="gray", ls="--", zorder=-200)
    smlvis._slices.connect_slices_to_top(  # noqa: SLF001
        fig, ax02, ax11i, left=bins[i], right=bins[i + 1], color="gray"
    )

    cplxs = np.ones((sel.sum(), 3)) * data_["plx"][:, None].numpy()
    ws = np.stack((bkg_prob_, stream_prob_, spur_prob_), axis=1)
    ax11i.hist(
        cplxs,
        bins=50,
        weights=ws,
        color=[cmap(0.01), cmap(0.99), "tab:olive"],
        alpha=0.75,
        density=True,
        stacked=True,
        label=["", "Stream Model (MLE)", "Spur Model (MLE)"],
    )

    ax11i.set_xlabel(r"$\phi_2$ [$\degree$]")
    if i == 0:
        ax11i.set_ylabel("frequency")

    # ---------------------------------------------------------------------------
    # PM-Phi1

    ax12i = fig.add_subplot(gs1[2, i])

    # Recovered
    cpmphi1s = np.ones((sel.sum(), 3)) * data_["pmphi1"][:, None].numpy()
    ws = np.stack((bkg_prob_, stream_prob_, spur_prob_), axis=1)
    ax12i.hist(
        cpmphi1s,
        bins=50,
        weights=ws,
        color=[cmap(0.01), cmap(0.99), "tab:olive"],
        alpha=0.75,
        density=True,
        stacked=True,
        label=["", "Stream Model (MLE)", "Spur Model (MLE)"],
    )

    xmin, xmax = data["pmphi1"].min().numpy(), data["pmphi1"].max().numpy()
    x = np.linspace(xmin, xmax)
    m = mpars["background.astrometric.pmphi1", "slope"][sel].mean()
    ax12i.plot(x, bkg_wgt * exp_distr(m, xmin, xmax).pdf(x), c="k")

    ax12i.set_xlabel(r"$\mu_{\phi_1}^*$ [mas yr$^{-1}$]")
    if i == 0:
        ax12i.set_ylabel("frequency")

    # ---------------------------------------------------------------------------
    # PM-Phi2

    ax13i = fig.add_subplot(gs1[3, i])
    ax13i.hist(
        np.ones((sel.sum(), 3)) * data_["pmphi2"][:, None].numpy(),
        bins=50,
        weights=np.stack((bkg_prob_, stream_prob_, spur_prob_), axis=1),
        color=[cmap(0.01), cmap(0.99), "tab:olive"],
        alpha=0.75,
        density=True,
        stacked=True,
        label=["", "Stream Model (MLE)", "Spur Model (MLE)"],
    )

    xmin, xmax = data["pmphi2"].min().numpy(), data["pmphi2"].max().numpy()
    x = np.linspace(xmin, xmax)
    m = mpars["background.astrometric.pmphi2", "slope"][sel].mean()
    ax13i.plot(x, bkg_wgt * exp_distr(m, xmin, xmax).pdf(x), c="k")

    ax13i.set_xlabel(r"$\mu_{phi_2}$ [mas yr$^{-1}$]")
    if i == 0:
        ax13i.set_ylabel("frequency")

    # ---------------------------------------------------------------------------
    # Photometry

    ax14i = fig.add_subplot(gs1[4, i])

    sorter = np.argsort(stream_prob_)
    ax14i.scatter(
        data_["g"][sorter] - data_["r"][sorter],
        data_["g"][sorter],
        c=stream_prob_[sorter],
        s=1,
        rasterized=True,
    )
    isspur = spur_prob_ > 0.75
    ax14i.scatter(
        (data_["g"] - data_["r"])[isspur],
        data_["g"][isspur],
        c="tab:olive",
        s=1,
        rasterized=True,
    )
    ax14i.set(xlabel=("g - r [mag]"), xlim=(0, 1), ylim=(22, 13))
    ax14i.set_xticklabels([])

    if i == 0:
        ax14i.set_ylabel("g [mag]")
    else:
        ax14i.set_yticklabels([])


fig.savefig(paths.figures / "gd1" / "results_panels.pdf")
