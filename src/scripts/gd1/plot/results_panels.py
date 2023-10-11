"""Plot Phi1-binned panels of the trained model."""

import sys

import asdf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch as xp
from astropy.table import QTable
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerTuple
from scipy import stats
from showyourwork.paths import user as user_paths

import stream_ml.visualization as smlvis
from stream_ml.core import WEIGHT_NAME, Data
from stream_ml.pytorch.params import Params
from stream_ml.visualization.background import (
    exponential_like_distribution as exp_distr,
)

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.gd1.datasets import data
from scripts.gd1.define_model import model
from scripts.helper import (
    color_by_probable_member,
    manually_set_dropout,
    p2alpha,
    recursive_iterate,
)
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
model.load_state_dict(xp.load(paths.data / "gd1" / "model" / "model_2700.pt"))
model = model.eval()

# Load results from 4-likelihoods.py
lik_tbl = QTable.read(paths.data / "gd1" / "membership_likelhoods.ecsv")
bkg_prob = np.array(lik_tbl["bkg (50%)"])
stream_prob = np.array(lik_tbl["stream (50%)"])
stream_wgt = np.array(lik_tbl["stream.ln-weight"])
spur_prob = np.array(lik_tbl["spur (50%)"])
spur_wgt = np.array(lik_tbl["spur.ln-weight"])
allstream_prob = np.array(lik_tbl["allstream (50%)"])

# =============================================================================
# Likelihood

# Sorter for plotting
psort = np.argsort(allstream_prob)

# Foreground
_is_strm = (stream_prob > 0.6) & (stream_wgt.mean(1) > -4)
strm_range = (np.min(data["phi1"][_is_strm].numpy()) <= data["phi1"]) & (
    data["phi1"] <= np.max(data["phi1"][_is_strm].numpy())
)
_is_spur = (spur_prob > 0.6) & (spur_wgt.mean(1) > -5)
spur_range = (np.min(data["phi1"][_is_spur].numpy()) <= data["phi1"]) & (
    data["phi1"] <= np.max(data["phi1"][_is_spur].numpy())
)

# Also evaluate the model with dropout on
with xp.no_grad():
    # turn dropout on
    model = model.train()
    manually_set_dropout(model, 0.15)

    # evaluate the model
    ldmpars = [model.unpack_params(model(data)) for i in range(100)]

    # mpars
    dmpars = Params(recursive_iterate(ldmpars, ldmpars[0], reduction=lambda x: x))
    mpars = Params(recursive_iterate(ldmpars, ldmpars[0]))

    # turn dropout back off
    manually_set_dropout(model, 0)
    model = model.eval()

# =============================================================================
# Make Figure

fig = plt.figure(figsize=(11, 13))

gs = GridSpec(
    2,
    1,
    figure=fig,
    height_ratios=(5, 10),
    hspace=0.12,
    left=0.07,
    right=0.98,
    top=0.965,
    bottom=0.03,
)
gs0 = gs[0].subgridspec(3, 1, height_ratios=(2, 3, 5))

colors = color_by_probable_member(
    (stream_prob[psort], cmap1), (spur_prob[psort], cmap2)
)
alphas = p2alpha(allstream_prob[psort])
xlims = (data["phi1"].min().numpy(), data["phi1"].max().numpy())

# ---------------------------------------------------------------------------
# Colormap

gs00 = gs0[0, :].subgridspec(2, 1, height_ratios=(1, 1), hspace=0.15)

# Stream
ax00 = fig.add_subplot(gs00[0, :])
cbar = fig.colorbar(ScalarMappable(cmap=cmap1), cax=ax00, orientation="horizontal")
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")
cbar.ax.text(0.5, 0.5, "Stream Probability", ha="center", va="center", fontsize=14)

# Spur
ax01 = fig.add_subplot(gs00[1, :])
cbar = fig.colorbar(ScalarMappable(cmap=cmap2), cax=ax01, orientation="horizontal")
cbar.ax.xaxis.set_ticks([])
cbar.ax.xaxis.set_label_position("bottom")
cbar.ax.text(0.5, 0.5, "Spur Probability", ha="center", va="center", fontsize=14)


# ---------------------------------------------------------------------------
# Weight plot

ax02 = fig.add_subplot(
    gs0[1, :], xlim=xlims, xticklabels=[], ylabel=r"$f_{\rm stream}$", ylim=(0, 0.3)
)

f1 = ax02.fill_between(
    data["phi1"],
    np.exp(np.percentile(stream_wgt, 5, axis=1)),
    np.exp(np.percentile(stream_wgt, 95, axis=1)),
    color=cmap1(0.99),
    alpha=0.25,
)
(l1,) = ax02.plot(
    data["phi1"],
    np.exp(np.percentile(stream_wgt, 50, axis=1)),
    c=cmap1(0.99),
    ls="--",
    lw=2,
)
f2 = ax02.fill_between(
    data["phi1"],
    np.exp(np.percentile(spur_wgt, 5, axis=1)),
    np.exp(np.percentile(spur_wgt, 95, axis=1)),
    color=cmap2(0.99),
    alpha=0.25,
)
(l2,) = ax02.plot(
    data["phi1"],
    np.exp(np.percentile(spur_wgt, 50, axis=1)),
    c=cmap2(0.99),
    ls="--",
    lw=2,
)

ax02.legend(
    [[(f1, l1), (f2, l2)]],
    [r"Models"],
    numpoints=1,
    handler_map={list: HandlerTuple(ndivide=None)},
    loc="upper left",
)

# ---------------------------------------------------------------------------
# Phi2

ax03 = fig.add_subplot(
    gs0[2, :],
    xlabel=r"$\phi_1$ [deg]",
    xlim=xlims,
    ylabel=r"$\phi_2$ [deg]",
    xticklabels=[],
    rasterization_zorder=0,
)

ax03.scatter(
    data["phi1"][psort], data["phi2"][psort], c=colors, alpha=alphas, s=2, zorder=-10
)

# =============================================================================
# Slice plots

gs1 = gs[1].subgridspec(5, 4, height_ratios=(1, 1, 1, 1, 2), hspace=0.1)

# Legend
legend1 = plt.legend(
    handles=[
        mpl.patches.Patch(color=cmap1(0.01), label="Background"),
        mpl.patches.Patch(color=cmap1(0.99), label="Stream"),
        mpl.patches.Patch(color=cmap2(0.99), label="Spur"),
    ],
    ncols=4,
    loc="upper right",
    bbox_to_anchor=(1, -0.18),
)
ax03.add_artist(legend1)

# Bin the data for plotting
bins = np.linspace(data["phi1"].min(), data["phi1"].max(), num=5, endpoint=True)
which_bin = np.digitize(data["phi1"], bins[:-1])

ax100 = ax110 = ax120 = ax130 = ax140 = None

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
        xlabel=r"$\phi_2$ [deg]",
        sharey=ax100 if ax100 is not None else None,
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
        label=["", "Stream Model", "Spur Model"],
    )

    xmin, xmax = data["phi2"].min().numpy(), data["phi2"].max().numpy()
    x = np.linspace(xmin, xmax)
    bkg_wgt = np.exp(mpars[f"background.{WEIGHT_NAME}",][sel].mean())
    m = mpars["background.astrometric.phi2pmphi1.phi2", "slope"][sel].mean()
    ax10i.plot(x, bkg_wgt * exp_distr(m, xmin, xmax).pdf(x), c="k")

    if i == 0:
        ax10i.set_ylabel("frequency")
        ax100 = ax10i

        # Add legend
        ax10i.legend(
            handles=[
                mpl.lines.Line2D([0], [0], color="k", lw=4, ls="-", label="Background")
            ],
            loc="upper left",
        )

    else:
        ax10i.tick_params(labelleft=False)

    # ---------------------------------------------------------------------------
    # Parallax

    ax11i = fig.add_subplot(
        gs1[1, i],
        xlabel=r"$\varpi$ [mas]",
        sharey=ax110 if ax110 is not None else None,
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
        label=["", "Stream Model", "Spur Model"],
    )

    if i == 0:
        ax11i.set_ylabel("frequency")
        ax110 = ax11i
    else:
        ax11i.tick_params(labelleft=False)

    # ---------------------------------------------------------------------------
    # PM-Phi1

    ax12i = fig.add_subplot(
        gs1[2, i],
        xlabel=r"$\mu_{\phi_1}^*$ [mas yr$^{-1}$]",
        sharey=ax120 if ax120 is not None else None,
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
        label=["", "Stream Model", "Spur Model"],
    )

    xmin, xmax = data["pmphi1"].min().numpy(), data["pmphi1"].max().numpy()
    x = np.linspace(xmin, xmax)
    m = mpars["background.astrometric.phi2pmphi1.pmphi1", "slope"][sel].mean()
    ax12i.plot(x, bkg_wgt * exp_distr(m, xmin, xmax).pdf(x), c="k")

    if i == 0:
        ax12i.set_ylabel("frequency")
        ax120 = ax12i
    else:
        ax12i.tick_params(labelleft=False)

    # ---------------------------------------------------------------------------
    # PM-Phi2

    ax13i = fig.add_subplot(
        gs1[3, i],
        xlabel=r"$\mu_{phi_2}$ [mas yr$^{-1}$]",
        sharey=ax130 if ax130 is not None else None,
    )
    ax13i.hist(
        np.ones((sel.sum(), 3)) * data_["pmphi2"][:, None].numpy(),
        bins=50,
        weights=np.stack((bkg_prob_, stream_prob_, spur_prob_), axis=1),
        color=[cmap1(0.01), cmap1(0.99), cmap2(0.99)],
        alpha=0.75,
        density=True,
        stacked=True,
        label=["", "Stream Model", "Spur Model"],
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
        ax130 = ax13i
    else:
        ax13i.tick_params(labelleft=False)

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
        ax140 = ax14i
    else:
        ax14i.tick_params(labelleft=False)


fig.savefig(paths.figures / "gd1" / "results_panels.pdf")
