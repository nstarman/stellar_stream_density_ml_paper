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

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.helper import manually_set_dropout, p2alpha, recursive_iterate
from scripts.mpl_colormaps import stream_cmap1 as cmap1
from scripts.pal5.datasets import data
from scripts.pal5.model import model

# =============================================================================

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

# Load model
model = pycopy.deepcopy(model)
model.load_state_dict(xp.load(paths.data / "pal5" / "models" / "model_12000.pt"))
model = model.eval()

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
is_strm_ = stream_prob > 0.75
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

fig = plt.figure(figsize=(11, 4.3))
gs = GridSpec(
    2,
    1,
    figure=fig,
    height_ratios=(1, 1.9),
    hspace=0.37,
    left=0.07,
    right=0.98,
    top=0.965,
    bottom=0.1,
)

colors = cmap1(stream_prob[psort])
alphas = p2alpha(stream_prob[psort])
xlim = (data["phi1"].min(), data["phi1"].max())

# ---------------------------------------------------------------------------
# Phi2

mpa = mpars.get_prefixed("stream.astrometric")

ax0 = fig.add_subplot(
    gs[0, :],
    xlabel=r"$\phi_1$ [deg]",
    xlim=xlim,
    ylabel=r"$\phi_2$ [deg]",
    ylim=(data["phi2"].min(), data["phi2"].max()),
    rasterization_zorder=0,
    aspect="auto",
)

ax0.scatter(
    data["phi1"][psort][~is_strm],
    data["phi2"][psort][~is_strm],
    c=colors[~is_strm],
    alpha=alphas[~is_strm],
    s=2,
    zorder=-10,
)

ax0.scatter(
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
ax0.add_artist(legend)

xlabel = ax0.xaxis.get_label()
xlabel.set_bbox({"facecolor": "white", "edgecolor": "white"})
ax0.locator_params(axis="x", nbins=10)

# =============================================================================
# Slice plots

gs1 = gs[1].subgridspec(1, 5, hspace=0.5, wspace=0.1)
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
ax0.add_artist(legend1)

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
    is_strm_ = is_strm[in_bin[psort]]
    colors_ = colors[in_bin[psort]]

    ws = np.stack((bkg_prob_, stream_prob_), axis=1)
    color = [cmap1(0.01), cmap1(0.99)]
    label = ["", "Stream Model (MLE)"]

    # ---------------------------------------------------------------------------
    # Photometry

    ax1i = fig.add_subplot(
        gs1[i],
        xlabel=("g - r [mag]"),
        xlim=(0, 1),
        ylim=(21, 13),
        rasterization_zorder=20,
        sharex=ax140 if ax140 is not None else None,
    )
    ax1i.set_xticks([0.1, 0.5, 0.9], ["0.1", "0.5", "0.9"])

    ax1i.scatter((data_["g"] - data_["r"]), data_["g"], c=colors_, s=1, zorder=-10)

    if len(is_strm_) > 0:
        ax1i.errorbar(
            (data_["g"] - data_["r"])[is_strm_],
            data_["g"][is_strm_],
            ls="none",
            c=cmap1(0.99),
            ms=1,
            zorder=-9,
        )

    for label in ax1i.get_xticklabels():
        label.set_horizontalalignment("center")

    # Connect to top plot(s)
    for ax in (ax0,):
        ax.axvline(bins[i], color="gray", ls="--", zorder=-200)
        ax.axvline(bins[i + 1], color="gray", ls="--", zorder=-200)
    smlvis._slices.connect_slices_to_top(  # noqa: SLF001
        fig, ax0, ax1i, left=bins[i], right=bins[i + 1], color="gray"
    )

    if i == 0:
        ax1i.set_ylabel("g [mag]")
    else:
        ax1i.tick_params(labelleft=False)

fig.savefig(paths.figures / "pal5" / "results_panels.pdf")
