"""Train photometry background flow."""

import sys

import asdf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch as xp
from matplotlib.gridspec import GridSpec
from showyourwork.paths import user as user_paths

import stream_ml.pytorch as sml
import stream_ml.visualization as smlvis
from stream_ml.core import WEIGHT_NAME

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.helper import manually_set_dropout, p2alpha, recursive_iterate
from scripts.mock.model import model

# =============================================================================

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

# Load model
model.load_state_dict(xp.load(paths.data / "mock" / "model.pt"))

# Load data
with asdf.open(paths.data / "mock" / "data.asdf") as af:
    data = sml.Data(**af["data"]).astype(xp.Tensor, dtype=xp.float32)
    where = sml.Data(**af["where"]).astype(xp.Tensor, dtype=xp.bool)
    stream_table = af["stream_table"]
    table = af["table"]
    n_stream = af["n_stream"]
    n_background = af["n_background"]

# =============================================================================
# Likelihood

# Also evaluate the model with dropout on
with xp.no_grad():
    # turn dropout on
    model = model.train()
    manually_set_dropout(model, 0.15)
    # evaluate the model
    ldmpars = [model.unpack_params(model(data)) for i in range(100)]
    # Mpars
    dmpars = sml.params.Params(
        recursive_iterate(ldmpars, ldmpars[0], reduction=lambda x: x)
    )

    # turn dropout back off
    manually_set_dropout(model, 0)
    model = model.eval()

# Evaluate model
with xp.no_grad():
    manually_set_dropout(model, 0)
    mpars = model.unpack_params(model(data))

    stream_lnlik = model.component_ln_posterior("stream", mpars, data, where=where)
    bkg_lnlik = model.component_ln_posterior("background", mpars, data, where=where)
    tot_lnlik = xp.logaddexp(stream_lnlik, bkg_lnlik)  # FIXME

# Weight
weight = mpars[(f"stream.{WEIGHT_NAME}",)]
where = weight > -5

# Probabilities
stream_prob = xp.exp(stream_lnlik - tot_lnlik)
bkg_prob = xp.exp(bkg_lnlik - tot_lnlik)
sel = (stream_prob > 0.4) & ~where  # a little post
stream_prob[sel] = 0
bkg_prob[sel] = 1

# Sorter for plotting
psort = np.argsort(stream_prob)

##############################################################################
# Make Figure

fig = plt.figure(figsize=(11, 12.5))
gs = GridSpec(
    2,
    1,
    figure=fig,
    height_ratios=(6, 5),
    hspace=0.15,
    left=0.07,
    right=0.98,
    top=0.965,
    bottom=0.03,
)

alpha = p2alpha(stream_prob, minval=0.1)
cmap = plt.get_cmap("Stream1")
xlim = (data["phi1"].min(), data["phi1"].max())

# =============================================================================
# Full plots

gs0 = gs[0].subgridspec(4, 1, height_ratios=(1, 3, 6.5, 6.5), hspace=0.1)


# ---------------------------------------------------------------------------
# Colormap

ax00 = fig.add_subplot(gs0[0, :])
cbar = fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap), cax=ax00, orientation="horizontal"
)
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")
cbar.ax.text(0.5, 0.5, "Stream Probability", ha="center", va="center", fontsize=14)


# ---------------------------------------------------------------------------
# Weight plot

ax01 = fig.add_subplot(
    gs0[1, :],
    xlim=xlim,
    ylabel="Stream fraction",
    ylim=(0, 0.5),
    xticklabels=[],
    aspect="auto",
    rasterization_zorder=0,
)

# Truth
phi1 = stream_table["phi1"].to_value("deg")

Hs, bin_edges = np.histogram(phi1, bins=55)
Ht, _ = np.histogram(data["phi1"], bins=bin_edges)
ax01.bar(
    bin_edges[:-1],
    Hs / Ht,
    width=bin_edges[1] - bin_edges[0],
    edgecolor="k",
    color="none",
    label="Ground Truth",
    zorder=-10,
)

# Model
ax01.plot(
    data["phi1"], np.exp(weight), c=cmap(0.99), ls="--", lw=2, label="Model", zorder=-5
)

for tick in ax01.get_yticklabels():
    tick.set_verticalalignment("bottom")
ax01.yaxis.set_label_coords(-0.045, 0.5)
ax01.legend(loc="upper left")

# ---------------------------------------------------------------------------
# Phi2 - variance

mpa = mpars.get_prefixed("stream.astrometric")

gs02 = gs0[2].subgridspec(2, 1, height_ratios=(1, 3), wspace=0.0, hspace=0.0)
ax02 = fig.add_subplot(
    gs02[0, :],
    xlim=xlim,
    xticklabels=[],
    ylabel=r"$\sigma_{\phi_2}$",
    aspect="auto",
    rasterization_zorder=0,
)

# Truth
bin_edges = np.histogram_bin_edges(phi1, bins=100)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
inds = np.digitize(stream_table["phi1"].value, bin_edges)
true_mu = np.array(mpa["phi2", "mu"][table["label"] == "stream"])
true_std = np.array(
    [
        np.std((stream_table["phi2"].value - true_mu)[inds == i])
        for i in range(1, len(bin_edges))
    ]
)
ax02.plot(bin_centers, true_std, c="k", label="Ground Truth", zorder=-10)

# Model
ax02.fill_between(
    data["phi1"],
    np.exp(np.percentile(dmpars["stream.astrometric.phi2", "ln-sigma"], 5, axis=1)),
    np.exp(np.percentile(dmpars["stream.astrometric.phi2", "ln-sigma"], 95, axis=1)),
    color=cmap(0.99),
    alpha=0.25,
    where=where,
    zorder=-5,
)
ax02.scatter(
    data["phi1"][where],
    np.exp(mpa["phi2", "ln-sigma"][where]),
    s=1,
    c=cmap(0.99),
    zorder=-4,
)

for tick in ax02.get_yticklabels():
    tick.set_verticalalignment("bottom")

# ---------------------------------------------------------------------------
# Phi2

ax03 = fig.add_subplot(
    gs02[1, :],
    xlim=xlim,
    ylabel=r"$\phi_2$ [$\degree$]",
    rasterization_zorder=0,
    xticklabels=[],
    aspect="auto",
)

ax03.scatter(
    phi1,
    stream_table["phi2"].to_value("deg"),
    s=10,
    c="k",
    alpha=0.5,
    label="Ground Truth",
    zorder=-11,
)
line = ax03.scatter(
    data["phi1"][psort],
    data["phi2"][psort],
    c=stream_prob[psort],
    alpha=alpha[psort],
    s=2,
    zorder=-10,
    cmap="seismic",
)

# Model
ax03.plot(
    data["phi1"][where],
    mpa["phi2", "mu"][where],
    c="salmon",
    ls="--",
    lw=1,
    label="Model",
    zorder=-9,
)

# legend
hdata = (
    mpl.lines.Line2D([], [], c=cmap(0.01), marker="o"),
    mpl.lines.Line2D([], [], c=cmap(0.99), marker="o"),
)
handles, labels = ax03.get_legend_handles_labels()
ax03.legend(
    [hdata] + handles,
    ["data"] + labels,
    handler_map={tuple: mpl.legend_handler.HandlerTuple(ndivide=None)},
    loc="upper left",
)

# ---------------------------------------------------------------------------
# Distance - variance

gs04 = gs0[3].subgridspec(2, 1, height_ratios=(1, 3), wspace=0.0, hspace=0.0)
ax04 = fig.add_subplot(
    gs04[0, :],
    xlim=xlim,
    xticklabels=[],
    ylabel=r"$\sigma_{\varpi}$",
    aspect="auto",
    rasterization_zorder=0,
)

# Truth
true_mu = np.array(mpa["parallax", "mu"][table["label"] == "stream"])
true_std = np.array(
    [
        np.std((stream_table["parallax"].value - true_mu)[inds == i])
        for i in range(1, len(bin_edges))
    ]
)
ax04.plot(bin_centers, true_std, c="k", label="Ground Truth", zorder=-10)

# Model
ax04.fill_between(
    data["phi1"],
    np.exp(np.percentile(dmpars["stream.astrometric.parallax", "ln-sigma"], 5, axis=1)),
    np.exp(
        np.percentile(dmpars["stream.astrometric.parallax", "ln-sigma"], 95, axis=1)
    ),
    color=cmap(0.99),
    alpha=0.25,
    where=where,
    zorder=-5,
)
ax04.scatter(
    data["phi1"][where],
    np.exp(mpa["parallax", "ln-sigma"][where]),
    s=1,
    c=cmap(0.99),
    zorder=-4,
)

for tick in ax04.get_yticklabels():
    tick.set_verticalalignment("bottom")

# ---------------------------------------------------------------------------
# Distance

ax05 = fig.add_subplot(
    gs04[1, :],
    xlim=xlim,
    xlabel=r"$\phi_1$ [deg]",
    ylabel=r"$\varpi$ [mas yr$^-1$]",
    rasterization_zorder=0,
    aspect="auto",
)

k_dist = "parallax"
ax05.scatter(
    phi1,
    stream_table["parallax"].to_value("mas"),
    s=10,
    c="k",
    alpha=0.5,
    label="Ground Truth",
    zorder=-11,
)
ax05.scatter(
    data["phi1"][psort],
    data["parallax"][psort],
    c=stream_prob[psort],
    alpha=alpha[psort],
    s=2,
    zorder=-10,
    cmap="seismic",
)

ax05.plot(
    data["phi1"][where],
    mpa[k_dist, "mu"][where],
    c="salmon",
    ls="--",
    lw=1,
    label="Model",
    zorder=-9,
)

xlabel = ax05.xaxis.get_label()
xlabel.set_bbox({"facecolor": "white", "edgecolor": "white"})

# =============================================================================
# Slice plots

gs1 = gs[1].subgridspec(3, 4, hspace=0.34)

# Legend
legend1 = plt.legend(
    handles=[
        mpl.patches.Patch(facecolor="white", edgecolor="black", label="Truth"),
        mpl.patches.Patch(color=cmap(0.01), label="Background"),
        mpl.patches.Patch(color=cmap(0.99), label="Stream"),
    ],
    ncols=3,
    loc="upper right",
    bbox_to_anchor=(1, -0.25),
)
ax05.add_artist(legend1)


# Bin the data for plotting
bins = np.linspace(data["phi1"].min(), data["phi1"].max(), num=5, endpoint=True)
which_bin = np.digitize(data["phi1"], bins[:-1])

ax100 = ax110 = ax120 = None

for i, b in enumerate(np.unique(which_bin)):
    sel = which_bin == b

    # ---------------------------------------------------------------------------
    # Phi2

    ax10i = fig.add_subplot(
        gs1[0, i],
        xlabel=r"$\phi_2$ [deg]",
        rasterization_zorder=0,
        sharey=ax100 if ax100 is not None else None,
    )

    # Connect to top plot(s)
    for ax in (ax01, ax02, ax03, ax04, ax05):
        ax.axvline(bins[i], color="gray", ls="--", zorder=-200)
        ax.axvline(bins[i + 1], color="gray", ls="--", zorder=-200)
    smlvis._slices.connect_slices_to_top(  # noqa: SL
        fig, ax05, ax10i, left=bins[i], right=bins[i + 1], color="gray"
    )

    # Recovered
    cphi2s = np.ones((sel.sum(), 2)) * table["phi2"][sel][:, None]
    ws = np.stack((bkg_prob[sel], stream_prob[sel]), axis=1)
    ax10i.hist(
        cphi2s,
        bins=50,
        weights=ws,
        color=[cmap(0.01), cmap(0.99)],
        alpha=0.75,
        density=True,
        stacked=True,
        label=["", "Model (MLE)"],
    )
    # Truth
    ws = np.stack(
        (table["label"][sel] == "background", table["label"][sel] == "stream"),
        axis=1,
        dtype=int,
    )
    ax10i.hist(
        cphi2s,
        bins=50,
        weights=ws,
        color=["k", "k"],
        histtype="step",
        density=True,
        stacked=True,
        label=["", "Ground Truth"],
    )

    if i == 0:
        ax10i.set_ylabel("frequency")
        ax100 = ax10i
    else:
        ax10i.tick_params(labelleft=False)

    # ---------------------------------------------------------------------------
    # Distance

    ax11i = fig.add_subplot(
        gs1[1, i],
        xlabel=r"$\varpi$ [mas]",
        rasterization_zorder=0,
        sharey=ax110 if ax110 is not None else None,
    )

    # Recovered
    cplxs = np.ones((sel.sum(), 2)) * table["parallax"][sel].value[:, None]
    ws = np.stack((bkg_prob[sel], stream_prob[sel]), axis=1)
    ax11i.hist(
        cplxs,
        bins=50,
        weights=ws,
        color=[cmap(0.01), cmap(0.99)],
        alpha=0.75,
        density=True,
        stacked=True,
        label=["", "Model (MLE)"],
    )
    # Truth
    ws = np.stack(
        (table["label"][sel] == "background", table["label"][sel] == "stream"),
        axis=1,
        dtype=int,
    )
    ax11i.hist(
        cplxs,
        bins=50,
        weights=ws,
        color=["k", "k"],
        histtype="step",
        density=True,
        stacked=True,
        label=["", "Ground Truth"],
    )

    if i == 0:
        ax11i.set_ylabel("frequency")
        ax110 = ax11i
    else:
        ax11i.tick_params(labelleft=False)

    # ------------------------------------------
    # Stream

    ax12i = fig.add_subplot(
        gs1[2, i],
        xlabel="g [mag]",
        xticklabels=[],
        rasterization_zorder=0,
        sharey=ax120 if ax120 is not None else None,
    )

    prob = stream_prob[sel]
    sorter = np.argsort(prob)
    ax12i.scatter(
        data["g"][sel][sorter],
        data["r"][sel][sorter],
        c=prob[sorter],
        cmap="seismic",
        s=1,
        zorder=-11,
    )

    prob = bkg_prob[sel]
    sorter = np.argsort(prob)
    ax12i.scatter(
        data["g"][sel][sorter],
        data["r"][sel][sorter],
        c=1 - prob[sorter],
        cmap="seismic",
        alpha=0.25,
        s=1,
        zorder=-10,
    )

    if i == 0:
        ax12i.set_ylabel("r [mag]")
        ax120 = ax12i
    else:
        ax12i.tick_params(labelleft=False)

fig.savefig(paths.figures / "mock" / "results.pdf")
