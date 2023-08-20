"""Train photometry background flow."""

import sys
from pathlib import Path

import asdf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch as xp
from matplotlib.gridspec import GridSpec

import stream_ml.pytorch as sml
import stream_ml.visualization as smlvis

# Add the parent directory to the path
sys.path.append(Path(__file__).parents[3].as_posix())
# isort: split

from scripts import paths
from scripts.mock.define_model import model
from scripts.mock.model import helper

# =============================================================================

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

# Load model
model.load_state_dict(xp.load(paths.data / "mock" / "model.pt"))

# Load data
with asdf.open(paths.data / "mock" / "data.asdf") as af:
    data = sml.Data(**af["data"]).astype(xp.Tensor, dtype=xp.float32)
    where = sml.Data(**af["data"]).astype(xp.Tensor, dtype=xp.bool)
    stream_table = af["stream_table"]
    table = af["table"]
    n_stream = af["n_stream"]
    n_background = af["n_background"]

# Evaluate model
with xp.no_grad():
    helper.manually_set_dropout(model, 0)
    mpars = model.unpack_params(model(data))

    stream_lik = model.component_posterior("stream", mpars, data, where=where)
    bkg_lik = model.component_posterior("background", mpars, data, where=where)

weight = mpars[("stream.weight",)]
where = weight > 2e-2

stream_prob = stream_lik / (stream_lik + bkg_lik)
stream_prob[(stream_prob > 0.4) & ~where] = 0
psort = np.argsort(stream_prob)
pmax = stream_prob.max()
pmin = stream_prob.min()

bkg_prob = bkg_lik / (stream_lik + bkg_lik)


# =============================================================================
# Make Figure

fig = plt.figure(constrained_layout=True, figsize=(14, 13))
gs = GridSpec(2, 1, height_ratios=(1, 1), figure=fig, hspace=0.0)
gs0 = gs[0].subgridspec(4, 1, height_ratios=(1, 5, 5, 5), hspace=0)

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
ax01.set(ylabel="Stream fraction", ylim=(0, 0.5))
ax01.set_xticklabels([])

# Truth
phi1 = stream_table["phi1"].to_value("deg")

Hs, bin_edges = np.histogram(phi1, bins=75)
Ht, bin_edges = np.histogram(data["phi1"], bins=bin_edges)
ax01.bar(bin_edges[:-1], Hs / Ht, width=bin_edges[1] - bin_edges[0], color=cmap(0.99))

# Model
# TODO: in a diagnostic plot
# with xp.no_grad():
#     helper.manually_set_dropout(model, 0.15)
#     weights = xp.stack(
#         [model.unpack_params(model(data))["stream.weight",] for i in range(100)], 1
#     )
#     helper.manually_set_dropout(model, 0)
#     weight_percentiles = np.c_[
#         np.percentile(weights, 5, axis=1), np.percentile(weights, 95, axis=1)
#     ]
# ax01.fill_between(
#     data["phi1"],
#     weight_percentiles[:, 0],
#     weight_percentiles[:, 1],
#     color="k",
#     alpha=0.25,
#     label=r"Model (15% dropout)",
# )
ax01.plot(data["phi1"], weight, c="k", ls="--", lw=2, label="Model (MLE)")

ax01.legend(loc="upper left")

# ---------------------------------------------------------------------------
# Phi2

mpa = mpars.get_prefixed("stream.astrometric")

ax02 = fig.add_subplot(gs0[2, :])
ax02.set_xticklabels([])
ax02.set(ylabel=r"$\phi_2$ [$\degree$]")

ax02.scatter(
    phi1,
    stream_table["phi2"].to_value("deg"),
    s=10,
    c="k",
    alpha=0.5,
    zorder=-100,
    label="Ground Truth",
)
line = ax02.scatter(
    data["phi1"][psort],
    data["phi2"][psort],
    c=stream_prob[psort],
    alpha=0.1 + (1 - 0.1) / (pmax - pmin) * (stream_prob[psort] - pmin),
    s=2,
    zorder=-10,
    cmap="seismic",
)
ax02.set_rasterization_zorder(0)

# with xp.no_grad():
#     helper.manually_set_dropout(model, 0.15)
#     n = ("stream.astrometric.phi2", "mu")
#     mus = xp.stack([model.unpack_params(model(data))[n] for i in range(100)], 1)
#     mu_percentiles = np.c_[
#         np.percentile(mus, 5, axis=1), np.percentile(mus, 95, axis=1)
#     ]
#     helper.manually_set_dropout(model, 0)
# ax02.fill_between(
#     data["phi1"], mu_percentiles[:, 0], mu_percentiles[:, 1], color="k", alpha=0.25
# )
ax02.plot(
    data["phi1"][where],
    mpa["phi2", "mu"][where],
    c="k",
    ls="--",
    lw=2,
    label="Model (MLE)",
)

# legend
hdata = (
    mpl.lines.Line2D([], [], c=cmap(0.01), marker="o"),
    mpl.lines.Line2D([], [], c=cmap(0.99), marker="o"),
)
handles, labels = ax02.get_legend_handles_labels()
ax02.legend(
    [hdata] + handles,
    ["data"] + labels,
    handler_map={tuple: mpl.legend_handler.HandlerTuple(ndivide=None)},
    loc="center left",
)

# ---------------------------------------------------------------------------
# Distance

ax03 = fig.add_subplot(gs0[3, :])
ax03.set(xlabel=r"$\phi_1$ [deg]", ylabel=r"$\varpi$ [mas yr$^-1$]")

k_dist = "parallax"
ax03.scatter(
    phi1,
    stream_table["parallax"].to_value("mas"),
    s=10,
    c="k",
    alpha=0.5,
    zorder=-100,
    label="Ground Truth",
)
ax03.scatter(
    data["phi1"][psort],
    data["parallax"][psort],
    c=stream_prob[psort],
    alpha=0.1 + (1 - 0.1) / (pmax - pmin) * (stream_prob[psort] - pmin),
    s=2,
    zorder=-10,
    cmap="seismic",
)
ax03.set_rasterization_zorder(0)

# with xp.no_grad():
#     helper.manually_set_dropout(model, 0.15)
#     n = ("stream.astrometric.parallax", "mu")
#     mus = xp.stack([model.unpack_params(model(data))[n] for i in range(100)], 1)
#     mu_percentiles = np.c_[
#         np.percentile(mus, 5, axis=1), np.percentile(mus, 95, axis=1)
#     ]
#     helper.manually_set_dropout(model, 0)
# ax03.fill_between(
#     data["phi1"], mu_percentiles[:, 0], mu_percentiles[:, 1], color="k", alpha=0.25
# )

ax03.plot(
    data["phi1"][where],
    mpa[k_dist, "mu"][where],
    c="k",
    ls="--",
    lw=2,
    label="Model (MLE)",
)

# =============================================================================
# Slice plots

gs1 = gs[1].subgridspec(4, 4)

# Legend
legend1 = plt.legend(
    handles=[
        mpl.patches.Patch(facecolor="white", edgecolor="black", label="Ground Truth"),
        mpl.patches.Patch(color=cmap(0.01), label="Background (MLE)"),
        mpl.patches.Patch(color=cmap(0.99), label="Stream (MLE)"),
    ],
    ncols=3,
    loc="upper right",
    bbox_to_anchor=(1, -0.05),
)
ax03.add_artist(legend1)


# Bin the data for plotting
bins = np.linspace(data["phi1"].min(), data["phi1"].max(), num=5, endpoint=True)
which_bin = np.digitize(data["phi1"], bins[:-1])

for i, b in enumerate(np.unique(which_bin)):
    sel = which_bin == b

    # ---------------------------------------------------------------------------
    # Phi2

    ax10i = fig.add_subplot(gs1[0, i])

    # Connect to top plot(s)
    for ax in (ax01, ax02):
        ax.axvline(bins[i], color="gray", ls="--", zorder=-200)
        ax.axvline(bins[i + 1], color="gray", ls="--", zorder=-200)
    smlvis._slices.connect_slices_to_top(  # noqa: SL
        fig, ax03, ax10i, left=bins[i], right=bins[i + 1], color="gray"
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

    ax10i.set_xlabel(r"$\phi_2$ [$\degree$]")
    if i == 0:
        ax10i.set_ylabel("frequency")
        # ax10i.legend(loc="upper left")

    # ---------------------------------------------------------------------------
    # Distance

    ax11i = fig.add_subplot(gs1[1, i])

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

    ax11i.set_xlabel(r"$\varpi$ [mas]")
    if i == 0:
        ax11i.set_ylabel("frequency")
        # ax11i.legend(loc="upper left")

    # ---------------------------------------------------------------------------
    # Photometry

    # ------------------------------------------
    # Stream

    ax12i = fig.add_subplot(gs1[2, i])

    prob = stream_prob[sel]
    sorter = np.argsort(prob)
    ax12i.scatter(
        data["g"][sel][sorter],
        data["r"][sel][sorter],
        c=prob[sorter],
        cmap="seismic",
        s=1,
        rasterized=True,
    )
    ax12i.set_xticklabels([])

    if i == 0:
        ax12i.set_ylabel("r [mag]")
    else:
        ax12i.set_yticklabels([])

    # ------------------------------------------
    # Background

    ax13i = fig.add_subplot(gs1[3, i])
    prob = bkg_prob[sel]
    sorter = np.argsort(prob)
    ax13i.scatter(
        data["g"][sel][sorter],
        data["r"][sel][sorter],
        c=1 - prob[sorter],
        cmap="seismic",
        s=1,
        rasterized=True,
    )
    ax13i.set(xlabel="g [mag]")

    if i == 0:
        ax13i.set_ylabel("r [mag]")
    else:
        ax13i.set_yticklabels([])

fig.savefig(paths.figures / "mock" / "results.pdf")
