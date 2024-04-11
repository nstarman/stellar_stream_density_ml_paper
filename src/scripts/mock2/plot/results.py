"""Train photometry background flow."""

import sys

import asdf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch as xp
from matplotlib.gridspec import GridSpec
from scipy.interpolate import InterpolatedUnivariateSpline
from showyourwork.paths import user as user_paths
from tqdm import tqdm

import stream_mapper.pytorch as sml
import stream_mapper.visualization as smlvis
from stream_mapper.core import WEIGHT_NAME

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.helper import manually_set_dropout, p2alpha, recursive_iterate
from scripts.mock2.model import model

# =============================================================================

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")
plt.rcParams["ytick.direction"] = "inout"

# Load model
# model.load_state_dict(xp.load(paths.data / "mock2" / "model.pt"))
model.load_state_dict(xp.load(paths.data / "mock2" / "models" / "epoch_06200.pt"))

# Load data
with asdf.open(
    paths.data / "mock2" / "data.asdf", lazy_load=False, copy_arrays=True
) as af:
    data = sml.Data(**af["data"]).astype(xp.Tensor, dtype=xp.float32)
    where = sml.Data(**af["where"]).astype(xp.Tensor, dtype=xp.bool)
    stream_table = af["stream_table"]
    bkg_table = af["bkg_table"]
    table = af["table"]
    n_stream = af["n_stream"]
    n_background = af["n_background"]
    stream_isochrone = af["stream_abs_mags"].value
    true_stream = af["true_stream"]

# =============================================================================
# Likelihood

# Also evaluate the model with dropout on
with xp.no_grad():
    # turn dropout on
    model = model.train()
    manually_set_dropout(model, 0.15)
    # evaluate the model
    ldmpars = [model.unpack_params(model(data)) for i in tqdm(range(100))]
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
    tot_lnlik = xp.logaddexp(stream_lnlik, bkg_lnlik)

# Weight
weight = mpars[(f"stream.{WEIGHT_NAME}",)]
where = weight > -6

# Probabilities
stream_prob = xp.exp(stream_lnlik - tot_lnlik)
bkg_prob = xp.exp(bkg_lnlik - tot_lnlik)
sel = (stream_prob > 0.4) & ~where  # a little post
stream_prob[sel] = 0
bkg_prob[sel] = 1

# Sorter for plotting
psort = np.argsort(stream_prob)
isstrm = (stream_prob > 0.75)[psort]

##############################################################################
# Make Figure

fig = plt.figure(figsize=(11, 14))
gs = GridSpec(
    2,
    1,
    figure=fig,
    height_ratios=(6, 5.5),
    hspace=0.10,
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

gs0 = gs[0].subgridspec(5, 1, height_ratios=(0.75, 2, 6.5, 6.5, 6.5), hspace=0.07)


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
    ylabel="weight",
    ylim=(0, 0.03),
    xticklabels=[],
    aspect="auto",
    rasterization_zorder=0,
)

# Truth
phi1 = stream_table["phi1"].to_value("deg")
Hs, bin_edges = np.histogram(phi1, bins=50)
Ht, _ = np.histogram(data["phi1"], bins=bin_edges)
ax01.bar(
    bin_edges[:-1],
    Hs / Ht,
    width=bin_edges[1] - bin_edges[0],
    edgecolor="k",
    color="none",
    label="ground truth",
    zorder=-10,
)

# Model
ax01.plot(
    data["phi1"], np.exp(weight), c=cmap(0.99), ls="--", lw=2, label="Model", zorder=-5
)

ax01.yaxis.set_label_coords(-0.055, 0.5)
for tick in ax01.get_yticklabels():
    tick.set_verticalalignment("center")
ax01.legend(loc="upper left", ncols=2)

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
bin_edges = np.histogram_bin_edges(true_stream["phi1"].value, bins=40)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
inds = np.digitize(true_stream["phi1"].value, bin_edges)
bin_vals = np.array(
    [np.mean(true_stream["phi2"].value[inds == i]) for i in range(1, len(bin_edges))]
)
mu = InterpolatedUnivariateSpline(bin_centers, bin_vals)(true_stream["phi1"].value)
std = np.array(
    [
        np.std((true_stream["phi2"].value - mu)[inds == i])
        for i in range(1, len(bin_edges))
    ]
)
ax02.plot(bin_centers, std, c="k", label="truth", zorder=-10)

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
    color=cmap(0.99),
    zorder=-4,
)

ax02.yaxis.set_label_coords(-0.052, 0.5)
for tick in ax02.get_yticklabels():
    tick.set_verticalalignment("center")

# ---------------------------------------------------------------------------
# Phi2

ax03 = fig.add_subplot(
    gs02[1, :],
    xlim=xlim,
    ylabel=r"$\phi_2$ [deg]",
    rasterization_zorder=0,
    xticklabels=[],
    aspect="auto",
)

ax03.scatter(  # background
    data["phi1"][psort][~isstrm],
    data["phi2"][psort][~isstrm],
    c=stream_prob[psort][~isstrm],
    alpha=alpha[psort][~isstrm],
    s=2,
    zorder=-11,
    cmap="seismic",
)
ax03.scatter(
    phi1,
    stream_table["phi2"].to_value("deg"),
    s=10,
    c="k",
    alpha=0.5,
    label="ground truth",
    zorder=-10,
)
ax03.errorbar(  # stream_errors
    data["phi1"][psort][isstrm],
    data["phi2"][psort][isstrm],
    yerr=data["phi2_err"][psort][isstrm],
    color="gray",
    ls="none",
    zorder=-10,
)
ax03.scatter(  # stream
    data["phi1"][psort][isstrm],
    data["phi2"][psort][isstrm],
    c=stream_prob[psort][isstrm],
    alpha=alpha[psort][isstrm],
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
handles[-1] = mpl.lines.Line2D(
    [], [], c="salmon", marker="none", linestyle="--", drawstyle="steps-mid"
)
ax03.legend(
    [hdata] + handles,
    ["data"] + labels,
    handler_map={tuple: mpl.legend_handler.HandlerTuple(ndivide=None)},
    loc="upper left",
)

ax03.yaxis.set_label_coords(-0.05, 0.5)
for tick in ax03.get_yticklabels():
    tick.set_verticalalignment("center")

# ---------------------------------------------------------------------------
# Distance - variance

gs04 = gs0[3].subgridspec(2, 1, height_ratios=(1, 3), wspace=0.0, hspace=0.0)
ax04 = fig.add_subplot(
    gs04[0, :],
    xlim=xlim,
    xticklabels=[],
    ylabel=r"$\sigma_{\varpi}$",
    aspect="auto",
    yscale="log",
    rasterization_zorder=0,
)

# Truth
bin_vals = np.array(
    [
        np.mean(true_stream["parallax"].value[inds == i])
        for i in range(1, len(bin_edges))
    ]
)
mu = InterpolatedUnivariateSpline(bin_centers, bin_vals)(true_stream["phi1"].value)
std = np.array(
    [
        np.std((true_stream["parallax"].value - mu)[inds == i])
        for i in range(1, len(bin_edges))
    ]
)
ax04.plot(bin_centers, std, c="k", label="truth", zorder=-10)

std = np.array(
    [
        np.std((stream_table["parallax"].value - mu)[inds == i])
        for i in range(1, len(bin_edges))
    ]
)
ax04.plot(bin_centers, std, c="gray", label="observed", zorder=-10)

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
    color=cmap(0.99),
    zorder=-4,
)

# legend
handles, labels = ax04.get_legend_handles_labels()
ax04.legend(
    handles,
    labels,
    handler_map={tuple: mpl.legend_handler.HandlerTuple(ndivide=None)},
    handlelength=0.5,
    columnspacing=0.5,
    handletextpad=0.5,
    loc="upper left",
    ncol=2,
)

ax04.yaxis.set_label_coords(-0.052, 0.5)
for tick in ax04.get_yticklabels():
    tick.set_verticalalignment("center")

# ---------------------------------------------------------------------------
# Distance

ax05 = fig.add_subplot(
    gs04[1, :],
    xlim=xlim,
    xlabel=r"$\phi_1$ [deg]",
    xticklabels=[],
    ylabel=r"$\varpi$ [mas yr$^-1$]",
    rasterization_zorder=0,
    aspect="auto",
)

k_dist = "parallax"
ax05.scatter(  # background
    data["phi1"][psort][~isstrm],
    data["parallax"][psort][~isstrm],
    c=stream_prob[psort][~isstrm],
    alpha=alpha[psort][~isstrm],
    s=2,
    zorder=-11,
    cmap="seismic",
)
ax05.scatter(
    phi1,
    stream_table["parallax"].to_value("mas"),
    s=10,
    c="k",
    alpha=0.5,
    label="ground truth",
    zorder=-10,
)
ax05.errorbar(  # stream_errors
    data["phi1"][psort][isstrm],
    data["parallax"][psort][isstrm],
    yerr=data["parallax_err"][psort][isstrm],
    color="gray",
    ls="none",
    zorder=-10,
)
ax05.scatter(  # stream
    data["phi1"][psort][isstrm],
    data["parallax"][psort][isstrm],
    c=stream_prob[psort][isstrm],
    alpha=alpha[psort][isstrm],
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
    label="model",
    zorder=-9,
)

ax05.set_ylim(bkg_table["parallax"].value.min(), bkg_table["parallax"].value.max())
ax05.yaxis.set_label_coords(-0.048, 0.5)
for tick in ax05.get_yticklabels():
    tick.set_verticalalignment("center")

# ---------------------------------------------------------------------------
# PM Phi1 - variance

gs06 = gs0[4].subgridspec(2, 1, height_ratios=(1, 3), wspace=0.0, hspace=0.0)
ax06 = fig.add_subplot(
    gs06[0, :],
    xlim=xlim,
    xticklabels=[],
    ylabel=r"$\sigma_{\mu_1}$",
    yscale="log",
    aspect="auto",
    rasterization_zorder=0,
)

# Truth
bin_vals = np.array(
    [np.mean(true_stream["phi2"].value[inds == i]) for i in range(1, len(bin_edges))]
)
mu = InterpolatedUnivariateSpline(bin_centers, bin_vals)(true_stream["phi1"].value)
std = np.array(
    [
        np.std((true_stream["pmphi1"].value - mu)[inds == i])
        for i in range(1, len(bin_edges))
    ]
)
ax06.plot(bin_centers, std, c="k", zorder=-10)

std = np.array(
    [
        np.std((stream_table["pmphi1"].value - mu)[inds == i])
        for i in range(1, len(bin_edges))
    ]
)
ax06.plot(bin_centers, std, c="gray", zorder=-10)

# Model
ax06.fill_between(
    data["phi1"],
    np.exp(np.percentile(dmpars["stream.astrometric.pmphi1", "ln-sigma"], 5, axis=1)),
    np.exp(np.percentile(dmpars["stream.astrometric.pmphi1", "ln-sigma"], 95, axis=1)),
    color=cmap(0.99),
    alpha=0.25,
    where=where,
    zorder=-5,
)
ax06.scatter(
    data["phi1"][where],
    np.exp(mpa["pmphi1", "ln-sigma"][where]),
    s=1,
    color=cmap(0.99),
    zorder=-4,
)

ax06.yaxis.set_label_coords(-0.049, 0.5)
for tick in ax06.get_yticklabels():
    tick.set_verticalalignment("center")

# ---------------------------------------------------------------------------
# PM Phi1

ax07 = fig.add_subplot(
    gs06[1, :],
    xlim=xlim,
    xlabel=r"$\phi_1$ [deg]",
    ylabel=r"$\mu_{1}$ [mas yr$^-1$]",
    ylim=(-15, 15),
    rasterization_zorder=0,
    aspect="auto",
)

k_dist = "pmphi1"
ax07.scatter(  # background
    data["phi1"][psort][~isstrm],
    data["pmphi1"][psort][~isstrm],
    c=stream_prob[psort][~isstrm],
    alpha=alpha[psort][~isstrm],
    s=2,
    zorder=-11,
    cmap="seismic",
)
ax07.scatter(
    phi1,
    stream_table["pmphi1"].to_value("mas/yr"),
    s=10,
    c="k",
    alpha=0.5,
    label="ground truth",
    zorder=-10,
)
ax07.errorbar(  # stream_errors
    data["phi1"][psort][isstrm],
    data["pmphi1"][psort][isstrm],
    yerr=data["pmphi1_err"][psort][isstrm],
    color="gray",
    ls="none",
    zorder=-10,
)
ax07.scatter(  # stream
    data["phi1"][psort][isstrm],
    data["pmphi1"][psort][isstrm],
    c=stream_prob[psort][isstrm],
    alpha=alpha[psort][isstrm],
    s=2,
    zorder=-10,
    cmap="seismic",
)

ax07.plot(
    data["phi1"][where],
    mpa["pmphi1", "mu"][where],
    c="salmon",
    ls="--",
    lw=1,
    label="Model",
    zorder=-9,
)

xlabel = ax07.xaxis.get_label()
xlabel.set_bbox({"facecolor": "white", "edgecolor": "white"})

# =============================================================================
# Slice plots

gs1 = gs[1].subgridspec(4, 4, hspace=0.34, height_ratios=[0.75, 0.75, 0.75, 1.5])

# Legend
legend1 = plt.legend(
    handles=[
        mpl.patches.Patch(facecolor="white", edgecolor="black", label="Truth"),
        mpl.patches.Patch(color=cmap(0.01), label="Background"),
        mpl.patches.Patch(color=cmap(0.99), label="Stream"),
    ],
    ncols=3,
    loc="upper right",
    bbox_to_anchor=(1, -0.125),
)
ax07.add_artist(legend1)


# Bin the data for plotting
bins = np.linspace(data["phi1"].min(), data["phi1"].max(), num=5, endpoint=True)
which_bin = np.digitize(data["phi1"], bins[:-1])

ax100 = ax110 = ax120 = ax130 = None

for i, b in enumerate(np.unique(which_bin)):
    sel = which_bin == b

    # ---------------------------------------------------------------------------
    # Phi2

    ax10i = fig.add_subplot(
        gs1[0, i],
        rasterization_zorder=0,
        sharey=ax100 if ax100 is not None else None,
    )
    ax10i.set_xlabel(r"$\phi_2$ [deg]", labelpad=-1)

    # Connect to top plot(s)
    for ax in (ax01, ax02, ax03, ax04, ax05, ax06, ax07):
        ax.axvline(bins[i], color="gray", ls="--", zorder=-200)
        ax.axvline(bins[i + 1], color="gray", ls="--", zorder=-200)
    smlvis._slices.connect_slices_to_top(  # noqa: SLF001
        fig, ax07, ax10i, left=bins[i], right=bins[i + 1], color="gray"
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
        label=["", "ground truth"],
    )

    ax10i.set_ylim(0.05, ax10i.get_ylim()[-1] + 0.01)
    if i == 0:
        ax10i.set_ylabel("frequency", labelpad=5)
        ax100 = ax10i
    else:
        ax10i.tick_params(labelleft=False)

    # ---------------------------------------------------------------------------
    # Distance

    ax11i = fig.add_subplot(
        gs1[1, i],
        rasterization_zorder=0,
        sharey=ax110 if ax110 is not None else None,
    )
    ax11i.set_xlabel(r"$\varpi$ [mas]", labelpad=-1)

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
        label=["", "ground truth"],
    )

    ax11i.set_ylim(2, ax11i.get_ylim()[-1] + 0.01)
    if i == 0:
        ax11i.set_ylabel("frequency", labelpad=21)
        ax110 = ax11i
    else:
        ax11i.tick_params(labelleft=False)

    # ---------------------------------------------------------------------------
    # PM Phi1

    ax12i = fig.add_subplot(
        gs1[2, i],
        rasterization_zorder=0,
        sharey=ax120 if ax120 is not None else None,
    )
    ax12i.set_xlabel(r"$\mu_1$ [mas/yr]", labelpad=-1)

    # Recovered
    cplxs = np.ones((sel.sum(), 2)) * table["pmphi1"][sel].value[:, None]
    ws = np.stack((bkg_prob[sel], stream_prob[sel]), axis=1)
    ax12i.hist(
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
    ax12i.hist(
        cplxs,
        bins=50,
        weights=ws,
        color=["k", "k"],
        histtype="step",
        density=True,
        stacked=True,
        label=["", "ground truth"],
    )

    if i == 0:
        ax12i.set_ylabel("frequency", labelpad=18)
        ax120 = ax12i
    else:
        ax12i.tick_params(labelleft=False)

    # ------------------------------------------
    # Photometry

    ax13i = fig.add_subplot(
        gs1[3, i],
        xlabel="g-r [mag]",
        ylim=(21, 10),
        rasterization_zorder=0,
        sharey=ax130 if ax130 is not None else None,
    )

    prob = bkg_prob[sel]
    sorter = np.argsort(prob)
    ax13i.scatter(
        (data["g"][sel] - data["r"][sel])[sorter],
        data["g"][sel][sorter],
        c=1 - prob[sorter],
        cmap="seismic",
        alpha=0.25,
        s=1,
        zorder=-10,
    )

    prob = stream_prob[sel]
    sorter = np.argsort(prob)
    ax13i.scatter(
        (data["g"][sel] - data["r"][sel])[sorter],
        data["g"][sel][sorter],
        c=prob[sorter],
        cmap="seismic",
        s=1,
        zorder=-9,
    )

    ax13i.plot(
        stream_isochrone[:, 0] - stream_isochrone[:, 1],
        stream_isochrone[:, 0]
        + np.array(mpars["stream.photometric.distmod", "mu"][sel].mean()),
        c="green",
        label="isochrone",
    )

    ax13i.set_ylim(21, 10)
    if i == 0:
        ax13i.set_ylabel("g [mag]", labelpad=10)
        ax13i.legend(loc="upper left")
        ax130 = ax13i
    else:
        ax13i.tick_params(labelleft=False)

fig.savefig(paths.figures / "mock2" / "results.pdf")
