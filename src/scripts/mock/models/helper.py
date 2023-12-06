"""Train photometry background flow."""

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch as xp
from astropy.table import QTable
from matplotlib.gridspec import GridSpec
from showyourwork.paths import user as user_paths

import stream_mapper.visualization as smlvis
from stream_mapper.core import WEIGHT_NAME, ModelAPI
from stream_mapper.pytorch import Data

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.helper import manually_set_dropout, p2alpha

# =============================================================================

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")


def diagnostic_plot(
    model: ModelAPI, data: Data, where: Data, table: QTable
) -> plt.Figure:
    """Plot the model."""
    # Evaluate model
    with xp.no_grad():
        model.train()
        mpars = model.unpack_params(model(data))
        model.eval()

        stream_lnlik = model.component_ln_posterior("stream", mpars, data, where=where)
        bkg_lnlik = model.component_ln_posterior("background", mpars, data, where=where)
        tot_lnlik = np.logaddexp(stream_lnlik, bkg_lnlik)

    # Determine if the point is in the stream
    is_strm = mpars[(f"stream.{WEIGHT_NAME}",)] > -4

    bkg_prob = np.exp(bkg_lnlik - tot_lnlik)
    stream_prob = np.exp(stream_lnlik - tot_lnlik)
    psort = np.argsort(stream_prob)
    alpha = p2alpha(stream_prob[psort], minval=0.1)

    # =============================================================================
    # Make Figure

    fig = plt.figure(constrained_layout=True, figsize=(14, 13))
    gs = GridSpec(2, 1, height_ratios=(1, 1), figure=fig)
    gs0 = gs[0].subgridspec(4, 1, height_ratios=(1, 5, 5, 5))

    cmap = plt.get_cmap()
    isstream = table["label"] == "stream"

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

    ax01 = fig.add_subplot(
        gs0[1, :], ylabel="Stream fraction", ylim=(0, 0.5), xticklabels=[]
    )

    # Truth
    phi1 = table["phi1"].to_value("deg")[isstream]

    Hs, bin_edges = np.histogram(phi1, bins=75)
    Ht, bin_edges = np.histogram(data["phi1"], bins=bin_edges)
    ax01.bar(bin_edges[:-1], Hs / Ht, width=bin_edges[1] - bin_edges[0])

    with xp.no_grad():
        manually_set_dropout(model, 0.15)
        weights = xp.stack(
            [
                model.unpack_params(model(data))[f"stream.{WEIGHT_NAME}",]
                for i in range(100)
            ],
            1,
        )
        manually_set_dropout(model, 0)
    ax01.fill_between(
        data["phi1"],
        np.exp(np.percentile(weights, 5, axis=1)),
        np.exp(np.percentile(weights, 95, axis=1)),
        color="k",
        alpha=0.25,
    )
    ax01.plot(
        data["phi1"],
        np.exp(mpars[(f"stream.{WEIGHT_NAME}",)]),
        c="k",
        ls="--",
        lw=2,
        label="Model (MLE)",
    )

    ax01.legend(loc="upper left")

    # ---------------------------------------------------------------------------
    # Phi2

    mpa = mpars.get_prefixed("stream.astrometric")

    ax02 = fig.add_subplot(
        gs0[2, :],
        xticklabels=[],
        ylabel=r"$\phi_2$ [$\degree$]",
        rasterization_zorder=0,
    )

    ax02.scatter(
        phi1,
        table["phi2"].to_value("deg")[isstream],
        s=10,
        c="k",
        alpha=0.5,
        zorder=-100,
        label="Ground Truth",
    )
    ax02.scatter(
        data["phi1"][psort],
        data["phi2"][psort],
        c=stream_prob[psort],
        alpha=alpha,
        s=2,
        zorder=-50,
        cmap="seismic",
    )
    ax02.fill_between(
        data["phi1"][is_strm],
        mpa["phi2", "mu"][is_strm] - xp.exp(mpa["phi2", "ln-sigma"][is_strm]),
        mpa["phi2", "mu"][is_strm] + xp.exp(mpa["phi2", "ln-sigma"][is_strm]),
        color="k",
        alpha=0.15,
        zorder=-10,
    )
    ax02.plot(
        data["phi1"][is_strm],
        mpa["phi2", "mu"][is_strm],
        c="k",
        ls="--",
        lw=2,
        label="Model (MLE)",
        zorder=-5,
    )
    ax02.legend(loc="upper left")

    # ---------------------------------------------------------------------------
    # Distance

    ax03 = fig.add_subplot(
        gs0[3, :],
        xlabel=r"$\phi_1$ [deg]",
        ylabel=r"$\varpi$ [mas yr$^-1$]",
        rasterization_zorder=0,
    )

    ax03.scatter(
        phi1,
        table["parallax"].to_value("mas")[isstream],
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
        alpha=alpha,
        s=2,
        zorder=-50,
        cmap="seismic",
    )
    ax03.fill_between(
        data["phi1"][is_strm],
        mpa["parallax", "mu"][is_strm] - xp.exp(mpa["parallax", "ln-sigma"][is_strm]),
        mpa["parallax", "mu"][is_strm] + xp.exp(mpa["parallax", "ln-sigma"][is_strm]),
        color="k",
        alpha=0.15,
        zorder=-10,
    )
    ax03.plot(
        data["phi1"][is_strm],
        mpa["parallax", "mu"][is_strm],
        c="k",
        ls="--",
        lw=2,
        label="Model (MLE)",
        zorder=-5,
    )
    ax03.legend(loc="upper left")

    # =============================================================================
    # Slice plots

    gs1 = gs[1].subgridspec(3, 4)

    # Bin the data for plotting
    bins = np.linspace(data["phi1"].min(), data["phi1"].max(), num=5, endpoint=True)
    which_bin = np.digitize(data["phi1"], bins[:-1])

    for i, b in enumerate(np.unique(which_bin)):
        sel = which_bin == b

        # ---------------------------------------------------------------------------
        # Phi2

        ax10i = fig.add_subplot(gs1[0, i], xlabel=r"$\phi_2$ [$\degree$]")

        # Connect to top plot(s)
        for ax in (ax01, ax02):
            ax.axvline(bins[i], color="gray", ls="--", zorder=-200)
            ax.axvline(bins[i + 1], color="gray", ls="--", zorder=-200)
        smlvis._slices.connect_slices_to_top(  # noqa: SLF001
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

        if i == 0:
            ax10i.set_ylabel("frequency")
            ax10i.legend(loc="upper left")

        # ---------------------------------------------------------------------------
        # Distance

        ax11i = fig.add_subplot(gs1[1, i], xlabel=r"$\varpi$ [mas]")

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
            ax11i.legend(loc="upper left")

        # ---------------------------------------------------------------------------
        # Photometry

        # ------------------------------------------
        # Stream

        ax12i = fig.add_subplot(gs1[2, i], xticklabels=[])

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

        if i == 0:
            ax12i.set_ylabel("r [mag]")
        else:
            ax12i.set_yticklabels([])

    return fig
