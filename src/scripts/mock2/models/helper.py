"""Train photometry background flow."""

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
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
from scripts.mock2.model import stream_cp

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
    is_strm = mpars[(f"stream.{WEIGHT_NAME}",)] > -8.0  # was -4

    bkg_prob = np.exp(bkg_lnlik - tot_lnlik)
    stream_prob = np.exp(stream_lnlik - tot_lnlik)
    psort = np.argsort(stream_prob)
    alpha = p2alpha(stream_prob[psort], minval=0.1)

    # ========================================================================
    # Make Figure

    fig = plt.figure(constrained_layout=True, figsize=(14, 20))
    gs = GridSpec(2, 1, height_ratios=(1, 1), figure=fig)
    gs0 = gs[0].subgridspec(6, 1, height_ratios=(1, 5, 5, 5, 5, 5))

    cmap = plt.get_cmap()
    isstream = table["label"] == "stream"

    # ------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------
    # Weight plot

    ax01 = fig.add_subplot(
        gs0[1, :], ylabel="Stream fraction", ylim=(0, 0.075), xticklabels=[]
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

    # ------------------------------------------------------------------------
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
    # Model
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
    # Control region
    ax02.errorbar(
        stream_cp["phi1"].value,
        stream_cp["phi2"].value,
        yerr=stream_cp["w_phi2"].value,
        fmt="o",
        color="r",
    )
    ax02.legend(loc="upper left")
    ax02.set(ylim=(table["phi2"].value.min(), table["phi2"].value.max()))

    # ------------------------------------------------------------------------
    # Distance

    ax03 = fig.add_subplot(
        gs0[3, :],
        xticklabels=[],
        ylabel=r"$\varpi$ [mas]",
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
    # Model
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
    # Control region
    ax03.errorbar(
        stream_cp["phi1"].value,
        stream_cp["parallax"].value,
        yerr=stream_cp["w_parallax"].value,
        fmt="o",
        color="r",
    )
    ax03.legend(loc="upper left")
    ax03.set(ylim=(table["parallax"].value.min(), table["parallax"].value.max()))

    # ------------------------------------------------------------------------
    # PM 1

    ax04 = fig.add_subplot(
        gs0[4, :],
        xticklabels=[],
        ylabel=r"$\mu_{\phi_1}$ [mas/yr]",
        ylim=(-25, 25),
        rasterization_zorder=0,
    )

    ax04.scatter(
        phi1,
        table["pmphi1"].to_value("mas/yr")[isstream],
        s=10,
        c="k",
        alpha=0.5,
        zorder=-100,
        label="Ground Truth",
    )
    ax04.scatter(
        data["phi1"][psort],
        data["pmphi1"][psort],
        c=stream_prob[psort],
        alpha=alpha,
        s=2,
        zorder=-50,
        cmap="seismic",
    )
    # Model
    ax04.fill_between(
        data["phi1"][is_strm],
        mpa["pmphi1", "mu"][is_strm] - xp.exp(mpa["pmphi1", "ln-sigma"])[is_strm],
        mpa["pmphi1", "mu"][is_strm] + xp.exp(mpa["pmphi1", "ln-sigma"])[is_strm],
        color="k",
        alpha=0.15,
        zorder=-10,
    )
    ax04.plot(
        data["phi1"][is_strm],
        mpa["pmphi1", "mu"][is_strm],
        c="k",
        ls="--",
        lw=2,
        label="Model (MLE)",
        zorder=-5,
    )
    # Control region
    ax04.errorbar(
        stream_cp["phi1"].value,
        stream_cp["pmphi1"].value,
        yerr=stream_cp["w_pmphi1"].value,
        fmt="o",
        color="r",
    )
    ax04.legend(loc="upper left")

    # ------------------------------------------------------------------------
    # PM 2

    ax05 = fig.add_subplot(
        gs0[5, :],
        xlabel=r"$\phi_1$ [deg]",
        ylabel=r"$\mu_{\phi_2}$ [mas/yr]",
        ylim=(-25, 25),
        rasterization_zorder=0,
    )

    ax05.scatter(
        phi1,
        table["pmphi2"].to_value("mas/yr")[isstream],
        s=10,
        c="k",
        alpha=0.5,
        zorder=-100,
        label="Ground Truth",
    )
    ax05.scatter(
        data["phi1"][psort],
        data["pmphi2"][psort],
        c=stream_prob[psort],
        alpha=alpha,
        s=2,
        zorder=-50,
        cmap="seismic",
    )
    # ax05.fill_between(
    #     data["phi1"][is_strm],
    #     mpa["pmphi2", "mu"][is_strm] - xp.exp(mpa["pmphi2", "ln-sigma"][is_strm]),
    #     mpa["pmphi2", "mu"][is_strm] + xp.exp(mpa["pmphi2", "ln-sigma"][is_strm]),
    #     color="k",
    #     alpha=0.15,
    #     zorder=-10,
    # )
    # ax05.plot(
    #     data["phi1"][is_strm],
    #     mpa["pmphi2", "mu"][is_strm],
    #     c="k",
    #     ls="--",
    #     lw=2,
    #     label="Model (MLE)",
    #     zorder=-5,
    # )
    ax05.legend(loc="upper left")

    # ========================================================================
    # Slice plots

    gs1 = gs[1].subgridspec(5, 4)

    # Bin the data for plotting
    bins = np.linspace(data["phi1"].min(), data["phi1"].max(), num=5, endpoint=True)
    which_bin = np.digitize(data["phi1"], bins[:-1])

    for i, thebin in enumerate(np.unique(which_bin)):
        sel = which_bin == thebin

        # ---------------------------------------------------------------------
        # Phi2

        ax10i = fig.add_subplot(gs1[0, i], xlabel=r"$\phi_2$ [$\degree$]")

        # Connect to top plot(s)
        for ax in (ax01, ax02, ax03, ax04, ax05):
            ax.axvline(bins[i], color="gray", ls="--", zorder=-200)
            ax.axvline(bins[i + 1], color="gray", ls="--", zorder=-200)
        smlvis._slices.connect_slices_to_top(  # noqa: SLF001
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

        # Distribution
        a, b = model["background"]["astrometric"]["phi2"].coord_bounds["phi2"]
        x = xp.linspace(a - 0.1, b + 0.1, 100)
        ax10i.plot(x, xp.ones_like(x) / (b - a))

        if i == 0:
            ax10i.set_ylabel("frequency")
            ax10i.legend(loc="upper left")

        # --------------------------------------------------------------------
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

        # Distribution
        a, b = model["background"]["astrometric"]["parallax"].coord_bounds["parallax"]
        m = mpars["background.astrometric.parallax.parallax", "slope"][sel].mean()
        x = xp.linspace(a, b, 100)
        ax11i.plot(
            x,
            xp.exp(
                xp.log(xp.abs(m))
                + (m * (b - x))
                - xp.log(xp.abs(xp.expm1(m * (b - a))))
            ),
        )

        if i == 0:
            ax11i.set_ylabel("frequency")
            ax11i.legend(loc="upper left")

        # --------------------------------------------------------------------
        # PM 1

        ax12i = fig.add_subplot(gs1[2, i], xlabel=r"$\mu_{\phi_1}$ [mas/yr]")

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
            label=["", "Ground Truth"],
        )

        a, b = model["background"]["astrometric"]["pm"].coord_bounds["pmphi1"]
        loc = mpars["background.astrometric.pm.pmphi1", "mu"].mean()
        scale = xp.exp(mpars["background.astrometric.pm.pmphi1", "ln-sigma"].mean())
        x = xp.linspace(a, b, 100)
        ax12i.plot(x, scipy.stats.norm.pdf(x, loc, scale))

        if i == 0:
            ax12i.set_ylabel("frequency")
            ax12i.legend(loc="upper left")

        # --------------------------------------------------------------------
        # PM 2

        ax13i = fig.add_subplot(gs1[3, i], xlabel=r"$\mu_{\phi_2}$ [mas/yr]")

        # Recovered
        cplxs = np.ones((sel.sum(), 2)) * table["pmphi2"][sel].value[:, None]
        ws = np.stack((bkg_prob[sel], stream_prob[sel]), axis=1)
        ax13i.hist(
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
        ax13i.hist(
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
            ax13i.set_ylabel("frequency")
            ax13i.legend(loc="upper left")

        # --------------------------------------------------------------------
        # Photometry

        # ------------------------------------------
        # Stream

        ax14i = fig.add_subplot(gs1[4, i], xticklabels=[])

        prob = stream_prob[sel]
        sorter = np.argsort(prob)
        ax14i.scatter(
            (data["g"] - data["r"])[sel][sorter],
            data["r"][sel][sorter],
            c=prob[sorter],
            cmap="seismic",
            s=1,
            rasterized=True,
        )

        ax14i.set_xlabel("g-r [mag]")
        if i == 0:
            ax14i.set_ylabel("r [mag]")

    return fig
