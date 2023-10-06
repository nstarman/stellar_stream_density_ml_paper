"""Helper function."""
# ruff: noqa: ERA001


import sys

import asdf
import galstreams
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch as xp
from astropy.table import QTable
from matplotlib.gridspec import GridSpec
from showyourwork.paths import user as user_paths

import stream_ml.visualization as smlvis
from stream_ml.core import ModelAPI
from stream_ml.pytorch import Data
from stream_ml.visualization.background import (
    exponential_like_distribution as exp_distr,
)

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.helper import color_by_probable_member, p2alpha
from scripts.mpl_colormaps import stream_cmap1 as cmap1
from scripts.pal5.datasets import masks
from scripts.pal5.frames import pal5_frame as frame

# =============================================================================
# Configuration

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")


# =============================================================================
# Load data

# galstreams
allstreams = galstreams.MWStreams(implement_Off=True)
pal5I21 = allstreams["Pal5-I21"].track.transform_to(frame)
pal5PW19 = allstreams["Pal5-PW19"].track.transform_to(frame)

# Control points
pal5_cp = QTable.read(paths.data / "pal5" / "control_points_stream.ecsv")

# Isochrone data
with asdf.open(
    paths.data / "pal5" / "isochrone.asdf", "r", lazy_load=False, copy_arrays=True
) as af:
    isochrone_data = Data(**af["isochrone_data"])

# Progenitor
progenitor_prob = np.zeros(len(masks))
progenitor_prob[~masks["Pal5"]] = 1


# =============================================================================


def diagnostic_plot(model: ModelAPI, data: Data, where: Data) -> plt.Figure:
    """Plot the model."""
    # Evaluate model
    with xp.no_grad():
        model.eval()
        mpars = model.unpack_params(model(data))

        stream_lnlik = model.component_ln_posterior("stream", mpars, data, where=where)
        bkg_lnlik = model.component_ln_posterior("background", mpars, data, where=where)
        # tot_lnlik = model.ln_posterior(mpars, data, where=where)  # FIXME!
        tot_lnlik = xp.logaddexp(stream_lnlik, bkg_lnlik)

    stream_weight = mpars[("stream.weight",)]
    stream_cutoff = stream_weight > 0  # everything has weight > 0

    bkg_prob = xp.exp(bkg_lnlik - tot_lnlik)
    stream_prob = xp.exp(stream_lnlik - tot_lnlik)
    allstream_prob = stream_prob

    psort = np.argsort(allstream_prob)

    # =============================================================================
    # Make Figure

    fig = plt.figure(constrained_layout="tight", figsize=(11, 13))
    gs = GridSpec(2, 1, figure=fig, height_ratios=(6, 5), hspace=0.12)
    gs0 = gs[0].subgridspec(5, 1, height_ratios=(1, 5, 5, 5, 5))

    colors = color_by_probable_member((stream_prob, cmap1))
    alphas = p2alpha(allstream_prob[psort])

    # ---------------------------------------------------------------------------
    # Colormap

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
        ylabel="Stream fraction",
        ylim=(1e-4, 1),
        xticklabels=[],
        yscale="log",
    )
    _bounds_kw = {"c": "gray", "ls": "-", "lw": 2, "alpha": 0.8}
    ax01.axhline(model.params[("stream.weight",)].bounds.lower[0], **_bounds_kw)
    ax01.axhline(model.params[("stream.weight",)].bounds.upper[0], **_bounds_kw)
    ax01.plot(data["phi1"], stream_weight, c="k", ls=":", lw=2, label="Model (MLE)")
    ax01.legend(loc="upper left")

    # ---------------------------------------------------------------------------
    # Phi2

    ax02 = fig.add_subplot(
        gs0[2, :],
        xticklabels=[],
        xlabel="",
        ylabel=r"$\phi_2$ [$\degree$]",
        rasterization_zorder=0,
        ylim=(np.nanmin(data["phi2"]), np.nanmax(data["phi2"])),
    )
    mpa = mpars.get_prefixed("stream.astrometric")

    # Data, colored by stream probability
    ax02.scatter(
        data["phi1"][psort],
        data["phi2"][psort],
        c=colors[psort],
        alpha=alphas,
        s=2,
        zorder=-10,
    )

    # Control points
    ax02.errorbar(
        pal5_cp["phi1"].value,
        pal5_cp["phi2"].value,
        yerr=pal5_cp["w_phi2"].value,
        ls="none",
        zorder=-9,
    )

    # Model
    ax02.fill_between(
        data["phi1"][stream_cutoff],
        (mpa["phi2", "mu"] - xp.exp(mpa["phi2", "ln-sigma"]))[stream_cutoff],
        (mpa["phi2", "mu"] + xp.exp(mpa["phi2", "ln-sigma"]))[stream_cutoff],
        color="k",
        alpha=0.25,
        label="Model (MLE)",
    )

    # Literature
    ax02.plot(
        pal5I21.phi1.degree,
        pal5I21.phi2.degree,
        c="k",
        ls="--",
        alpha=0.5,
        label="Ibata+21",
    )
    ax02.plot(
        pal5PW19.phi1.degree,
        pal5PW19.phi2.degree,
        c="k",
        ls="--",
        alpha=0.5,
        label="PW+19",
    )

    ax02.legend(loc="upper left")

    # ---------------------------------------------------------------------------
    # PM-Phi1

    ax03 = fig.add_subplot(
        gs0[3, :],
        ylabel=r"$\mu_{\phi_1}^*$ [mas yr$^{-1}$]",
        xticklabels=[],
        rasterization_zorder=0,
        ylim=(np.nanmin(data["pmphi1"]), np.nanmax(data["pmphi1"])),
    )

    ax03.scatter(
        data["phi1"][psort],
        data["pmphi1"][psort],
        c=colors[psort],
        alpha=alphas,
        s=2,
        zorder=-10,
    )
    ax03.fill_between(
        data["phi1"][stream_cutoff],
        (mpa["pmphi1", "mu"] - xp.exp(mpa["pmphi1", "ln-sigma"]))[stream_cutoff],
        (mpa["pmphi1", "mu"] + xp.exp(mpa["pmphi1", "ln-sigma"]))[stream_cutoff],
        color="k",
        alpha=0.25,
    )

    # Control points
    ax03.errorbar(
        pal5_cp["phi1"].value,
        pal5_cp["pmphi1"].value,
        yerr=pal5_cp["w_pmphi1"].value,
        ls="none",
    )

    # ---------------------------------------------------------------------------
    # PM-Phi2

    ax04 = fig.add_subplot(
        gs0[4, :],
        ylabel=r"$\mu_{\phi_2}$ [mas yr$^{-1}$]",
        # xticklabels=[],
        xlabel=r"$\phi_1$ [deg]",
    )

    ax04.scatter(
        data["phi1"][psort],
        data["pmphi2"][psort],
        c=colors[psort],
        alpha=alphas,
        s=2,
        zorder=-10,
    )

    # Control points
    ax04.errorbar(
        pal5_cp["phi1"].value,
        pal5_cp["pmphi2"].value,
        yerr=pal5_cp["w_pmphi2"].value,
        ls="none",
    )
    ax04.fill_between(
        data["phi1"][stream_cutoff],
        (mpa["pmphi2", "mu"] - xp.exp(mpa["pmphi2", "ln-sigma"]))[stream_cutoff],
        (mpa["pmphi2", "mu"] + xp.exp(mpa["pmphi2", "ln-sigma"]))[stream_cutoff],
        color="k",
        alpha=0.25,
    )

    # =============================================================================
    # Slice plots

    legend1 = plt.legend(
        handles=[
            mpl.patches.Patch(color=cmap1(0.99), label="Stream (MLE)"),
            mpl.patches.Patch(color=cmap1(0.01), label="Background (MLE)"),
            mpl.lines.Line2D(
                [0], [0], color="k", lw=3, ls="-", label="Background Distribution"
            ),
        ],
        ncols=4,
        loc="upper right",
        bbox_to_anchor=(1, -0.36),
    )
    ax04.add_artist(legend1)

    gs1 = gs[1].subgridspec(4, 4, height_ratios=(1, 1, 1, 2), hspace=0)

    # Bin the data for plotting
    bins = np.linspace(data["phi1"].min(), data["phi1"].max(), num=5, endpoint=True)
    which_bin = np.digitize(data["phi1"], bins[:-1])

    for i, b in enumerate(np.unique(which_bin)):
        sel = which_bin == b

        data_ = data[sel]
        bkg_prob_ = bkg_prob[sel]
        stream_prob_ = stream_prob[sel]

        # ---------------------------------------------------------------------------
        # Phi2

        ax10i = fig.add_subplot(gs1[0, i], xlabel=r"$\phi_2$ [$\degree$]")

        # Plot vertical lines
        for ax in (ax01, ax02, ax03, ax04, ax04):
            ax.axvline(bins[i], color="gray", ls="--", zorder=-200)
            ax.axvline(bins[i + 1], color="gray", ls="--", zorder=-200)
        # Connect to top plot(s)
        smlvis._slices.connect_slices_to_top(  # noqa: SLF001
            fig, ax04, ax10i, left=bins[i], right=bins[i + 1], color="gray"
        )

        cphi2s = np.ones((sel.sum(), 3)) * data_["phi2"][:, None].numpy()
        ws = np.stack((bkg_prob_, stream_prob_, progenitor_prob[sel]), axis=1)
        ax10i.hist(
            cphi2s,
            bins=50,
            weights=ws,
            color=[cmap1(0.01), cmap1(0.99), "yellow"],
            alpha=0.75,
            density=True,
            stacked=True,
            log=True,
            label=["", "Stream Model (MLE)"],
        )

        xmin, xmax = data["phi2"].min().numpy(), data["phi2"].max().numpy()
        x = np.linspace(xmin, xmax)
        bkg_wgt = mpars["background.weight",][sel].mean()
        m = mpars["background.astrometric.phi2.phi2", "slope"][sel].mean()
        ax10i.plot(x, bkg_wgt * exp_distr(m, xmin, xmax).pdf(x), c="k")

        if i == 0:
            ax10i.set_ylabel("frequency")

        # ---------------------------------------------------------------------------
        # PM-Phi1

        ax12i = fig.add_subplot(gs1[1, i], xlabel=r"$\mu_{\phi_1}^*$ [mas yr$^{-1}$]")

        # Recovered
        cpmphi1s = np.ones((sel.sum(), 2)) * data_["pmphi1"][:, None].numpy()
        ws = np.stack((bkg_prob_, stream_prob_), axis=1)
        ax12i.hist(
            cpmphi1s,
            bins=50,
            weights=ws,
            color=[cmap1(0.01), cmap1(0.99)],
            alpha=0.75,
            density=True,
            stacked=True,
            label=["", "Stream Model (MLE)"],
        )

        # xmin, xmax = data["pmphi1"].min().numpy(), data["pmphi1"].max().numpy()
        # x = np.linspace(xmin, xmax)
        # m = mpars["background.astrometric.pmphi1", "slope"][sel].mean()
        # ax12i.plot(x, bkg_wgt * exp_distr(m, xmin, xmax).pdf(x), c="k")

        if i == 0:
            ax12i.set_ylabel("frequency")

        # ---------------------------------------------------------------------------
        # PM-Phi2

        ax12i = fig.add_subplot(gs1[2, i], xlabel=r"$\mu_{phi_2}$ [mas yr$^{-1}$]")
        ax12i.hist(
            np.ones((sel.sum(), 2)) * data_["pmphi2"][:, None].numpy(),
            bins=50,
            weights=np.stack((bkg_prob_, stream_prob_), axis=1),
            color=[cmap1(0.01), cmap1(0.99)],
            alpha=0.75,
            density=True,
            stacked=True,
            label=["", "Stream Model (MLE)"],
        )

        # xmin, xmax = data["pmphi2"].min().numpy(), data["pmphi2"].max().numpy()
        # x = np.linspace(xmin, xmax)
        # m = mpars["background.astrometric.pmphi2", "slope"][sel].mean()
        # ax12i.plot(x, bkg_wgt * exp_distr(m, xmin, xmax).pdf(x), c="k")

        if i == 0:
            ax12i.set_ylabel("frequency")

        # ---------------------------------------------------------------------------
        # Photometry

        ax13i = fig.add_subplot(
            gs1[3, i],
            xlabel=("g - r [mag]"),
            xlim=(0, 1),
            ylim=(22, 13),
            xticklabels=[],
            rasterization_zorder=20,
        )

        sorter = np.argsort(allstream_prob[sel])
        ax13i.scatter(
            data_["g"][sorter] - data_["r"][sorter],
            data_["g"][sorter],
            c=colors[sel][sorter],
            s=1,
        )

        # # isochrone
        # ax13i.plot(
        #     isochrone_data["g"] - isochrone_data["r"],
        #     isochrone_data["g"]
        #     + mpars["stream.photometric.distmod", "mu"][sel].mean().numpy(),
        #     c="green",
        # )

        if i == 0:
            ax13i.set_ylabel("g [mag]")
        else:
            ax13i.set_yticklabels([])

    return fig
