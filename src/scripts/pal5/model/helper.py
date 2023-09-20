"""Helper function."""
# ruff: noqa: ERA001


import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch as xp
from astropy.coordinates import Distance
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

# =============================================================================

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")


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
    stream_cutoff = stream_weight > 2e-2

    bkg_prob = xp.exp(bkg_lnlik - tot_lnlik)
    stream_prob = xp.exp(stream_lnlik - tot_lnlik)
    allstream_prob = xp.exp(stream_lnlik - tot_lnlik)

    psort = np.argsort(stream_prob)
    pmax = xp.max(stream_prob.max(), xp.tensor(0.95))
    pmin = stream_prob.min()

    # =============================================================================
    # Make Figure

    fig = plt.figure(constrained_layout="tight", figsize=(11, 15))
    gs = GridSpec(2, 1, figure=fig, height_ratios=(1.2, 1), hspace=0)
    gs0 = gs[0].subgridspec(6, 1, height_ratios=(1, 5, 5, 5, 5, 5))

    cmap = plt.get_cmap("Stream1")

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
        gs0[1, :],
        ylabel="Stream fraction",
        ylim=(1e-4, 1),
        xticklabels=[],
        yscale="log",
    )

    ax01.plot(data["phi1"], stream_weight, c="k", ls="--", lw=2, label="Model (MLE)")
    ax01.legend(loc="upper left")

    # ---------------------------------------------------------------------------
    # Phi2

    mpa = mpars.get_prefixed("stream.astrometric")

    ax02 = fig.add_subplot(
        gs0[2, :],
        xticklabels=[],
        xlabel="",
        ylabel=r"$\phi_2$ [$\degree$]",
        rasterization_zorder=0,
        ylim=(np.nanmin(data["phi2"]), np.nanmax(data["phi2"])),
    )

    ax02.scatter(
        data["phi1"][psort],
        data["phi2"][psort],
        c=allstream_prob[psort],
        alpha=0.1 + (1 - 0.1) / (pmax - pmin) * (allstream_prob[psort] - pmin),
        s=2,
        zorder=-10,
    )
    ax02.fill_between(
        data["phi1"][stream_cutoff],
        (mpa["phi2", "mu"] - xp.exp(mpa["phi2", "ln-sigma"]))[stream_cutoff],
        (mpa["phi2", "mu"] + xp.exp(mpa["phi2", "ln-sigma"]))[stream_cutoff],
        color="k",
        alpha=0.25,
        label="Model (MLE)",
    )

    pal5_cp = QTable.read(paths.data / "pal5" / "control_points_stream.ecsv")
    ax02.errorbar(
        pal5_cp["phi1"].value,
        pal5_cp["phi2"].value,
        yerr=pal5_cp["w_phi2"].value,
        ls="none",
    )

    ax02.legend(loc="upper left")

    # # ---------------------------------------------------------------------------
    # # PM-Phi1

    ax03 = fig.add_subplot(
        gs0[3, :],
        ylabel=r"$\mu_{\phi_1}^*$ [mas yr$^{-1}$]",
        xticklabels=[],
        rasterization_zorder=0,
    )

    # ax03.scatter(
    #     data["phi1"][psort],
    #     data["pmphi1"][psort],
    #     c=allstream_prob[psort],
    #     alpha=0.1 + (1 - 0.1) / (pmax - pmin) * (stream_prob[psort] - pmin),
    #     s=2,
    #     zorder=-10,
    # )
    # ax03.set_rasterization_zorder(0)
    # ax03.fill_between(
    #     data["phi1"][stream_cutoff],
    #     (mpa["pmphi1", "mu"] - xp.exp(mpa["pmphi1", "ln-sigma"]))[stream_cutoff],
    #     (mpa["pmphi1", "mu"] + xp.exp(mpa["pmphi1", "ln-sigma"]))[stream_cutoff],
    #     color="k",
    #     alpha=0.25,
    #     label="Model (MLE)",
    # )
    # ax03.legend(loc="upper left")

    # # ---------------------------------------------------------------------------
    # # PM-Phi2

    ax04 = fig.add_subplot(gs0[4, :])
    ax04.set_xticklabels([])
    ax04.set(ylabel=r"$\mu_{\phi_2}$ [mas yr$^{-1}$]")

    # ax04.scatter(
    #     data["phi1"][psort],
    #     data["pmphi2"][psort],
    #     c=allstream_prob[psort],
    #     alpha=0.1 + (1 - 0.1) / (pmax - pmin) * (stream_prob[psort] - pmin),
    #     s=2,
    #     zorder=-10,
    # )
    # ax04.set_rasterization_zorder(0)
    # ax04.fill_between(
    #     data["phi1"][stream_cutoff],
    #     (mpa["pmphi2", "mu"] - xp.exp(mpa["pmphi2", "ln-sigma"]))[stream_cutoff],
    #     (mpa["pmphi2", "mu"] + xp.exp(mpa["pmphi2", "ln-sigma"]))[stream_cutoff],
    #     color="k",
    #     alpha=0.25,
    #     label="Model (MLE)",
    # )
    # ax04.legend(loc="upper left")

    # # ---------------------------------------------------------------------------
    # # Distance

    ax03 = fig.add_subplot(gs0[5, :], xlabel=r"$\phi_1$ [deg]", ylabel=r"$d$ [kpc]")

    try:
        mpa = mpars["stream.photometric.distmod"]
    except KeyError:
        skip_phot = True
    else:
        skip_phot = False

    if not skip_phot:
        d2sm = Distance(distmod=(mpa["mu"] - 2 * xp.exp(mpa["ln-sigma"])) * u.mag)
        d2sp = Distance(distmod=(mpa["mu"] + 2 * xp.exp(mpa["ln-sigma"])) * u.mag)
        d1sm = Distance(distmod=(mpa["mu"] - xp.exp(mpa["ln-sigma"])) * u.mag)
        d1sp = Distance(distmod=(mpa["mu"] + xp.exp(mpa["ln-sigma"])) * u.mag)

        ax03.fill_between(
            data["phi1"][stream_cutoff],
            d2sm[stream_cutoff].to_value("kpc"),
            d2sp[stream_cutoff].to_value("kpc"),
            alpha=0.15,
            color="k",
        )
        ax03.fill_between(
            data["phi1"][stream_cutoff],
            d1sm[stream_cutoff].to_value("kpc"),
            d1sp[stream_cutoff].to_value("kpc"),
            alpha=0.25,
            color="k",
            label="Model (MLE)",
        )
        ax03.legend(loc="upper left")

    # =============================================================================
    # Slice plots

    gs1 = gs[1].subgridspec(5, 4, height_ratios=(0.01, 1, 1, 1, 2), hspace=0)

    # Bin the data for plotting
    bins = np.linspace(data["phi1"].min(), data["phi1"].max(), num=5, endpoint=True)
    which_bin = np.digitize(data["phi1"], bins[:-1])

    # Legend
    ax10 = fig.add_subplot(gs1[0, :])
    ax10.axis(False)
    ax10.legend(
        handles=[
            mpl.patches.Patch(color=cmap(0.01), label="Background"),
            mpl.lines.Line2D([0], [0], color="k", lw=4),
            mpl.patches.Patch(color=cmap(0.99), label="Stream"),
        ],
        ncols=4,
    )

    for i, b in enumerate(np.unique(which_bin)):
        sel = which_bin == b

        data_ = data[sel]
        bkg_prob_ = bkg_prob[sel]
        stream_prob_ = stream_prob[sel]

        # ---------------------------------------------------------------------------
        # Phi2

        ax11i = fig.add_subplot(gs1[1, i], xlabel=r"$\phi_2$ [$\degree$]")

        # Connect to top plot(s)
        for ax in (ax01, ax02, ax03, ax04):
            ax.axvline(bins[i], color="gray", ls="--", zorder=-200)
            ax.axvline(bins[i + 1], color="gray", ls="--", zorder=-200)
        smlvis._slices.connect_slices_to_top(  # noqa: SLF001
            fig, ax03, ax11i, left=bins[i], right=bins[i + 1], color="gray"
        )

        notna = ~np.isnan(data_["phi2"])
        cphi2s = np.ones((sel.sum(), 2)) * data_["phi2"][:, None].numpy()
        ws = np.stack((bkg_prob_, stream_prob_), axis=1)
        ax11i.hist(
            cphi2s[notna],
            bins=50,
            weights=ws[notna],
            color=[cmap(0.01), cmap(0.99)],
            alpha=0.75,
            density=True,
            stacked=True,
            label=["", "Stream Model (MLE)"],
        )

        xmin, xmax = data["phi2"].min().numpy(), data["phi2"].max().numpy()
        x = np.linspace(xmin, xmax)
        bkg_wgt = mpars["background.weight",][sel].mean()
        m = mpars["background.astrometric.phi2", "slope"][sel].mean()
        ax11i.plot(x, bkg_wgt * exp_distr(m, xmin, xmax).pdf(x), c="k")

        if i == 0:
            ax11i.set_ylabel("frequency")

        # # ---------------------------------------------------------------------------
        # # PM-Phi1

        ax12i = fig.add_subplot(gs1[2, i], xlabel=r"$\mu_{\phi_1}^*$ [mas yr$^{-1}$]")

        # Recovered
        cpmphi1s = np.ones((sel.sum(), 2)) * data_["pmphi1"][:, None].numpy()
        ws = np.stack((bkg_prob_, stream_prob_), axis=1)
        ax12i.hist(
            cpmphi1s,
            bins=50,
            weights=ws,
            color=[cmap(0.01), cmap(0.99)],
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

        # # ---------------------------------------------------------------------------
        # # PM-Phi2

        ax13i = fig.add_subplot(gs1[3, i], xlabel=r"$\mu_{phi_2}$ [mas yr$^{-1}$]")
        ax13i.hist(
            np.ones((sel.sum(), 2)) * data_["pmphi2"][:, None].numpy(),
            bins=50,
            weights=np.stack((bkg_prob_, stream_prob_), axis=1),
            color=[cmap(0.01), cmap(0.99)],
            alpha=0.75,
            density=True,
            stacked=True,
            label=["", "Stream Model (MLE)"],
        )

        # xmin, xmax = data["pmphi2"].min().numpy(), data["pmphi2"].max().numpy()
        # x = np.linspace(xmin, xmax)
        # m = mpars["background.astrometric.pmphi2", "slope"][sel].mean()
        # ax13i.plot(x, bkg_wgt * exp_distr(m, xmin, xmax).pdf(x), c="k")

        if i == 0:
            ax13i.set_ylabel("frequency")

        # # ---------------------------------------------------------------------------
        # # Photometry

        ax14i = fig.add_subplot(
            gs1[4, i],
            xlabel=("g - r [mag]"),
            xlim=(0, 1),
            ylim=(22, 13),
            xticklabels=[],
            rasterization_zorder=20,
        )

        sorter = np.argsort(stream_prob_)
        ax14i.scatter(
            data_["g"][sorter] - data_["r"][sorter],
            data_["g"][sorter],
            c=stream_prob_[sorter],
            s=1,
            rasterized=True,
        )

        if i == 0:
            ax14i.set_ylabel("g [mag]")
        else:
            ax14i.set_yticklabels([])

    return fig
