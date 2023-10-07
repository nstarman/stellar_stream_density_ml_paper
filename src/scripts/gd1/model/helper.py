"""GD-1 model helper functions."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import asdf
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch as xp
from astropy.coordinates import Distance
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerTuple
from scipy import stats
from showyourwork.paths import user as user_paths

import stream_ml.visualization as smlvis
from scripts import helper
from stream_ml.core import WEIGHT_NAME, Data
from stream_ml.visualization.background import (
    exponential_like_distribution as exp_distr,
)

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.helper import color_by_probable_member, p2alpha
from scripts.mpl_colormaps import stream_cmap1 as cmap1
from scripts.mpl_colormaps import stream_cmap2 as cmap2

if TYPE_CHECKING:
    from stream_ml.core import ModelAPI

# =============================================================================
# Setup

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

# isochrone data
with asdf.open(
    paths.data / "gd1" / "isochrone.asdf", "r", lazy_load=False, copy_arrays=True
) as af:
    isochrone_data = Data(**af["isochrone_data"])


# TODO: load control points
# # Load Control Points
# stream_cp = QTable.read(paths.data / "gd1" / "control_points_stream.ecsv")
# spur_cp = QTable.read(paths.data / "gd1" / "control_points_spur.ecsv")


# =============================================================================


def diagnostic_plot(model: ModelAPI, data: Data, where: Data) -> plt.Figure:
    """Plot the model."""
    # =============================================================================
    # Evaluate model

    with xp.no_grad():
        model.eval()
        mpars = model.unpack_params(model(data))
        model.train()

        stream_lnlik = model.component_ln_posterior("stream", mpars, data, where=where)
        spur_lnlik = model.component_ln_posterior("spur", mpars, data, where=where)
        bkg_lnlik = model.component_ln_posterior("background", mpars, data, where=where)
        # tot_lnlik = model.ln_posterior(mpars, data, where=where)  # FIXME
        tot_lnlik = xp.logsumexp(xp.stack((stream_lnlik, spur_lnlik, bkg_lnlik), 1), 1)

    # =============================================================================

    stream_weight = mpars[(f"stream.{WEIGHT_NAME}",)]
    stream_cutoff = stream_weight > -4

    spur_weight = mpars[(f"spur.{WEIGHT_NAME}",)]
    spur_cutoff = spur_weight > -5

    bkg_prob = xp.exp(bkg_lnlik - tot_lnlik)
    stream_prob = xp.exp(stream_lnlik - tot_lnlik)
    spur_prob = xp.exp(spur_lnlik - tot_lnlik)
    allstream_prob = xp.exp(xp.logaddexp(stream_lnlik, spur_lnlik) - tot_lnlik)

    psort = np.argsort(allstream_prob)

    # =============================================================================
    # Make Figure

    fig = plt.figure(constrained_layout="tight", figsize=(10, 16))
    gs = GridSpec(2, 1, figure=fig, height_ratios=(6, 5), hspace=0.12)
    gs0 = gs[0].subgridspec(8, 1, height_ratios=(1, 1, 5, 5, 5, 5, 5, 5))

    colors = color_by_probable_member(
        (stream_prob[psort], cmap1), (spur_prob[psort], cmap2)
    )
    alphas = p2alpha(allstream_prob[psort])

    # ---------------------------------------------------------------------------
    # Colormap

    # Stream probability
    ax00 = fig.add_subplot(gs0[0, :])
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap1),
        cax=ax00,
        orientation="horizontal",
        label="Stream Probability",
    )
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")

    # Spur probability
    ax01 = fig.add_subplot(gs0[1, :])
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap2),
        cax=ax01,
        orientation="horizontal",
        label="Spur Probability",
    )
    cbar.ax.xaxis.set_ticks([])
    cbar.ax.xaxis.set_label_position("bottom")

    # ---------------------------------------------------------------------------
    # Weight plot

    ax02 = fig.add_subplot(
        gs0[2, :], ylabel="Stream fraction", ylim=(0, 0.4), xticklabels=[]
    )

    with xp.no_grad():
        helper.manually_set_dropout(model, 0.15)
        _stream_weights = []
        _spur_weights = []
        for _ in range(25):
            _mpars = model.unpack_params(model(data))
            _stream_weights.append(_mpars[f"stream.{WEIGHT_NAME}",])
            _spur_weights.append(_mpars[f"spur.{WEIGHT_NAME}",])

        stream_weights = xp.stack(_stream_weights, 1)
        stream_weight_percentiles = np.c_[
            np.percentile(stream_weights, 5, axis=1),
            np.percentile(stream_weights, 95, axis=1),
        ]

        spur_weights = xp.stack(_spur_weights, 1)
        spur_weight_percentiles = np.c_[
            np.percentile(spur_weights, 5, axis=1),
            np.percentile(spur_weights, 95, axis=1),
        ]

        helper.manually_set_dropout(model, 0)

    f1 = ax02.fill_between(
        data["phi1"],
        stream_weight_percentiles[:, 0],
        stream_weight_percentiles[:, 1],
        color=cmap1(0.99),
        alpha=0.25,
    )
    (l1,) = ax02.plot(
        data["phi1"], stream_weights.mean(1), c="k", ls="--", lw=2, label="Model (MLE)"
    )
    f2 = ax02.fill_between(
        data["phi1"],
        spur_weight_percentiles[:, 0],
        spur_weight_percentiles[:, 1],
        color=cmap2(0.99),
        alpha=0.25,
    )
    (l2,) = ax02.plot(data["phi1"], spur_weights.mean(1), c="k", ls="--", lw=2)

    ax02.legend(
        [(f1, f2), l1],
        [r"Models (15% dropout)", l1.get_label()],
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="upper left",
    )

    # ---------------------------------------------------------------------------
    # Phi2

    mpa = mpars.get_prefixed("stream.astrometric")
    mpb = mpars.get_prefixed("spur.astrometric")

    ax03 = fig.add_subplot(
        gs0[3, :],
        xticklabels=[],
        ylabel=r"$\phi_2$ [$\degree$]",
        rasterization_zorder=0,
    )

    p1 = ax03.scatter(
        data["phi1"][psort],
        data["phi2"][psort],
        c=colors,
        alpha=alphas,
        s=2,
        zorder=-10,
    )
    f1 = ax03.fill_between(
        data["phi1"][stream_cutoff],
        (mpa["phi2", "mu"] - xp.exp(mpa["phi2", "ln-sigma"]))[stream_cutoff],
        (mpa["phi2", "mu"] + xp.exp(mpa["phi2", "ln-sigma"]))[stream_cutoff],
        color=cmap1(0.99),
        alpha=0.25,
    )
    f2 = ax03.fill_between(
        data["phi1"][spur_cutoff],
        (mpb["phi2", "mu"] - xp.exp(mpb["phi2", "ln-sigma"]))[spur_cutoff],
        (mpb["phi2", "mu"] + xp.exp(mpb["phi2", "ln-sigma"]))[spur_cutoff],
        color=cmap2(0.99),
        alpha=0.25,
    )
    ax03.legend(
        [p1, (f1, f2)],
        ["Data", r"Models"],
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="upper left",
    )

    # ---------------------------------------------------------------------------
    # Parallax

    ax04 = fig.add_subplot(
        gs0[4, :], ylabel=r"$\varpi$ [mas]", xticklabels=[], rasterization_zorder=0
    )

    p1 = ax04.scatter(
        data["phi1"][psort],
        data["plx"][psort],
        c=colors,
        alpha=alphas,
        s=2,
        zorder=-10,
    )
    f1 = ax04.fill_between(
        data["phi1"][stream_cutoff],
        (mpa["plx", "mu"] - xp.exp(mpa["plx", "ln-sigma"]))[stream_cutoff],
        (mpa["plx", "mu"] + xp.exp(mpa["plx", "ln-sigma"]))[stream_cutoff],
        color=cmap1(0.99),
        alpha=0.25,
    )
    f2 = ax04.fill_between(
        data["phi1"][spur_cutoff],
        (mpb["plx", "mu"] - xp.exp(mpb["plx", "ln-sigma"]))[spur_cutoff],
        (mpb["plx", "mu"] + xp.exp(mpb["plx", "ln-sigma"]))[spur_cutoff],
        color=cmap2(0.99),
        alpha=0.25,
    )
    ax04.legend(
        [p1, (f1, f2)],
        ["Data", r"Models"],
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="upper left",
    )

    # ---------------------------------------------------------------------------
    # PM-Phi1

    ax05 = fig.add_subplot(
        gs0[5, :],
        ylabel=r"$\mu_{\phi_1}^*$ [mas yr$^{-1}$]",
        rasterization_zorder=0,
        xticklabels=[],
    )

    p1 = ax05.scatter(
        data["phi1"][psort],
        data["pmphi1"][psort],
        c=colors,
        alpha=alphas,
        s=2,
        zorder=-10,
    )
    f1 = ax05.fill_between(
        data["phi1"][stream_cutoff],
        (mpa["pmphi1", "mu"] - xp.exp(mpa["pmphi1", "ln-sigma"]))[stream_cutoff],
        (mpa["pmphi1", "mu"] + xp.exp(mpa["pmphi1", "ln-sigma"]))[stream_cutoff],
        color=cmap1(0.99),
        alpha=0.25,
    )
    f2 = ax05.fill_between(
        data["phi1"][spur_cutoff],
        (mpb["pmphi1", "mu"] - xp.exp(mpb["pmphi1", "ln-sigma"]))[spur_cutoff],
        (mpb["pmphi1", "mu"] + xp.exp(mpb["pmphi1", "ln-sigma"]))[spur_cutoff],
        color=cmap2(0.99),
        alpha=0.25,
    )
    ax05.legend(
        [p1, (f1, f2)],
        ["Data", r"Models"],
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="upper left",
    )

    # ---------------------------------------------------------------------------
    # PM-Phi2

    ax06 = fig.add_subplot(gs0[6, :])
    ax06.set_xticklabels([])
    ax06.set(ylabel=r"$\mu_{\phi_2}$ [mas yr$^{-1}$]")
    ax06.set_rasterization_zorder(0)

    p1 = ax06.scatter(
        data["phi1"][psort],
        data["pmphi2"][psort],
        c=colors,
        alpha=alphas,
        s=2,
        zorder=-10,
    )
    f1 = ax06.fill_between(
        data["phi1"][stream_cutoff],
        (mpa["pmphi2", "mu"] - xp.exp(mpa["pmphi2", "ln-sigma"]))[stream_cutoff],
        (mpa["pmphi2", "mu"] + xp.exp(mpa["pmphi2", "ln-sigma"]))[stream_cutoff],
        color=cmap1(0.99),
        alpha=0.25,
        label="Model (MLE)",
    )
    f2 = ax06.fill_between(
        data["phi1"][spur_cutoff],
        (mpb["pmphi2", "mu"] - xp.exp(mpb["pmphi2", "ln-sigma"]))[spur_cutoff],
        (mpb["pmphi2", "mu"] + xp.exp(mpb["pmphi2", "ln-sigma"]))[spur_cutoff],
        color=cmap2(0.99),
        alpha=0.25,
    )
    ax06.legend(
        [p1, (f1, f2)],
        ["Data", r"Models"],
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="upper left",
    )

    # ---------------------------------------------------------------------------
    # Distance

    ax07 = fig.add_subplot(
        gs0[7, :], xlabel=r"$\phi_1$ [deg]", ylabel=r"$d$ [kpc]", ylim=(7, 11)
    )

    mpa = mpars["stream.photometric.distmod"]
    d2sm = Distance(distmod=(mpa["mu"] - 2 * xp.exp(mpa["ln-sigma"])) * u.mag)
    d2sp = Distance(distmod=(mpa["mu"] + 2 * xp.exp(mpa["ln-sigma"])) * u.mag)
    d1sm = Distance(distmod=(mpa["mu"] - xp.exp(mpa["ln-sigma"])) * u.mag)
    d1sp = Distance(distmod=(mpa["mu"] + xp.exp(mpa["ln-sigma"])) * u.mag)

    ax07.fill_between(
        data["phi1"][stream_cutoff],
        d2sm[stream_cutoff].to_value("kpc"),
        d2sp[stream_cutoff].to_value("kpc"),
        alpha=0.15,
        color=cmap1(0.99),
    )
    f1 = ax07.fill_between(
        data["phi1"][stream_cutoff],
        d1sm[stream_cutoff].to_value("kpc"),
        d1sp[stream_cutoff].to_value("kpc"),
        alpha=0.25,
        color=cmap1(0.99),
    )

    mpb = mpars["spur.photometric.distmod"]
    d2sm = Distance(distmod=(mpb["mu"] - 2 * xp.exp(mpb["ln-sigma"])) * u.mag)
    d2sp = Distance(distmod=(mpb["mu"] + 2 * xp.exp(mpb["ln-sigma"])) * u.mag)
    d1sm = Distance(distmod=(mpb["mu"] - xp.exp(mpb["ln-sigma"])) * u.mag)
    d1sp = Distance(distmod=(mpb["mu"] + xp.exp(mpb["ln-sigma"])) * u.mag)

    ax07.fill_between(
        data["phi1"][spur_cutoff],
        d2sm[spur_cutoff].to_value("kpc"),
        d2sp[spur_cutoff].to_value("kpc"),
        alpha=0.15,
        color=cmap2(0.99),
    )
    f2 = ax07.fill_between(
        data["phi1"][spur_cutoff],
        d1sm[spur_cutoff].to_value("kpc"),
        d1sp[spur_cutoff].to_value("kpc"),
        alpha=0.25,
        color=cmap2(0.99),
    )

    ax07.legend(
        [(f1, f2)],
        [r"Models"],
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="upper left",
    )

    # =============================================================================
    # Slice plots

    # Legend
    legend1 = plt.legend(
        handles=[
            mpl.patches.Patch(color=cmap1(0.01), label="Background (MLE)"),
            mpl.lines.Line2D(
                [0], [0], color="k", lw=3, ls="-", label="Background Distribution"
            ),
            mpl.patches.Patch(color=cmap1(0.99), label="Stream (MLE)"),
            mpl.patches.Patch(color=cmap2(0.99), label="Spur (MLE)"),
        ],
        ncols=4,
        loc="upper right",
        bbox_to_anchor=(1, -0.45),
    )
    ax07.add_artist(legend1)

    # gridspec
    gs1 = gs[1].subgridspec(5, 4, height_ratios=(1, 1, 1, 1, 2), hspace=0)

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

        ax10i = fig.add_subplot(gs1[0, i], xlabel=r"$\phi_2$ [$\degree$]")

        # Connect to top plot(s)
        for ax in (ax01, ax02, ax03, ax04, ax05, ax06, ax07):
            ax.axvline(bins[i], color="gray", ls="--", zorder=-200)
            ax.axvline(bins[i + 1], color="gray", ls="--", zorder=-200)
        smlvis._slices.connect_slices_to_top(  # noqa: SLF001
            fig, ax07, ax10i, left=bins[i], right=bins[i + 1], color="gray"
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
            label=["", "Stream Model (MLE)", "Spur Model (MLE)"],
        )

        xmin, xmax = data["phi2"].min().numpy(), data["phi2"].max().numpy()
        x = np.linspace(xmin, xmax)
        bkg_wgt = mpars[f"background.{WEIGHT_NAME}",][sel].mean()
        m = mpars["background.astrometric.phi2pmphi1.phi2", "slope"][sel].mean()
        ax10i.plot(x, bkg_wgt * exp_distr(m, xmin, xmax).pdf(x), c="k")

        if i == 0:
            ax10i.set_ylabel("frequency")

        # ---------------------------------------------------------------------------
        # Parallax

        ax11i = fig.add_subplot(gs1[1, i], xlabel=r"$\phi_2$ [$\degree$]")

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
            label=["", "Stream Model (MLE)", "Spur Model (MLE)"],
        )

        if i == 0:
            ax11i.set_ylabel("frequency")

        # ---------------------------------------------------------------------------
        # PM-Phi1

        ax12i = fig.add_subplot(gs1[2, i], xlabel=r"$\mu_{\phi_1}^*$ [mas yr$^{-1}$]")

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
            label=["", "Stream Model (MLE)", "Spur Model (MLE)"],
        )

        xmin, xmax = data["pmphi1"].min().numpy(), data["pmphi1"].max().numpy()
        x = np.linspace(xmin, xmax)
        m = mpars["background.astrometric.phi2pmphi1.pmphi1", "slope"][sel].mean()
        ax12i.plot(x, bkg_wgt * exp_distr(m, xmin, xmax).pdf(x), c="k")

        if i == 0:
            ax12i.set_ylabel("frequency")

        # ---------------------------------------------------------------------------
        # PM-Phi2

        ax13i = fig.add_subplot(gs1[3, i], xlabel=r"$\mu_{phi_2}$ [mas yr$^{-1}$]")
        ax13i.hist(
            np.ones((sel.sum(), 3)) * data_["pmphi2"][:, None].numpy(),
            bins=50,
            weights=np.stack((bkg_prob_, stream_prob_, spur_prob_), axis=1),
            color=[cmap1(0.01), cmap1(0.99), cmap2(0.99)],
            alpha=0.75,
            density=True,
            stacked=True,
            label=["", "Stream Model (MLE)", "Spur Model (MLE)"],
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

        sorter = np.argsort(allstream_prob[sel])
        ax14i.scatter(
            data_["g"][sorter] - data_["r"][sorter],
            data_["g"][sorter],
            c=colors[sel][sorter],
            s=1,
        )
        # isochrone
        ax14i.plot(
            isochrone_data["g"] - isochrone_data["r"],
            isochrone_data["g"]
            + mpars["stream.photometric.distmod", "mu"][sel].mean().numpy(),
            c="green",
        )

        if i == 0:
            ax14i.set_ylabel("g [mag]")
        else:
            ax14i.set_yticklabels([])

    return fig
