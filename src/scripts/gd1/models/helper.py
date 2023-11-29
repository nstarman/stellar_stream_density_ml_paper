"""GD-1 model helper functions."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import asdf
import astropy.units as u
import galstreams
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch as xp
from astropy.coordinates import Distance
from astropy.table import QTable
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerTuple
from scipy.interpolate import InterpolatedUnivariateSpline
from showyourwork.paths import user as user_paths

import stream_mapper.visualization as smlvis
from stream_mapper.core import WEIGHT_NAME, Data

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.gd1.frames import gd1_frame
from scripts.helper import (
    color_by_probable_member,
    manually_set_dropout,
    p2alpha,
)
from scripts.mpl_colormaps import stream_cmap1 as cmap_stream
from scripts.mpl_colormaps import stream_cmap2 as cmap_spur

if TYPE_CHECKING:
    from stream_mapper.core import ModelAPI

# =============================================================================
# Setup

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

# isochrone data
with asdf.open(
    paths.data / "gd1" / "isochrone.asdf", "r", lazy_load=False, copy_arrays=True
) as af:
    isochrone_data = Data(**af["isochrone_data"])

# Control points
distance_cp = QTable.read(paths.data / "gd1" / "control_points_distance.ecsv")

mws = galstreams.MWStreams()
gd1 = mws["GD-1-I21"]
gd1_sc = gd1.track.transform_to(gd1_frame)[::100]

spline_pmphi1 = InterpolatedUnivariateSpline(
    gd1_sc.phi1.value, gd1_sc.pm_phi1_cosphi2.value
)
spline_pmphi2 = InterpolatedUnivariateSpline(gd1_sc.phi1.value, gd1_sc.pm_phi2.value)


# =============================================================================


def diagnostic_plot(  # noqa: C901, PLR0912
    model: ModelAPI, data: Data, where: Data, cutoff: float = -7
) -> plt.Figure:
    """Plot the model."""
    # =============================================================================
    # Evaluate model

    with xp.no_grad():
        # turn dropout off
        model = model.eval()
        manually_set_dropout(model, 0)

        # evaluate the model
        mpars = model.unpack_params(model(data))

        # evaluate the likelihoods
        stream_lnlik = model.component_ln_posterior("stream", mpars, data, where=where)
        spur_lnlik = model.component_ln_posterior("spur", mpars, data, where=where)
        bkg_lnlik = model.component_ln_posterior("background", mpars, data, where=where)

        # total likelihood
        tot_lnlik = xp.logsumexp(xp.stack((stream_lnlik, spur_lnlik, bkg_lnlik), 1), 1)

        # turn dropout off
        manually_set_dropout(model, 0)
        model = model.eval()

    # =============================================================================

    stream_cutoff = mpars[(f"stream.{WEIGHT_NAME}",)] > cutoff
    spur_cutoff = mpars[(f"spur.{WEIGHT_NAME}",)] > cutoff

    bkg_prob = xp.exp(bkg_lnlik - tot_lnlik)
    stream_prob = xp.exp(stream_lnlik - tot_lnlik)
    spur_prob = xp.exp(spur_lnlik - tot_lnlik)
    allstream_prob = xp.exp(
        xp.logsumexp(xp.stack((stream_lnlik, spur_lnlik), 1), 1) - tot_lnlik
    )

    psort = np.argsort(allstream_prob)

    # =============================================================================
    # Make Figure

    fig = plt.figure(constrained_layout="tight", figsize=(10, 16))
    gs = GridSpec(2, 1, figure=fig, height_ratios=(6, 5), hspace=0.12)
    gs0 = gs[0].subgridspec(8, 1, height_ratios=(1, 1, 5, 5, 5, 5, 5, 5))

    colors = color_by_probable_member(
        (stream_prob[psort], cmap_stream), (spur_prob[psort], cmap_spur)
    )
    alphas = p2alpha(allstream_prob[psort], minval=0.1)

    # ---------------------------------------------------------------------------
    # Colormap

    # Stream probability
    ax00 = fig.add_subplot(gs0[0, :])
    cbar = fig.colorbar(
        ScalarMappable(cmap=cmap_stream), cax=ax00, orientation="horizontal"
    )
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.text(0.5, 0.5, "Stream Probability", ha="center", va="center", fontsize=13)

    # Spur probability
    ax01 = fig.add_subplot(gs0[1, :])
    cbar = fig.colorbar(
        ScalarMappable(cmap=cmap_spur), cax=ax01, orientation="horizontal"
    )
    cbar.ax.xaxis.set_ticks([])
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.ax.text(0.5, 0.5, "Spur Probability", ha="center", va="center", fontsize=13)

    # ---------------------------------------------------------------------------
    # Weight plot

    ax02 = fig.add_subplot(gs0[2, :], ylabel=r"$\ln f_{\rm stream}$", xticklabels=[])

    (l1,) = ax02.plot(
        data["phi1"],
        mpars["stream.ln-weight",],
        c=cmap_stream(0.99),
        ls="--",
        lw=2,
    )
    (l2,) = ax02.plot(
        data["phi1"], mpars["spur.ln-weight",], c=cmap_spur(0.99), ls="--", lw=2
    )

    ax02.legend(
        [l1, l2],
        ["Models"],
        numpoints=1,
        handler_map={list: HandlerTuple(ndivide=None)},
        loc="upper left",
    )

    # ---------------------------------------------------------------------------
    # Phi2

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

    stream_cp = model["stream"]["astrometric"].priors[0].center
    stream_cp_w = model["stream"]["astrometric"].priors[0].width
    ax03.errorbar(
        stream_cp["phi1"],
        stream_cp["phi2"],
        yerr=stream_cp_w["phi2"],
        fmt=".",
        c=cmap_stream(0.99),
        capsize=2,
        zorder=-5,
    )
    spur_cp = model["spur"]["astrometric"].priors[0].center
    spur_cp_w = model["spur"]["astrometric"].priors[0].width
    ax03.errorbar(
        spur_cp["phi1"],
        spur_cp["phi2"],
        yerr=spur_cp_w["phi2"],
        fmt=".",
        c=cmap_spur(0.99),
        capsize=2,
        zorder=-5,
    )

    handles = []
    for k, cutoff_, cmap in (
        ("stream", stream_cutoff, cmap_stream),
        ("spur", spur_cutoff, cmap_spur),
    ):
        mp = mpars.get_prefixed(f"{k}.astrometric")
        f1 = ax03.fill_between(
            data["phi1"],
            (mp["phi2", "mu"] - xp.exp(mp["phi2", "ln-sigma"])),
            (mp["phi2", "mu"] + xp.exp(mp["phi2", "ln-sigma"])),
            color=cmap(0.99),
            alpha=0.25,
            where=cutoff_,
            zorder=-5,
        )
        handles.append(f1)

    ax03.legend(
        [p1, handles],
        ["Data", r"Models"],
        numpoints=1,
        handler_map={list: HandlerTuple(ndivide=None)},
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

    # Control points
    ax04.errorbar(
        distance_cp["phi1"],
        distance_cp["parallax"],
        yerr=distance_cp["w_parallax"],
        fmt=".",
        c=cmap_stream(0.99),
        capsize=2,
        zorder=-5,
    )

    for k, cutoff_, cmap in (
        ("stream", stream_cutoff, cmap_stream),
        ("spur", spur_cutoff, cmap_spur),
    ):
        mp = mpars.get_prefixed(f"{k}.astrometric")
        ax04.fill_between(
            data["phi1"],
            (mp["plx", "mu"] - xp.exp(mp["plx", "ln-sigma"])),
            (mp["plx", "mu"] + xp.exp(mp["plx", "ln-sigma"])),
            color=cmap(0.99),
            alpha=0.25,
            where=cutoff_,
        )

    ax04.set_ylim(data["plx"].min().numpy(), data["plx"].max().numpy())

    # ---------------------------------------------------------------------------
    # PM-Phi1

    ax05 = fig.add_subplot(
        gs0[5, :],
        ylabel=r"$\mu_{\phi_1}^*$ - GD1 [mas yr$^{-1}$]",
        rasterization_zorder=0,
        xticklabels=[],
    )

    p1 = ax05.scatter(
        data["phi1"][psort],
        data["pmphi1"][psort] - spline_pmphi1(data["phi1"][psort]),
        c=colors,
        alpha=alphas,
        s=2,
        zorder=-10,
    )

    # Control points
    ax05.errorbar(
        stream_cp["phi1"],
        stream_cp["pmphi1"] - spline_pmphi1(stream_cp["phi1"]),
        yerr=stream_cp_w["pmphi1"],
        fmt=".",
        c=cmap_stream(0.99),
        capsize=2,
        zorder=-5,
    )
    ax05.errorbar(
        spur_cp["phi1"],
        spur_cp["pmphi1"] - spline_pmphi1(spur_cp["phi1"]),
        yerr=spur_cp_w["pmphi1"],
        fmt=".",
        c=cmap_spur(0.99),
        capsize=2,
        zorder=-5,
    )

    for k, cutoff_, cmap in (
        ("stream", stream_cutoff, cmap_stream),
        ("spur", spur_cutoff, cmap_spur),
    ):
        mp = mpars.get_prefixed(f"{k}.astrometric")
        ax05.fill_between(
            data["phi1"],
            (
                mp["pmphi1", "mu"]
                - spline_pmphi1(data["phi1"])
                - xp.exp(mp["pmphi1", "ln-sigma"])
            ),
            (
                mp["pmphi1", "mu"]
                - spline_pmphi1(data["phi1"])
                + xp.exp(mp["pmphi1", "ln-sigma"])
            ),
            color=cmap(0.99),
            alpha=0.25,
            zorder=-2,
            where=cutoff_,
        )

    # ---------------------------------------------------------------------------
    # PM-Phi2

    ax06 = fig.add_subplot(
        gs0[6, :],
        xticklabels=[],
        ylabel=r"$\mu_{\phi_2}$ - GD1 [mas yr$^{-1}$]",
        rasterization_zorder=0,
    )

    p1 = ax06.scatter(
        data["phi1"][psort],
        data["pmphi2"][psort] - spline_pmphi2(data["phi1"][psort]),
        c=colors,
        alpha=alphas,
        s=2,
        zorder=-10,
    )

    for k, cutoff_, cmap in (
        ("stream", stream_cutoff, cmap_stream),
        ("spur", spur_cutoff, cmap_spur),
    ):
        mp = mpars.get_prefixed(f"{k}.astrometric")
        ax06.fill_between(
            data["phi1"],
            (
                mp["pmphi2", "mu"]
                - spline_pmphi2(data["phi1"])
                - xp.exp(mp["pmphi2", "ln-sigma"])
            ),
            (
                mp["pmphi2", "mu"]
                - spline_pmphi2(data["phi1"])
                + xp.exp(mp["pmphi2", "ln-sigma"])
            ),
            color=cmap(0.99),
            alpha=0.25,
            where=cutoff_,
            zorder=-2,
        )

    # ---------------------------------------------------------------------------
    # Distance

    ax07 = fig.add_subplot(
        gs0[7, :], xlabel=r"$\phi_1$ [deg]", ylabel=r"$d$ [kpc]", ylim=(7, 11)
    )

    # Control points
    ax07.errorbar(
        distance_cp["phi1"],
        Distance(distmod=distance_cp["distmod"]).to_value("kpc"),
        yerr=(
            Distance(distmod=distance_cp["distmod"] + distance_cp["w_distmod"])
            - Distance(distmod=distance_cp["distmod"] - distance_cp["w_distmod"])
        ).to_value("kpc")
        / 2,
        fmt=".",
        c=cmap_stream(0.99),
        capsize=2,
        zorder=-5,
    )

    for k, cutoff_, cmap in (
        ("stream", stream_cutoff, cmap_stream),
        ("spur", spur_cutoff, cmap_spur),
    ):
        mp = mpars[f"{k}.photometric.distmod"]

        d2sm = Distance(distmod=(mp["mu"] - 2 * xp.exp(mp["ln-sigma"])) * u.mag)
        d2sp = Distance(distmod=(mp["mu"] + 2 * xp.exp(mp["ln-sigma"])) * u.mag)
        d1sm = Distance(distmod=(mp["mu"] - xp.exp(mp["ln-sigma"])) * u.mag)
        d1sp = Distance(distmod=(mp["mu"] + xp.exp(mp["ln-sigma"])) * u.mag)

        ax07.fill_between(
            data["phi1"],
            d2sm.to_value("kpc"),
            d2sp.to_value("kpc"),
            alpha=0.15,
            color=cmap(0.99),
            where=cutoff_,
        )
        ax07.fill_between(
            data["phi1"],
            d1sm.to_value("kpc"),
            d1sp.to_value("kpc"),
            alpha=0.25,
            color=cmap(0.99),
            where=cutoff_,
        )

    # =============================================================================
    # Slice plots

    # Legend
    legend1 = plt.legend(
        handles=[
            mpl.patches.Patch(color=cmap_stream(0.01), label="Background"),
            mpl.patches.Patch(color=cmap_stream(0.99), label="Stream"),
            mpl.patches.Patch(color=cmap_spur(0.99), label="Spur"),
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
        allstream_prob_ = allstream_prob[sel]

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
            color=[cmap_stream(0.01), cmap_stream(0.99), cmap_spur(0.99)],
            alpha=0.75,
            density=True,
            stacked=True,
            label=["", "Stream Model", "Spur Model"],
        )

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
            color=[cmap_stream(0.01), cmap_stream(0.99), cmap_spur(0.99)],
            alpha=0.75,
            density=True,
            stacked=True,
            label=["", "Stream Model", "Spur Model"],
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
            color=[cmap_stream(0.01), cmap_stream(0.99), cmap_spur(0.99)],
            alpha=0.75,
            density=True,
            stacked=True,
            label=["", "Stream Model", "Spur Model"],
        )

        if i == 0:
            ax12i.set_ylabel("frequency")

        # ---------------------------------------------------------------------------
        # PM-Phi2

        ax13i = fig.add_subplot(gs1[3, i], xlabel=r"$\mu_{phi_2}$ [mas yr$^{-1}$]")
        ax13i.hist(
            np.ones((sel.sum(), 3)) * data_["pmphi2"][:, None].numpy(),
            bins=50,
            weights=np.stack((bkg_prob_, stream_prob_, spur_prob_), axis=1),
            color=[cmap_stream(0.01), cmap_stream(0.99), cmap_spur(0.99)],
            alpha=0.75,
            density=True,
            stacked=True,
            label=["", "Stream Model", "Spur Model"],
        )

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

        sorter = np.argsort(allstream_prob_)
        ax14i.scatter(
            data_["g"][sorter] - data_["r"][sorter],
            data_["g"][sorter],
            c=color_by_probable_member(
                (stream_prob_, cmap_stream), (spur_prob_, cmap_spur)
            )[sorter],
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
