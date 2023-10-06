"""Plot the trained pal5 model."""

from __future__ import annotations

import sys

import galstreams
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch as xp
from astropy.table import QTable
from matplotlib.gridspec import GridSpec
from showyourwork.paths import user as user_paths

import stream_ml.pytorch as sml

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.helper import manually_set_dropout, p2alpha
from scripts.mpl_colormaps import stream_cmap1 as cmap1
from scripts.pal5.datasets import data, masks, where
from scripts.pal5.define_model import model
from scripts.pal5.frames import pal5_frame as frame

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

# =============================================================================
# Load data

# galstreams
allstreams = galstreams.MWStreams(implement_Off=True)
pal5I21 = allstreams["Pal5-I21"].track.transform_to(frame)
pal5PW19 = allstreams["Pal5-PW19"].track.transform_to(frame)

# Control points
stream_cp = QTable.read(paths.data / "pal5" / "control_points_stream.ecsv")

# Progenitor
progenitor_prob = np.zeros(len(masks))
progenitor_prob[~masks["Pal5"]] = 1

# Load model
model.load_state_dict(xp.load(paths.data / "pal5" / "model.pt"))
model = model.eval()

# =============================================================================

# Evaluate model
with xp.no_grad():
    model.eval()
    mpars = model.unpack_params(model(data))

    stream_lnlik = model.component_ln_posterior("stream", mpars, data, where=where)
    bkg_lnlik = model.component_ln_posterior("background", mpars, data, where=where)
    # tot_lnlik = model.ln_posterior(mpars, data, where=where)  # FIXME
    tot_lnlik = xp.logaddexp(stream_lnlik, bkg_lnlik)


# Also evaluate the model with dropout on
def _iter_mpars(key: str, subkey: str | None) -> np.ndarray:
    fullkey = (key,) if subkey is None else (key, subkey)
    return xp.stack([mp[fullkey] for mp in dmpars], 1).mean(1)


with xp.no_grad():
    # turn dropout on
    model = model.train()
    manually_set_dropout(model, 0.15)
    # evaluate the model
    dmpars = [model.unpack_params(model(data)) for i in range(100)]
    # Mpars
    mpars = sml.params.Params(
        {
            f"{k}.weight": _iter_mpars(f"{k}.weight", None)
            for k in ["stream", "stream.astrometric"]
        }
        | {
            f"stream.astrometric.{k}": {
                kk: _iter_mpars(f"stream.astrometric.{k}", kk)
                for kk in ["mu", "ln-sigma"]
            }
            for k in ["phi2", "pmphi1", "pmphi2"]
        }
    )
    # weights
    stream_weights = xp.stack([mp["stream.weight",] for mp in dmpars], 1)
    stream_weight_percentiles = np.c_[
        np.percentile(stream_weights, 5, axis=1),
        np.percentile(stream_weights, 95, axis=1),
    ]

    # turn dropout back off
    manually_set_dropout(model, 0)
    model = model.eval()

stream_weight = mpars[("stream.weight",)]
stream_cutoff = stream_weight > 0  # everything has weight > 0

bkg_prob = xp.exp(bkg_lnlik - tot_lnlik)
stream_prob = xp.exp(stream_lnlik - tot_lnlik)
allstream_prob = stream_prob

psort = np.argsort(allstream_prob)

# =============================================================================
# Make Figure

fig = plt.figure(constrained_layout="tight", figsize=(11, 10))
gs = GridSpec(6, 1, figure=fig, height_ratios=(1, 3, 5, 5, 5, 5))

colors = cmap1(stream_prob[psort])
alphas = p2alpha(allstream_prob[psort])

xlims = (data["phi1"].min(), data["phi1"].max())

# ---------------------------------------------------------------------------
# Colormap

# Stream probability
ax00 = fig.add_subplot(gs[0, :])
cbar = fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap1), cax=ax00, orientation="horizontal"
)
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")
cbar.ax.text(0.5, 0.5, "Stream Probability", ha="center", va="center", fontsize=14)


# ---------------------------------------------------------------------------
# Weight plot

ax01 = fig.add_subplot(
    gs[1, :],
    ylabel="Stream fraction",
    xlim=xlims,
    ylim=(1e-4, 1),
    xticklabels=[],
    yscale="log",
    rasterization_zorder=0,
)

f1 = ax01.fill_between(
    data["phi1"],
    stream_weight_percentiles[:, 0],
    stream_weight_percentiles[:, 1],
    color=cmap1(0.99),
    alpha=0.25,
)
(l1,) = ax01.plot(
    data["phi1"], stream_weights.mean(1), c="k", ls="--", lw=2, label="Model (MLE)"
)

ax01.legend(
    [f1, l1],
    [r"Models (15% dropout)", l1.get_label()],
    numpoints=1,
    loc="upper left",
)

# ---------------------------------------------------------------------------
# Phi2

mpa = mpars.get_prefixed("stream.astrometric")

ax02 = fig.add_subplot(
    gs[2, :],
    xlabel="",
    ylabel=r"$\phi_2$ [$\degree$]",
    xlim=xlims,
    ylim=(data["phi2"].min(), data["phi2"].max()),
    xticklabels=[],
    rasterization_zorder=0,
)

# Stream control points
ax02.errorbar(
    stream_cp["phi1"],
    stream_cp["phi2"],
    yerr=stream_cp["w_phi2"],
    fmt="o",
    color="k",
    capthick=3,
    elinewidth=3,
    capsize=3,
    zorder=-21,
)
p1 = ax02.errorbar(
    stream_cp["phi1"],
    stream_cp["phi2"],
    yerr=stream_cp["w_phi2"],
    fmt=".",
    c=cmap1(0.99),
    capsize=2,
    zorder=-20,
    label="Stream Control Points",
)

# Data
d1 = ax02.scatter(
    data["phi1"][psort], data["phi2"][psort], c=colors, alpha=alphas, s=2, zorder=-10
)
# Model
f1 = ax02.fill_between(
    data["phi1"][stream_cutoff],
    (mpa["phi2", "mu"] - xp.exp(mpa["phi2", "ln-sigma"]))[stream_cutoff],
    (mpa["phi2", "mu"] + xp.exp(mpa["phi2", "ln-sigma"]))[stream_cutoff],
    color=cmap1(0.99),
    alpha=0.25,
)

ax02.legend(
    [d1, p1, f1], ["Data", "Control points", "Models"], numpoints=1, loc="upper left"
)


# ---------------------------------------------------------------------------
# Parallax

ax03 = fig.add_subplot(
    gs[3, :],
    xlabel="",
    ylabel=r"$\varpi$ [mas]",
    xlim=xlims,
    ylim=(max(data["plx"].min(), -2.5), data["plx"].max()),
    xticklabels=[],
    rasterization_zorder=0,
)

# Data
d1 = ax03.scatter(
    data["phi1"][psort], data["plx"][psort], c=colors, alpha=alphas, s=2, zorder=-10
)


# ---------------------------------------------------------------------------
# PM-Phi1

ax04 = fig.add_subplot(
    gs[4, :],
    xlabel="",
    ylabel=r"$\mu_{\phi_1}^*$ [mas yr$^{-1}$]",
    xlim=xlims,
    ylim=(data["pmphi1"].min(), data["pmphi1"].max()),
    xticklabels=[],
    rasterization_zorder=0,
)

# Data
d1 = ax04.scatter(
    data["phi1"][psort], data["pmphi1"][psort], c=colors, alpha=alphas, s=2, zorder=-10
)
# Model
ax04.fill_between(
    data["phi1"][stream_cutoff],
    (mpa["pmphi1", "mu"] - xp.exp(mpa["pmphi1", "ln-sigma"]))[stream_cutoff],
    (mpa["pmphi1", "mu"] + xp.exp(mpa["pmphi1", "ln-sigma"]))[stream_cutoff],
    color=cmap1(0.99),
    alpha=0.25,
)


# ---------------------------------------------------------------------------
# PM-Phi2

ax05 = fig.add_subplot(
    gs[5, :],
    ylabel=r"$\mu_{\phi_2}$ [mas yr$^{-1}$]",
    xlim=xlims,
    rasterization_zorder=0,
    xticklabels=[],
)

# Data
ax05.scatter(
    data["phi1"][psort], data["pmphi2"][psort], c=colors, alpha=alphas, s=2, zorder=-10
)
# Model
ax05.fill_between(
    data["phi1"][stream_cutoff],
    (mpa["pmphi2", "mu"] - xp.exp(mpa["pmphi2", "ln-sigma"]))[stream_cutoff],
    (mpa["pmphi2", "mu"] + xp.exp(mpa["pmphi2", "ln-sigma"]))[stream_cutoff],
    color=cmap1(0.99),
    alpha=0.25,
)

fig.savefig(paths.figures / "pal5" / "results_full.pdf")
