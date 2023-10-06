"""Plot the trained pal5 model."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

import galstreams
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch as xp
from astropy.table import QTable
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
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
model.load_state_dict(xp.load(paths.data / "pal5" / "model" / "model_12499.pt"))
model = model.eval()

# =============================================================================


def recursive_iterate(
    dmpars: list[sml.params.Params[str, Any]],
    structure: dict[str, Any],
    _prefix: str = "",
) -> dict[str, Any]:
    """Recursively iterate and compute the mean of each parameter."""
    out = dict[str, Any]()
    _prefix = _prefix.lstrip(".")
    for k, v in structure.items():
        if isinstance(v, Mapping):
            out[k] = recursive_iterate(dmpars, v, _prefix=f"{_prefix}.{k}")
            continue

        key: tuple[str] | tuple[str, str] = (f"{_prefix}", k) if _prefix else (k,)
        out[k] = xp.stack([mp[key] for mp in dmpars], 1).mean(1)

    return out


# Also evaluate the model with dropout on
with xp.no_grad():
    # turn dropout on
    model = model.train()
    manually_set_dropout(model, 0.15)
    # evaluate the model
    dmpars = [model.unpack_params(model(data)) for i in range(100)]
    # Mpars
    mpars = sml.params.Params(recursive_iterate(dmpars, dmpars[0]))
    # weights
    stream_weights = xp.stack([mp["stream.weight",] for mp in dmpars], 1)
    stream_weight_percentiles = np.c_[
        np.percentile(stream_weights, 5, axis=1),
        np.percentile(stream_weights, 95, axis=1),
    ]

    # Likelihoods
    stream_lnlik = model.component_ln_posterior("stream", mpars, data, where=where)
    bkg_lnlik = model.component_ln_posterior("background", mpars, data, where=where)
    # tot_lnlik = model.ln_posterior(mpars, data, where=where)  # FIXME
    tot_lnlik = xp.logaddexp(stream_lnlik, bkg_lnlik)

    # turn dropout back off
    manually_set_dropout(model, 0)
    model = model.eval()


# Evaluate model
with xp.no_grad():
    manually_set_dropout(model, 0)
    model = model.eval()

    mpars = model.unpack_params(model(data))

    stream_lnlik = model.component_ln_posterior("stream", mpars, data, where=where)
    bkg_lnlik = model.component_ln_posterior("background", mpars, data, where=where)
    # tot_lnlik = model.ln_posterior(mpars, data, where=where)
    tot_lnlik = xp.logaddexp(stream_lnlik, bkg_lnlik)


stream_weight = mpars[("stream.weight",)]
stream_cutoff = stream_weight > 1e-4  # everything has weight > 0

bkg_prob = xp.exp(bkg_lnlik - tot_lnlik)
stream_prob = xp.exp(stream_lnlik - tot_lnlik)
allstream_prob = stream_prob

# print(allstream_prob.max(), allstream_prob.min())

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

# Upper and lower bounds
_bounds_kw = {"c": "gray", "ls": "-", "lw": 2, "alpha": 0.8}
ax01.axhline(model.params[("stream.weight",)].bounds.lower[0], **_bounds_kw)
ax01.axhline(model.params[("stream.weight",)].bounds.upper[0], **_bounds_kw)

# 15% dropout
f1 = ax01.fill_between(
    data["phi1"],
    stream_weight_percentiles[:, 0],
    stream_weight_percentiles[:, 1],
    color=cmap1(0.99),
    alpha=0.25,
)
# Mean
(l1,) = ax01.plot(
    data["phi1"], stream_weights.mean(1), c="k", ls="--", lw=2, label="Model"
)

ax01.legend(
    [f1, l1],
    [r"Models (15% dropout)", l1.get_label()],
    numpoints=1,
    loc="upper left",
)

# ---------------------------------------------------------------------------
# Phi2

ax02 = fig.add_subplot(
    gs[2, :],
    xlabel="",
    ylabel=r"$\phi_2$ [$\degree$]",
    xlim=xlims,
    ylim=(data["phi2"].min(), data["phi2"].max()),
    xticklabels=[],
    rasterization_zorder=0,
)
mpa = mpars.get_prefixed("stream.astrometric")

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
    data["phi1"],
    (mpa["phi2", "mu"] - xp.exp(mpa["phi2", "ln-sigma"])),
    (mpa["phi2", "mu"] + xp.exp(mpa["phi2", "ln-sigma"])),
    color=cmap1(0.99),
    alpha=0.25,
    where=stream_cutoff,
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

legend_elements_data = (
    Line2D([0], [0], marker="o", markerfacecolor=cmap1(0.01), markersize=10),
    Line2D([0], [0], marker="o", markerfacecolor=cmap1(0.99), markersize=10),
)

ax02.legend(
    [legend_elements_data, p1, f1],
    ["Data", "Control points", "Models"],
    numpoints=1,
    loc="upper left",
    handler_map={tuple: HandlerTuple(ndivide=None)},
)


# ---------------------------------------------------------------------------
# Parallax

ax03 = fig.add_subplot(
    gs[3, :],
    xlabel="",
    ylabel=r"$\varpi$ [mas]",
    xlim=xlims,
    ylim=(max(data["plx"].min(), -1), data["plx"].max()),
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
