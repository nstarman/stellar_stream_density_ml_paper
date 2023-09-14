"""Plot the trained GD1 model."""

from __future__ import annotations

import sys

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch as xp
from astropy.coordinates import Distance
from astropy.table import QTable
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerTuple
from showyourwork.paths import user as user_paths

import stream_ml.pytorch as sml

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.as_posix())
# isort: split

from scripts.gd1.datasets import data, where
from scripts.gd1.define_model import model
from scripts.gd1.model.helper import color_by_probable_member, p2alpha
from scripts.helper import manually_set_dropout
from scripts.mpl_colormaps import stream_cmap1 as cmap1
from scripts.mpl_colormaps import stream_cmap2 as cmap2

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

# =============================================================================

# Load control points
stream_cp = QTable.read(paths.data / "gd1" / "control_points_stream.ecsv")
spur_cp = QTable.read(paths.data / "gd1" / "control_points_spur.ecsv")
distance_cp = QTable.read(paths.data / "gd1" / "control_points_distance.ecsv")

# Load model
model.load_state_dict(xp.load(paths.data / "gd1" / "model.pt"))
model = model.eval()

# Evaluate model
with xp.no_grad():
    model.eval()
    mpars = model.unpack_params(model(data))

    stream_lnlik = model.component_ln_posterior("stream", mpars, data, where=where)
    spur_lnlik = model.component_ln_posterior("spur", mpars, data, where=where)
    bkg_lnlik = model.component_ln_posterior("background", mpars, data, where=where)
    # tot_lnlik = model.ln_posterior(mpars, data, where=where)  # FIXME
    tot_lnlik = xp.logsumexp(xp.stack((stream_lnlik, spur_lnlik, bkg_lnlik), 1), 1)


def _iter_mpars(key: str, subkey: str | None) -> np.ndarray:
    fullkey = (key,) if subkey is None else (key, subkey)
    return xp.stack([mp[fullkey] for mp in dmpars], 1).mean(1)


# Also evaluate the model with dropout on
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
            for k in ["stream", "stream.astrometric", "spur", "spur.astrometric"]
        }
        | {
            k: v
            for d in (
                {
                    f"{comp}.astrometric.{k}": {
                        **{
                            kk: _iter_mpars(f"{comp}.astrometric.{k}", kk)
                            for kk in ["mu", "ln-sigma"]
                        }
                    }
                    for k in ["phi2", "plx", "pmphi1", "pmphi2"]
                }
                | {
                    f"{comp}.photometric.{k}": {
                        **{
                            kk: _iter_mpars(f"{comp}.photometric.{k}", kk)
                            for kk in ["mu", "ln-sigma"]
                        }
                    }
                    for k in ["distmod"]
                }
                for comp in ["stream", "spur"]
            )
            for k, v in d.items()
        }
    )
    # weights
    stream_weights = xp.stack([mp["stream.weight",] for mp in dmpars], 1)
    stream_weight_percentiles = np.c_[
        np.percentile(stream_weights, 5, axis=1),
        np.percentile(stream_weights, 95, axis=1),
    ]
    spur_weights = xp.stack([mp["spur.weight",] for mp in dmpars], 1)
    spur_weight_percentiles = np.c_[
        np.percentile(spur_weights, 5, axis=1), np.percentile(spur_weights, 95, axis=1)
    ]

    # turn dropout back off
    manually_set_dropout(model, 0)
    model = model.eval()

stream_weight = mpars[("stream.weight",)]
stream_cutoff = stream_weight > 2e-2

spur_weight = mpars[("spur.weight",)]
spur_cutoff = spur_weight > 1e-2

bkg_prob = xp.exp(bkg_lnlik - tot_lnlik)
stream_prob = xp.exp(stream_lnlik - tot_lnlik)
spur_prob = xp.exp(spur_lnlik - tot_lnlik)
allstream_prob = xp.exp(xp.logaddexp(stream_lnlik, spur_lnlik) - tot_lnlik)

psort = np.argsort(allstream_prob)
# data = data[psort]
# stream_weight_percentiles = stream_weight_percentiles[psort]
# spur_weight_percentiles = spur_weight_percentiles[psort]

# =============================================================================
# Make Figure

fig = plt.figure(constrained_layout="tight", figsize=(11, 10))
gs = GridSpec(8, 1, figure=fig, height_ratios=(1, 1, 3, 5, 5, 5, 5, 5))

colors = color_by_probable_member(
    (stream_prob[psort], cmap1), (spur_prob[psort], cmap2)
)
alphas = p2alpha(allstream_prob[psort])
xlims = (data["phi1"].min(), 10)

# ---------------------------------------------------------------------------
# Colormap

# Stream probability
ax00 = fig.add_subplot(gs[0, :])
cbar = fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap1),
    cax=ax00,
    orientation="horizontal",
    label="Stream Probability",
)
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")

# Spur probability
ax01 = fig.add_subplot(gs[1, :])
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

ax02 = fig.add_subplot(gs[2, :], ylabel="Stream fraction", xlim=xlims, ylim=(0, 0.4))
ax02.set_xticklabels([])

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
    gs[3, :],
    xlabel="",
    ylabel=r"$\phi_2$ [$\degree$]",
    xlim=xlims,
    ylim=(data["phi2"].min(), data["phi2"].max()),
    xticklabels=[],
    rasterization_zorder=0,
)

# Stream control points
ax03.errorbar(
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
p1 = ax03.errorbar(
    stream_cp["phi1"],
    stream_cp["phi2"],
    yerr=stream_cp["w_phi2"],
    fmt=".",
    c=cmap1(0.99),
    capsize=2,
    zorder=-20,
    label="Stream Control Points",
)
# Spur control points
ax03.errorbar(
    spur_cp["phi1"],
    spur_cp["phi2"],
    yerr=spur_cp["w_phi2"],
    fmt="o",
    color="k",
    capthick=3,
    elinewidth=3,
    capsize=3,
    zorder=-21,
)
p2 = ax03.errorbar(
    spur_cp["phi1"],
    spur_cp["phi2"],
    yerr=spur_cp["w_phi2"],
    fmt=".",
    c=cmap2(0.99),
    capsize=2,
    zorder=-20,
    label="Spur Control Points",
)

# Data
d1 = ax03.scatter(
    data["phi1"][psort], data["phi2"][psort], c=colors, alpha=alphas, s=2, zorder=-10
)
# Model
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
    [d1, (p1, p2), (f1, f2)],
    ["Data", "Control points", "Models"],
    numpoints=1,
    handler_map={tuple: HandlerTuple(ndivide=None)},
    loc="upper left",
)


# ---------------------------------------------------------------------------
# Parallax

ax04 = fig.add_subplot(
    gs[4, :],
    xlabel="",
    ylabel=r"$\varpi$ [mas]",
    xlim=xlims,
    xticklabels=[],
    rasterization_zorder=0,
)

# Stream control points
ax04.errorbar(
    distance_cp["phi1"],
    distance_cp["parallax"],
    yerr=distance_cp["w_parallax"],
    fmt="o",
    color="k",
    capthick=3,
    elinewidth=3,
    capsize=3,
    zorder=-21,
)
p1 = ax04.errorbar(
    distance_cp["phi1"],
    distance_cp["parallax"],
    yerr=distance_cp["w_parallax"],
    fmt=".",
    c=cmap1(0.99),
    capsize=2,
    zorder=-20,
    label="Stream Control Points",
)
# Spur control points
ax04.errorbar(
    distance_cp["phi1"],
    distance_cp["parallax"],
    yerr=distance_cp["w_parallax"],
    fmt="o",
    color="k",
    capthick=3,
    elinewidth=3,
    capsize=3,
    zorder=-21,
)
p2 = ax04.errorbar(
    distance_cp["phi1"],
    distance_cp["parallax"],
    yerr=distance_cp["w_parallax"],
    fmt=".",
    c=cmap2(0.99),
    capsize=2,
    zorder=-20,
    label="Spur Control Points",
)

# Data
d1 = ax04.scatter(
    data["phi1"][psort], data["plx"][psort], c=colors, alpha=alphas, s=2, zorder=-10
)

# Model
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


# ---------------------------------------------------------------------------
# PM-Phi1

ax05 = fig.add_subplot(
    gs[5, :],
    xlabel="",
    ylabel=r"$\mu_{\phi_1}^*$ [mas yr$^{-1}$]",
    xlim=xlims,
    ylim=(data["pmphi1"].min(), data["pmphi1"].max()),
    xticklabels=[],
    rasterization_zorder=0,
)

# Stream control points
ax05.errorbar(
    stream_cp["phi1"],
    stream_cp["pm_phi1"],
    yerr=stream_cp["w_pm_phi1"],
    fmt="o",
    color="k",
    capthick=3,
    elinewidth=3,
    capsize=3,
    zorder=-21,
)
p1 = ax05.errorbar(
    stream_cp["phi1"],
    stream_cp["pm_phi1"],
    yerr=stream_cp["w_pm_phi1"],
    fmt=".",
    c=cmap1(0.99),
    capsize=2,
    zorder=-20,
    label="Stream Control Points",
)
# Spur control points
ax05.errorbar(
    spur_cp["phi1"],
    spur_cp["pm_phi1"],
    yerr=spur_cp["w_pm_phi1"],
    fmt="o",
    color="k",
    capthick=3,
    elinewidth=3,
    capsize=3,
    zorder=-21,
)
p2 = ax05.errorbar(
    spur_cp["phi1"],
    spur_cp["pm_phi1"],
    yerr=spur_cp["w_pm_phi1"],
    fmt=".",
    c=cmap2(0.99),
    capsize=2,
    zorder=-20,
    label="Spur Control Points",
)

# Data
d1 = ax05.scatter(
    data["phi1"][psort], data["pmphi1"][psort], c=colors, alpha=alphas, s=2, zorder=-10
)
# Model
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


# ---------------------------------------------------------------------------
# PM-Phi2

ax06 = fig.add_subplot(
    gs[6, :],
    ylabel=r"$\mu_{\phi_2}$ [mas yr$^{-1}$]",
    xlim=xlims,
    rasterization_zorder=0,
    xticklabels=[],
)

# Data
ax06.scatter(
    data["phi1"][psort], data["pmphi2"][psort], c=colors, alpha=alphas, s=2, zorder=-10
)
# Model
f1 = ax06.fill_between(
    data["phi1"][stream_cutoff],
    (mpa["pmphi2", "mu"] - xp.exp(mpa["pmphi2", "ln-sigma"]))[stream_cutoff],
    (mpa["pmphi2", "mu"] + xp.exp(mpa["pmphi2", "ln-sigma"]))[stream_cutoff],
    color=cmap1(0.99),
    alpha=0.25,
)
f2 = ax06.fill_between(
    data["phi1"][spur_cutoff],
    (mpb["pmphi2", "mu"] - xp.exp(mpb["pmphi2", "ln-sigma"]))[spur_cutoff],
    (mpb["pmphi2", "mu"] + xp.exp(mpb["pmphi2", "ln-sigma"]))[spur_cutoff],
    color=cmap2(0.99),
    alpha=0.25,
)


# ---------------------------------------------------------------------------
# Distance

mpa = mpars["stream.photometric.distmod"]

ax07 = fig.add_subplot(
    gs[7, :], xlabel=r"$\phi_1$ [deg]", ylabel=r"$d$ [kpc]", xlim=xlims
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

fig.savefig(paths.figures / "gd1" / "results_full.pdf")
