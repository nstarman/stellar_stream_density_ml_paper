"""Plot GD1 Likelihoods."""

import sys
from pathlib import Path

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch as xp
from astropy.coordinates import Distance
from matplotlib.gridspec import GridSpec

# Add the parent directory to the path
sys.path.append(Path(__file__).parents[3].as_posix())
# isort: split

from scripts import paths
from scripts.gd1.datasets import data, where
from scripts.gd1.define_model import model
from scripts.helper import manually_set_dropout

# =============================================================================

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

# Load model
model.load_state_dict(xp.load(paths.data / "gd1" / "model.pt"))
model = model.eval()

# Evaluate model
with xp.no_grad():
    mpars = model.unpack_params(model(data))

    stream_lik = model.component_posterior("stream", mpars, data, where=where)
    spur_lik = model.component_posterior("spur", mpars, data, where=where)
    bkg_lik = model.component_posterior("background", mpars, data, where=where)
    tot_lik = model.posterior(mpars, data, where=where)


stream_weight = mpars[("stream.weight",)]
stream_cutoff = stream_weight > 2e-2

bkg_prob = bkg_lik / tot_lik
stream_prob = stream_lik / tot_lik
spur_prob = spur_lik / tot_lik
allstream_prob = (stream_lik + spur_lik) / tot_lik

psort = np.argsort(allstream_prob)
pmax = allstream_prob.max()
pmin = allstream_prob.min()


# =============================================================================
# Make Figure

fig = plt.figure(constrained_layout="tight", figsize=(11, 10))
gs = GridSpec(6, 1, figure=fig, height_ratios=(1, 3, 5, 5, 5, 5))

cmap = plt.get_cmap()

# ---------------------------------------------------------------------------
# Colormap

ax00 = fig.add_subplot(gs[0, :])
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

ax01 = fig.add_subplot(gs[1, :])
ax01.set(ylabel="Stream fraction", ylim=(0, 0.35))
ax01.set_xticklabels([])

with xp.no_grad():
    model = model.train()
    manually_set_dropout(model, 0.15)
    weights = xp.stack(
        [model.unpack_params(model(data))["stream.weight",] for i in range(100)], 1
    )
    weight_percentiles = np.c_[
        np.percentile(weights, 5, axis=1), np.percentile(weights, 95, axis=1)
    ]
    manually_set_dropout(model, 0)
    model = model.eval()
ax01.fill_between(
    data["phi1"],
    weight_percentiles[:, 0],
    weight_percentiles[:, 1],
    color="k",
    alpha=0.25,
    label=r"Model (15% dropout)",
)
ax01.plot(data["phi1"], stream_weight, c="k", ls="--", lw=2, label="Model (MLE)")
ax01.legend(loc="upper left")

# ---------------------------------------------------------------------------
# Phi2

mpa = mpars.get_prefixed("stream.astrometric")

ax02 = fig.add_subplot(gs[2, :])
ax02.set_xticklabels([])
ax02.set(ylabel=r"$\phi_2$ [$\degree$]")

ax02.scatter(
    data["phi1"][psort],
    data["phi2"][psort],
    c=allstream_prob[psort],
    alpha=0.1 + (1 - 0.1) / (pmax - pmin) * (allstream_prob[psort] - pmin),
    s=2,
    zorder=-10,
)
ax02.set_rasterization_zorder(0)
ax02.fill_between(
    data["phi1"][stream_cutoff],
    (mpa["phi2", "mu"] - xp.exp(mpa["phi2", "ln-sigma"]))[stream_cutoff],
    (mpa["phi2", "mu"] + xp.exp(mpa["phi2", "ln-sigma"]))[stream_cutoff],
    color="k",
    alpha=0.25,
    label="Model (MLE)",
)
ax02.legend(loc="upper left")

# ---------------------------------------------------------------------------
# PM-Phi1

ax03 = fig.add_subplot(gs[3, :])
ax03.set_xticklabels([])
ax03.set(ylabel=r"$\mu_{\phi_1}^*$ [mas yr$^{-1}$]")

ax03.scatter(
    data["phi1"][psort],
    data["pmphi1"][psort],
    c=allstream_prob[psort],
    alpha=0.1 + (1 - 0.1) / (pmax - pmin) * (stream_prob[psort] - pmin),
    s=2,
    zorder=-10,
)
ax03.set_rasterization_zorder(0)
ax03.fill_between(
    data["phi1"][stream_cutoff],
    (mpa["pmphi1", "mu"] - xp.exp(mpa["pmphi1", "ln-sigma"]))[stream_cutoff],
    (mpa["pmphi1", "mu"] + xp.exp(mpa["pmphi1", "ln-sigma"]))[stream_cutoff],
    color="k",
    alpha=0.25,
    label="Model (MLE)",
)
ax03.legend(loc="upper left")

# ---------------------------------------------------------------------------
# PM-Phi2

ax04 = fig.add_subplot(gs[4, :])
ax04.set_xticklabels([])
ax04.set(ylabel=r"$\mu_{\phi_2}$ [mas yr$^{-1}$]")

ax04.scatter(
    data["phi1"][psort],
    data["pmphi2"][psort],
    c=allstream_prob[psort],
    alpha=0.1 + (1 - 0.1) / (pmax - pmin) * (stream_prob[psort] - pmin),
    s=2,
    zorder=-10,
)
ax04.set_rasterization_zorder(0)
ax04.fill_between(
    data["phi1"][stream_cutoff],
    (mpa["pmphi2", "mu"] - xp.exp(mpa["pmphi2", "ln-sigma"]))[stream_cutoff],
    (mpa["pmphi2", "mu"] + xp.exp(mpa["pmphi2", "ln-sigma"]))[stream_cutoff],
    color="k",
    alpha=0.25,
    label="Model (MLE)",
)
ax04.legend(loc="upper left")


# ---------------------------------------------------------------------------
# Distance

mpa = mpars["stream.photometric.distmod"]

ax03 = fig.add_subplot(gs[5, :])
ax03.set(xlabel=r"$\phi_1$ [deg]", ylabel=r"$d$ [kpc]")

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

fig.savefig(paths.figures / "gd1" / "results_full.pdf")
