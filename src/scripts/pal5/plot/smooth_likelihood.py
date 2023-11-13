"""Plot pal5 Likelihoods."""

import copy
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch as xp
from scipy import stats
from showyourwork.paths import user as user_paths

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split
from scripts.pal5.datasets import data, masks, where
from scripts.pal5.model import model

# =============================================================================

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

# Load model
model = copy.deepcopy(model)
model.load_state_dict(xp.load(paths.data / "pal5" / "models" / "model_11700.pt"))
model = model.eval()

# Let's cut out the progenitor
# data_prog = data[~masks["Pal5"]]
# where_prog = where[~masks["Pal5"]]
# data = data[masks["Pal5"]]
# where = where[masks["Pal5"]]

# Evaluate likelihood
with xp.no_grad():
    mpars = model.unpack_params(model(data))

    stream_lnlik = model.component_ln_likelihood("stream", mpars, data, where=where)
    tot_lnlik = model.ln_likelihood(mpars, data, where=where)

tot_stream_prob = np.exp(stream_lnlik - tot_lnlik)

# =============================================================================
# Plot

fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

nddata = data.astype(np.ndarray)

# Cut out the progenitor
nddata_prog = nddata[~masks["Pal5"]]
nddata = nddata[masks["Pal5"]]

tot_stream_prob_prog = tot_stream_prob[~masks["Pal5"]]
tot_stream_prob = tot_stream_prob[masks["Pal5"]]

# -------------------------------------------------
# Phi2(phi1)

kernel_inp = np.vstack([nddata["phi1"], nddata["phi2"]])
kernel = stats.gaussian_kde(kernel_inp, weights=tot_stream_prob, bw_method=0.1)
X, Y = np.mgrid[
    nddata["phi1"].min() : nddata["phi1"].max() : 100j,
    nddata["phi2"].min() : nddata["phi2"].max() : 100j,
]
kernal_eval = np.vstack([X.flatten(), Y.flatten()])
im = axs[0].pcolormesh(
    X,
    Y,
    kernel(kernal_eval).reshape(X.shape),
    rasterized=True,
    shading="gouraud",
)

# Show progenitor mask
axs[0].axvspan(
    nddata_prog["phi1"].min(), nddata_prog["phi1"].max(), color="black", zorder=100
)

axs[0].set(ylabel=(r"$\phi_2$ [deg]"))

# -------------------------------------------------
# pmphi1(phi1)

kernel_inp = np.vstack([nddata["phi1"], nddata["pmphi1"]])
kernel = stats.gaussian_kde(kernel_inp, weights=tot_stream_prob, bw_method=0.1)
X, Y = np.mgrid[
    nddata["phi1"].min() : nddata["phi1"].max() : 100j,
    nddata["pmphi1"].min() : nddata["pmphi1"].max() : 100j,
]
kernal_eval = np.vstack([X.flatten(), Y.flatten()])
axs[1].pcolormesh(
    X,
    Y,
    kernel(kernal_eval).reshape(X.shape),
    rasterized=True,
    shading="gouraud",
)

# Show progenitor mask
axs[1].axvspan(
    nddata_prog["phi1"].min(), nddata_prog["phi1"].max(), color="black", zorder=100
)

axs[1].set(ylabel=(r"$\mu_{\phi_1}^*$ [deg]"))

# -------------------------------------------------
# pmphi2(phi1)

kernel_inp = np.vstack([nddata["phi1"], nddata["pmphi2"]])
kernel = stats.gaussian_kde(kernel_inp, weights=tot_stream_prob, bw_method=0.1)
X, Y = np.mgrid[
    nddata["phi1"].min() : nddata["phi1"].max() : 100j,
    nddata["pmphi2"].min() : nddata["pmphi2"].max() : 100j,
]
kernal_eval = np.vstack([X.flatten(), Y.flatten()])
axs[2].pcolormesh(
    X,
    Y,
    kernel(kernal_eval).reshape(X.shape),
    rasterized=True,
    shading="gouraud",
)

# Show progenitor mask
axs[2].axvspan(
    nddata_prog["phi1"].min(), nddata_prog["phi1"].max(), color="black", zorder=100
)

axs[2].set(xlabel=(r"$\phi_1$ [deg]"), ylabel=(r"$\mu_{\phi_2}$ [deg]"))

# -------------------------------------------------

fig.savefig(paths.figures / "pal5" / "smooth_likelihood.pdf")
