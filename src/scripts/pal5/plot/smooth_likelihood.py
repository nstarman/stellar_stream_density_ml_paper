"""Plot pal5 Likelihoods."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as xp
from scipy import stats

# Add the parent directory to the path
sys.path.append(Path(__file__).parents[3].as_posix())
# isort: split

from scripts import paths
from scripts.pal5.datasets import data, where
from scripts.pal5.define_model import model

# =============================================================================

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

# Load model
model.load_state_dict(xp.load(paths.data / "pal5" / "model.pt"))
model = model.eval()

with xp.no_grad():
    mpars = model.unpack_params(model(data))

    stream_lnlik = model.component_ln_likelihood("stream", mpars, data, where=where)
    tot_lnlik = model.ln_likelihood(mpars, data, where=where)

tot_stream_prob = np.exp(stream_lnlik - tot_lnlik)

# =============================================================================
# Plot

fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

nddata = data.astype(np.ndarray)

# -------------------------------------------------
# Phi2(phi1)

kernel_inp = np.vstack([nddata["phi1"], nddata["phi2"]])
kernel = stats.gaussian_kde(kernel_inp, weights=tot_stream_prob, bw_method=0.2)
X, Y = np.mgrid[
    nddata["phi1"].min() : nddata["phi1"].max() : 100j,
    nddata["phi2"].min() : nddata["phi2"].max() : 100j,
]
kernal_eval = np.vstack([X.flatten(), Y.flatten()])
axs[0].pcolormesh(
    X,
    Y,
    kernel(kernal_eval).reshape(X.shape),
    # cmap="twilight",
    rasterized=True,
    shading="gouraud",
)
axs[0].set(ylabel=(r"$\phi_2$ [deg]"), xlim=(-90, 10), ylim=(-5, 5))

# -------------------------------------------------
# pmphi1(phi1)

kernel_inp = np.vstack([nddata["phi1"], nddata["pmphi1"]])
kernel = stats.gaussian_kde(kernel_inp, weights=tot_stream_prob, bw_method=0.2)
X, Y = np.mgrid[
    nddata["phi1"].min() : nddata["phi1"].max() : 100j,
    nddata["pmphi1"].min() : nddata["pmphi1"].max() : 100j,
]
kernal_eval = np.vstack([X.flatten(), Y.flatten()])
axs[1].pcolormesh(
    X,
    Y,
    kernel(kernal_eval).reshape(X.shape),
    # cmap="twilight",
    rasterized=True,
    shading="gouraud",
)
axs[1].set(ylabel=(r"$\mu_{\phi_1}^*$ [deg]"), xlim=(-90, 10))

# -------------------------------------------------
# pmphi2(phi1)

kernel_inp = np.vstack([nddata["phi1"], nddata["pmphi2"]])
kernel = stats.gaussian_kde(kernel_inp, weights=tot_stream_prob, bw_method=0.2)
X, Y = np.mgrid[
    nddata["phi1"].min() : nddata["phi1"].max() : 100j,
    nddata["pmphi2"].min() : nddata["pmphi2"].max() : 100j,
]
kernal_eval = np.vstack([X.flatten(), Y.flatten()])
axs[2].pcolormesh(
    X,
    Y,
    kernel(kernal_eval).reshape(X.shape),
    # cmap="twilight",
    rasterized=True,
    shading="gouraud",
)
axs[2].set(xlabel=(r"$\phi_1$ [deg]"), ylabel=(r"$\mu_{\phi_2}$ [deg]"), xlim=(-90, 10))

# -------------------------------------------------

fig.savefig(paths.figures / "pal5" / "smooth_likelihood.pdf")
