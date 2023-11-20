"""Plot GD1 Likelihoods."""

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch as xp
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from scipy import stats
from showyourwork.paths import user as user_paths

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split
from scripts.gd1.datasets import data, where
from scripts.gd1.model import make_model

# =============================================================================

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

# Load model
model = make_model()
model.load_state_dict(xp.load(paths.data / "gd1" / "models" / "model_11700.pt"))
model = model.eval()

with xp.no_grad():
    mpars = model.unpack_params(model(data))

    stream_lnlik = model.component_ln_likelihood("stream", mpars, data, where=where)
    spur_lnlik = model.component_ln_likelihood("spur", mpars, data, where=where)
    tot_lnlik = model.ln_likelihood(mpars, data, where=where)

tot_stream_prob = np.exp(np.logaddexp(stream_lnlik, spur_lnlik) - tot_lnlik)

nddata = data.astype(np.ndarray)

# =============================================================================
# Plot

fig = plt.figure(figsize=(6, 4.5))

gs = GridSpec(
    4,
    1,
    figure=fig,
    height_ratios=(1, 5, 5, 5),
    hspace=0.15,
    left=0.12,
    right=0.98,
    top=0.94,
    bottom=0.1,
)
cmap = plt.get_cmap()

# ---------------------------------------------------------------------------
# Colormap

ax0 = fig.add_subplot(gs[0, :])
cbar = fig.colorbar(ScalarMappable(cmap=cmap), cax=ax0, orientation="horizontal")
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")
cbar.ax.text(0.5, 0.5, "Stream Probability", ha="center", va="center", fontsize=14)

# -------------------------------------------------
# Phi2(phi1)

ax1 = fig.add_subplot(
    gs[1, :], xlim=(-90, 10), ylabel=(r"$\phi_2$ [deg]"), ylim=(-5, 5)
)
ax1.tick_params(labelbottom=False)

kernel_inp = np.vstack([nddata["phi1"], nddata["phi2"]])
kernel = stats.gaussian_kde(kernel_inp, weights=tot_stream_prob, bw_method=0.2)
X, Y = np.mgrid[
    nddata["phi1"].min() : nddata["phi1"].max() : 100j,
    nddata["phi2"].min() : nddata["phi2"].max() : 100j,
]
kernal_eval = np.vstack([X.flatten(), Y.flatten()])
ax1.pcolormesh(
    X,
    Y,
    kernel(kernal_eval).reshape(X.shape),
    rasterized=True,
    shading="gouraud",
    cmap=cmap,
)

# -------------------------------------------------
# pmphi1(phi1)

ax2 = fig.add_subplot(gs[2, :], xlim=(-90, 10), ylabel=r"$\mu_{\phi_1}$ [mas/yr]")
ax2.tick_params(labelbottom=False)

kernel_inp = np.vstack([nddata["phi1"], nddata["pmphi1"]])
kernel = stats.gaussian_kde(kernel_inp, weights=tot_stream_prob, bw_method=0.2)
X, Y = np.mgrid[
    nddata["phi1"].min() : nddata["phi1"].max() : 100j,
    nddata["pmphi1"].min() : nddata["pmphi1"].max() : 100j,
]
kernal_eval = np.vstack([X.flatten(), Y.flatten()])
ax2.pcolormesh(
    X,
    Y,
    kernel(kernal_eval).reshape(X.shape),
    rasterized=True,
    shading="gouraud",
    cmap=cmap,
)

# -------------------------------------------------
# pmphi2(phi1)

ax3 = fig.add_subplot(
    gs[3, :],
    xlabel=r"$\phi_1$ [mas/yr]",
    xlim=(-90, 10),
    ylabel=r"$\mu_{\phi_2}$ [mas/yr]",
)

kernel_inp = np.vstack([nddata["phi1"], nddata["pmphi2"]])
kernel = stats.gaussian_kde(kernel_inp, weights=tot_stream_prob, bw_method=0.2)
X, Y = np.mgrid[
    nddata["phi1"].min() : nddata["phi1"].max() : 100j,
    nddata["pmphi2"].min() : nddata["pmphi2"].max() : 100j,
]
kernal_eval = np.vstack([X.flatten(), Y.flatten()])
ax3.pcolormesh(
    X,
    Y,
    kernel(kernal_eval).reshape(X.shape),
    rasterized=True,
    shading="gouraud",
    cmap=cmap,
)
ax3.set(
    xlabel=(r"$\phi_1$ [mas/yr]"), ylabel=(r"$\mu_{\phi_2}$ [mas/yr]"), xlim=(-90, 10)
)

# -------------------------------------------------

fig.savefig(paths.figures / "gd1" / "smooth_likelihood.pdf")
