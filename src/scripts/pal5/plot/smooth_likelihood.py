"""Plot pal5 Likelihoods."""

import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from scipy import stats
from showyourwork.paths import user as user_paths

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.pal5.datasets import data

# =============================================================================

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

# Likelihood
lik_tbl = QTable.read(paths.data / "pal5" / "membership_likelhoods.ecsv")
tot_stream_prob = lik_tbl["stream (MLE)"]

cmap = plt.get_cmap("bone_r")

nddata = data.astype(np.ndarray)

# =============================================================================
# Kernel density estimates

Ys = [None, None, None]
Zs = [None, None, None]
for i, k in enumerate(("phi2", "pmphi1", "pmphi2")):
    kernel_inp = np.vstack([nddata["phi1"], nddata[k]])
    kernel = stats.gaussian_kde(kernel_inp, weights=tot_stream_prob, bw_method=0.05)
    X, Ys[i] = np.mgrid[
        nddata["phi1"].min() : nddata["phi1"].max() : 100j,
        nddata[k].min() : nddata[k].max() : 100j,
    ]
    kernal_eval = np.vstack([X.flatten(), Ys[i].flatten()])

    Zs[i] = kernel(kernal_eval).reshape(X.shape)

# Rescale
z_max = max([np.max(Z) for Z in Zs])
Zs = [Z / z_max for Z in Zs]


# =============================================================================
# Plot

fig = plt.figure(figsize=(8, 6))
gs = GridSpec(4, 1, figure=fig, height_ratios=(1, 4, 4, 4), wspace=0.05)

# Colorbar
ax0 = fig.add_subplot(gs[0, 0], xticklabels=[], yticklabels=[])

# -------------------------------------------------
# Phi2(phi1)

ax1 = fig.add_subplot(
    gs[1, 0],
    ylabel=r"$\phi_2$ [deg]",
    ylim=(-0.5, 1.5),
    rasterization_zorder=100,
    axisbelow=False,
)
ax1.tick_params(labelbottom=False)
ax1.pcolormesh(X, Ys[0], Zs[0], zorder=0, shading="gouraud", cmap=cmap)

# -------------------------------------------------
# pmphi1(phi1)

ax2 = fig.add_subplot(
    gs[2, 0],
    ylabel=(r"$\mu_{\phi_1}^*$ [deg]"),
    rasterization_zorder=100,
    axisbelow=False,
    sharex=ax1,
)
ax2.tick_params(labelbottom=False)
ax2.pcolormesh(X, Ys[1], Zs[1], zorder=0, shading="gouraud", cmap=cmap)


# -------------------------------------------------
# pmphi2(phi1)

ax3 = fig.add_subplot(
    gs[3, 0],
    xlabel=(r"$\phi_1$ [deg]"),
    ylabel=(r"$\mu_{\phi_2}$ [deg]"),
    rasterization_zorder=100,
    axisbelow=False,
    sharex=ax1,
)
out = ax3.pcolormesh(X, Ys[2], Zs[2], zorder=0, shading="gouraud", cmap=cmap)


# -------------------------------------------------
# Colorbar

cbar = fig.colorbar(ScalarMappable(cmap=cmap), cax=ax0, orientation="horizontal")
cbar.ax.xaxis.set_ticks_position("top")
cbar.ax.xaxis.set_label_position("top")
cbar.ax.text(0.5, 0.5, "Stream Probability", ha="center", va="center", fontsize=14)

# -------------------------------------------------

fig.savefig(paths.figures / "pal5" / "smooth_likelihood.pdf")
