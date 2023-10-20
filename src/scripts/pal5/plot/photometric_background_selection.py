"""Train photometry background flow."""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import gaussian_kde

# isort: split
from showyourwork.paths import user as user_paths

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.pal5.datasets import data, off_stream

data = data.astype(np.ndarray)

# =============================================================================

bw_method = 0.05
background_kde = gaussian_kde(
    np.c_[data["g"][off_stream], data["r"][off_stream]].T,
    bw_method=bw_method,
)
positions = np.c_[data["g"][~off_stream], data["r"][~off_stream]]
stream_kde = gaussian_kde(positions.T, bw_method=bw_method)

# =============================================================================

plt.style.use(paths.scripts / "paper.mplstyle")

fig, axs = plt.subplots(2, 1, figsize=(4, 4))

# -------------------------------------------------
# Phi1-phi2

axs[0].set(
    xlabel=(r"$\phi_1 \ $ [deg]"),
    ylabel=(r"$\phi_2 \ $ [deg]"),
    rasterization_zorder=0,
)
axs[0].hist2d(
    data["phi1"][off_stream],
    data["phi2"][off_stream],
    cmap="gray",
    bins=300,
    zorder=-10,
)
axs[0].plot(
    data["phi1"][~off_stream],
    data["phi2"][~off_stream],
    ls="none",
    marker=",",
    ms=1e-2,
    color="tab:blue",
    alpha=0.5,
    zorder=-5,
)
axs[0].grid(visible=True, which="both", axis="y")
axs[0].grid(visible=True, which="major", axis="x")

# -------------------------------------------------
# Photometric

axs[1].set(
    xlabel=(r"$g-r \ $ [mag]"),
    ylabel=(r"$g \ $ [mag]"),
    aspect="auto",
    xlim=(0, 0.8),
    rasterization_zorder=0,
)
axs[1].scatter(
    data["g"][off_stream] - data["r"][off_stream],
    data["g"][off_stream],
    s=1,
    color="k",
    label="off",
    zorder=-10,
)
alpha = stream_kde(positions.T) - background_kde(positions.T)
alpha[alpha < 0] = 0
alpha[alpha > 1] = 1
axs[1].scatter(
    data["g"][~off_stream] - data["r"][~off_stream],
    data["g"][~off_stream],
    s=1,
    alpha=0.05 + 0.95 * alpha,
    label="on-off",
    zorder=-5,
)
axs[1].invert_yaxis()
axs[1].yaxis.set_major_formatter(FormatStrFormatter("%d"))
leg = axs[1].legend(loc="upper left", fontsize=11, markerscale=5)
for lh in leg.legendHandles:
    lh.set_alpha(1)


fig.tight_layout()
fig.savefig(paths.figures / "pal5" / "photometric_background_selection.pdf")
