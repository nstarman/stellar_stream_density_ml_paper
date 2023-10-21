"""Plot the photometry background flow."""

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# isort: split
from showyourwork.paths import user as user_paths

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.gd1.datasets import data, off_stream

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

fig, axs = plt.subplots(2, 1, figsize=(6, 6))

# -------------------------------------------------
# Phi1-phi2

axs[0].set(
    xlabel=(r"$\phi_1 \ $ [deg]"),
    ylabel=(r"$\phi_2 \ $ [deg]"),
    aspect=2,
    rasterization_zorder=0,
)
_kw = {"ls": "none", "marker": ",", "ms": 1e-2}
axs[0].plot(
    data["phi1"][off_stream], data["phi2"][off_stream], color="k", **_kw, zorder=-10
)
axs[0].plot(
    data["phi1"][~off_stream],
    data["phi2"][~off_stream],
    color="tab:blue",
    alpha=0.5,
    **_kw,
    zorder=-10
)
axs[0].set_axisbelow(False)

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
    label="off-stream",
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
    zorder=-10,
    label="on-off",
)
axs[1].invert_yaxis()
leg = axs[1].legend(loc="upper left", fontsize=11, markerscale=5)
for lh in leg.legendHandles:
    lh.set_alpha(1)

fig.tight_layout()
fig.savefig(paths.figures / "gd1" / "photometric_background_selection.pdf")
