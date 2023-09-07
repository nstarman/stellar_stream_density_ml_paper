"""Train photometry background flow."""

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# isort: split
from showyourwork.paths import user as user_paths

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.as_posix())
# isort: split

from scripts.pal5.datasets import data, off_stream

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

axs[0].plot(
    data["phi1"][off_stream],
    data["phi2"][off_stream],
    ls="none",
    marker=",",
    ms=1e-2,
    color="k",
    rasterized=True,
)
axs[0].plot(
    data["phi1"][~off_stream],
    data["phi2"][~off_stream],
    ls="none",
    marker=",",
    ms=1e-2,
    color="tab:blue",
    alpha=0.5,
    rasterized=True,
)
axs[0].set(
    xlabel=(r"$\phi_1 \ $ [$\degree$]"), ylabel=(r"$\phi_2 \ $ [$\degree$]"), aspect=2
)
axs[0].grid(visible=True, which="both", axis="y")
axs[0].grid(visible=True, which="major", axis="x")

axs[1].scatter(
    data["g"][off_stream] - data["r"][off_stream],
    data["g"][off_stream],
    s=1,
    color="k",
    rasterized=True,
    label="off-stream",
)
alpha = stream_kde(positions.T) - background_kde(positions.T)
alpha[alpha < 0] = 0
axs[1].scatter(
    data["g"][~off_stream] - data["r"][~off_stream],
    data["g"][~off_stream],
    s=1,
    alpha=alpha / alpha.max(),
    rasterized=True,
    label="on-stream - off-stream KDE",
)
axs[1].set(
    xlabel=(r"$g-r \ $ [mag]"), ylabel=(r"$g \ $ [mag]"), aspect="auto", xlim=(0, 0.8)
)
axs[1].invert_yaxis()
axs[1].grid(visible=True, which="major")
axs[1].legend(loc="upper left")

fig.tight_layout()
fig.savefig(paths.figures / "pal5" / "photometric_background_selection.pdf")
