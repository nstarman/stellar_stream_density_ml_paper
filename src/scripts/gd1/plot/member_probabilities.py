"""Plot the photometry background flow."""

import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
from showyourwork.paths import user as user_paths

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.mpl_colormaps import stream_cmap1 as cmap_stream

# =============================================================================
# Load data

lik_tbl = QTable.read(paths.data / "gd1" / "membership_likelhoods.ecsv")

liks = lik_tbl["allstream (MLE)"]
prob_arr = np.linspace(0.01, liks.max(), num=100)
num_gtr = np.array([np.sum(liks >= p) for p in prob_arr], dtype=int)
num_gtr = num_gtr[0] - num_gtr

# =============================================================================
# Plot

plt.style.use(paths.scripts / "paper.mplstyle")

fig, ax1 = plt.subplots(1, 1, figsize=(7, 3.5))

# Plotting on the primary y-axis
n, bins, patches = ax1.hist(liks, bins=30, edgecolor="black", log=True)
# Coloring each bar
for b, patch in zip((bins[1:] + bins[:-1]) / 2, patches):
    color = cmap_stream(b)
    patch.set_facecolor(color)

ax1.set_xlabel(r"Membership Probability", fontsize=15)
ax1.set_ylabel("Membership Frequency", fontsize=15)

# Creating a second y-axis
ax2 = ax1.twinx()

# Plotting on the secondary y-axis
ax2.plot(prob_arr, num_gtr, color="#228B22", lw=4, rasterized=True)
ax2.set_ylabel(r"Stars with $p(X \leq x)$", fontsize=15, color="#228B22")
for label in ax2.get_yticklabels():
    label.set_color("#228B22")

fig.tight_layout()
fig.savefig(paths.figures / "gd1" / "member_probabilities.pdf")
