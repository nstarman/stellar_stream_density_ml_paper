"""Plot the photometry background flow."""

import sys

import asdf
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
from scipy.stats import gaussian_kde
from showyourwork.paths import user as user_paths

import stream_mapper.pytorch as sml

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

# =============================================================================

# Load data
table = QTable.read(paths.data / "gd1" / "gaia_ps1_xm.asdf")
masks = QTable.read(paths.data / "gd1" / "masks.asdf")

# Make Mask
pm_tight_mask = (
    (table["pm_phi1"] > -15 * u.mas / u.yr)
    & (table["pm_phi1"] < -10 * u.mas / u.yr)
    & (table["pm_phi2"] > -4.5 * u.mas / u.yr)
    & (table["pm_phi2"] < -2 * u.mas / u.yr)
)
completeness_mask = table["gaia_g"] < 20 * u.mag

sel = (
    pm_tight_mask
    & completeness_mask
    & masks["low_phi2"]
    & ~np.isnan(table["g0"])
    & ~np.isnan(table["r0"])
    & (table["phi1"] > -80 * u.deg)
    & (table["phi1"] < 10 * u.deg)
)

# Apply mask
table = table[sel]
masks = masks[sel]

# Get off-stream selection
off_stream = ~masks["offstream"]

# Turn into sml.Data
with asdf.open(
    paths.data / "gd1" / "info.asdf", lazy_load=False, copy_arrays=True
) as af:
    names = tuple(af["names"])
    renamer = af["renamer"]
data = sml.Data.from_format(table, fmt="astropy.table", names=names, renamer=renamer)
where = sml.Data((~np.isnan(data.array)), names=data.names)

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

fig, axs = plt.subplots(2, 1, figsize=(5, 5), height_ratios=[1, 3])

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
    zorder=-10,
)
axs[0].set_axisbelow(False)

# -------------------------------------------------
# Photometric

axs[1].set(
    xlabel=(r"$g-r \ $ [mag]"),
    ylabel=(r"$g \ $ [mag]"),
    aspect="auto",
    xlim=(0.1, 0.6),
    ylim=(21, 13),
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
    alpha=0.1 + 0.9 * alpha,
    zorder=-10,
    label="on-off",
)
leg = axs[1].legend(loc="upper left", fontsize=11, markerscale=5)
for lh in leg.legend_handles:
    lh.set_alpha(1)

fig.tight_layout()

fig.savefig(paths.figures / "gd1" / "photometric_background_selection.pdf")
