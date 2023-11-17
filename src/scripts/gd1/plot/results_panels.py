"""Plot Phi1-binned panels of the trained model."""

import sys

import asdf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch as xp
from astropy.table import QTable
from matplotlib.gridspec import GridSpec
from showyourwork.paths import user as user_paths
from tqdm import tqdm

import stream_ml.visualization as smlvis
from stream_ml.core import Data, Params

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.gd1.datasets import data
from scripts.gd1.model import make_model
from scripts.helper import (
    color_by_probable_member,
    manually_set_dropout,
    p2alpha,
    recursive_iterate,
)
from scripts.mpl_colormaps import stream_cmap1 as cmap1
from scripts.mpl_colormaps import stream_cmap2 as cmap2

# =============================================================================

# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

# isochrone data
with asdf.open(
    paths.data / "gd1" / "isochrone.asdf", "r", lazy_load=False, copy_arrays=True
) as af:
    isochrone_data = Data(**af["isochrone_data"])

# Load model
model = make_model()
model.load_state_dict(xp.load(paths.data / "gd1" / "models" / "model_11700.pt"))
model = model.eval()

# Load results from 4-likelihoods.py
lik_tbl = QTable.read(paths.data / "gd1" / "membership_likelhoods.ecsv")
bkg_prob = np.array(lik_tbl["bkg (50%)"])
stream_prob = np.array(lik_tbl["stream (50%)"])
stream_wgt = np.array(lik_tbl["stream.ln-weight"])
spur_prob = np.array(lik_tbl["spur (50%)"])
spur_wgt = np.array(lik_tbl["spur.ln-weight"])
allstream_prob = np.array(lik_tbl["allstream (50%)"])

# =============================================================================
# Cut down

keep = data["phi1"] < 10  # [deg]
data = data[keep]
bkg_prob = bkg_prob[keep]
stream_prob = stream_prob[keep]
stream_wgt = stream_wgt[keep]
spur_prob = spur_prob[keep]
spur_wgt = spur_wgt[keep]
allstream_prob = allstream_prob[keep]

# =============================================================================
# Likelihood

# Sorter for plotting
psort = np.argsort(allstream_prob)

# Foreground
_is_strm = (stream_prob > 0.6) & (stream_wgt.mean(1) > -4)
strm_range = (np.min(data["phi1"][_is_strm].numpy()) <= data["phi1"]) & (
    data["phi1"] <= np.max(data["phi1"][_is_strm].numpy())
)
_is_spur = (spur_prob > 0.6) & (spur_wgt.mean(1) > -5)
spur_range = (np.min(data["phi1"][_is_spur].numpy()) <= data["phi1"]) & (
    data["phi1"] <= np.max(data["phi1"][_is_spur].numpy())
)

# Also evaluate the model with dropout on
with xp.no_grad():
    # turn dropout on
    model = model.train()
    manually_set_dropout(model, 0.15)

    # evaluate the model
    ldmpars = [model.unpack_params(model(data)) for i in tqdm(range(100))]

    # mpars
    dmpars = Params(recursive_iterate(ldmpars, ldmpars[0], reduction=lambda x: x))
    mpars = Params(recursive_iterate(ldmpars, ldmpars[0]))

    # turn dropout back off
    manually_set_dropout(model, 0)
    model = model.eval()

# =============================================================================
# Make Figure

fig = plt.figure(figsize=(11, 4.2))

gs = GridSpec(
    2,
    1,
    figure=fig,
    height_ratios=(1, 2),
    hspace=0.4,
    left=0.07,
    right=0.98,
    top=0.965,
    bottom=0.8,
)

colors = color_by_probable_member(
    (stream_prob[psort], cmap1), (spur_prob[psort], cmap2)
)
alphas = p2alpha(allstream_prob[psort])
xlims = (data["phi1"].min().numpy(), data["phi1"].max().numpy())

# ---------------------------------------------------------------------------
# Phi2

ax0 = fig.add_subplot(
    gs[0, :],
    xlabel=r"$\phi_1$ [deg]",
    xlim=xlims,
    ylabel=r"$\phi_2$ [deg]",
    ylim=(-4, 4),
    rasterization_zorder=0,
)

ax0.scatter(
    data["phi1"][psort], data["phi2"][psort], c=colors, alpha=alphas, s=2, zorder=-10
)

xlabel = ax0.xaxis.get_label()
xlabel.set_bbox({"facecolor": "white", "edgecolor": "white"})

# =============================================================================
# Slice plots

gs1 = gs[1].subgridspec(1, 4, hspace=0.45)

# Legend
legend1 = plt.legend(
    handles=[
        mpl.patches.Patch(color=cmap1(0.01), label="Background"),
        mpl.patches.Patch(color=cmap1(0.99), label="Stream"),
        mpl.patches.Patch(color=cmap2(0.99), label="Spur"),
    ],
    ncols=4,
    loc="upper right",
    bbox_to_anchor=(1, -0.14),
)
ax0.add_artist(legend1)

# Bin the data for plotting
bins = np.linspace(data["phi1"].min(), data["phi1"].max(), num=5, endpoint=True)
which_bin = np.digitize(data["phi1"], bins[:-1])

ax10 = None

for i, b in enumerate(np.unique(which_bin)):
    sel = which_bin == b

    data_ = data[psort][sel[psort]]
    bkg_prob_ = bkg_prob[psort][sel[psort]]
    stream_prob_ = stream_prob[psort][sel[psort]]
    spur_prob_ = spur_prob[psort][sel[psort]]

    # ---------------------------------------------------------------------------
    # Photometry

    ax1i = fig.add_subplot(
        gs1[i],
        xlabel=("g - r [mag]"),
        xlim=(0, 1),
        ylim=(21, 13),
        xticklabels=[],
        rasterization_zorder=20,
    )

    ax1i.scatter(
        data_["g"] - data_["r"],
        data_["g"],
        c=colors[sel[psort]],
        s=1,
        zorder=-10,
    )
    ax1i.plot(
        isochrone_data["g"] - isochrone_data["r"],
        isochrone_data["g"]
        + mpars["stream.photometric.distmod", "mu"][sel].mean().numpy(),
        c="green",
        label="Isochrone",
        zorder=-5,
    )

    # Connect to top plot(s)
    for ax in (ax0,):
        ax.axvline(bins[i], color="gray", ls="--", zorder=-200)
        ax.axvline(bins[i + 1], color="gray", ls="--", zorder=-200)
    smlvis._slices.connect_slices_to_top(  # noqa: SLF001
        fig, ax0, ax1i, left=bins[i], right=bins[i + 1], color="gray"
    )

    if i == 0:
        ax1i.set_ylabel("g [mag]")
        ax1i.legend(loc="upper left")
        ax10 = ax1i
    else:
        ax1i.tick_params(labelleft=False)


fig.savefig(paths.figures / "gd1" / "results_panels.pdf")
