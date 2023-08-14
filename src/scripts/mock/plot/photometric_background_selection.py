"""Train photometry background flow."""

import sys
from pathlib import Path

import asdf
import matplotlib.pyplot as plt
import numpy as np
import torch as xp

import stream_ml.pytorch as sml

# isort: split
# Add the parent directory to the path
sys.path.append(Path(__file__).parents[3].as_posix())
# isort: split

from scripts import paths

# =============================================================================

plt.style.use(paths.scripts / "paper.mplstyle")


with asdf.open(
    paths.data / "mock" / "data.asdf", lazy_load=False, copy_arrays=True
) as af:
    data = sml.Data(**af["data"]).astype(xp.Tensor, dtype=xp.float32)
    where = sml.Data(**af["data"]).astype(xp.Tensor, dtype=xp.bool)
    off_stream = np.array(af["off_stream"], dtype=bool)

# =============================================================================

fig, axs = plt.subplots(2, 1, figsize=(4, 4))

axs[0].plot(
    data["phi1"][off_stream],
    data["phi2"][off_stream],
    ls="none",
    marker=",",
    ms=1,
    color="k",
    rasterized=True,
)
axs[0].set_xlabel(r"$\phi_1 \ $ [$\degree$]")
axs[0].set_ylabel(r"$\phi_2 \ $ [$\degree$]")
axs[0].set_aspect(2)
axs[0].grid(visible=True, which="both", axis="y")
axs[0].grid(visible=True, which="major", axis="x")

axs[1].scatter(
    data["g-r"][off_stream], data["g"][off_stream], s=0.5, color="k", rasterized=True
)
axs[1].set_xlabel(r"$g-r \ $ [mag]")
axs[1].set_ylabel(r"$g \ $ [mag]")
axs[1].set_aspect("auto")
axs[1].grid(visible=True, which="major")

fig.tight_layout()
fig.savefig(paths.figures / "mock" / "photometric_background_selection.pdf")
