"""Train photometry background flow."""

import asdf
import matplotlib.pyplot as plt
import numpy as np
import torch as xp
from showyourwork.paths import user as user_paths

import stream_mapper.pytorch as sml

paths = user_paths()


# =============================================================================

plt.style.use(paths.scripts / "paper.mplstyle")


with asdf.open(
    paths.data / "mock" / "data.asdf", lazy_load=False, copy_arrays=True
) as af:
    data = sml.Data(**af["data"]).astype(xp.Tensor, dtype=xp.float32)
    where = sml.Data(**af["where"]).astype(xp.Tensor, dtype=xp.bool)
    off_stream = np.array(af["off_stream"], dtype=bool)

# =============================================================================

fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# Plot the astrometric background selection
axs[0].plot(
    data["phi1"][off_stream],
    data["phi2"][off_stream],
    ls="none",
    marker=",",
    ms=1,
    color="k",
)
axs[0].plot(
    data["phi1"][~off_stream],
    data["phi2"][~off_stream],
    ls="none",
    marker=",",
    ms=1,
    color="tab:blue",
    alpha=0.25,
)
axs[0].set_xlabel(r"$\phi_1 \ $ [$\degree$]")
axs[0].set_ylabel(r"$\phi_2 \ $ [$\degree$]")
axs[0].grid(visible=True, which="both", axis="y")
axs[0].grid(visible=True, which="major", axis="x")
axs[0].set_rasterization_zorder(100)

# Plot the photometric background selection
axs[1].scatter(data["g-r"][off_stream], data["g"][off_stream], s=0.5, color="k")
axs[1].scatter(
    data["g-r"][~off_stream],
    data["g"][~off_stream],
    s=0.5,
    color="tab:blue",
    alpha=0.25,
)
axs[1].set_xlabel(r"$g-r \ $ [mag]")
axs[1].set_ylabel(r"$g \ $ [mag]")
axs[1].set_aspect("equal")
axs[1].grid(visible=True, which="major")
axs[1].set_rasterization_zorder(100)

fig.tight_layout()
fig.savefig(paths.figures / "mock" / "photometric_background_selection.pdf")
