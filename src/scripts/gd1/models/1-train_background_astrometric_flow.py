"""Train parallax background flow."""

import sys
from dataclasses import replace
from typing import Any

import galstreams
import matplotlib.pyplot as plt
import numpy as np
import torch as xp
import torch.utils.data as td
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from scipy.interpolate import InterpolatedUnivariateSpline
from showyourwork.paths import user as user_paths
from torch import optim
from tqdm import tqdm

import stream_mapper.pytorch as sml
from stream_mapper.core import Data

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.gd1.datasets import data, off_stream, where
from scripts.gd1.frames import gd1_frame
from scripts.gd1.model import make_model
from scripts.mpl_colormaps import stream_cmap1 as cmap_stream

model_without_grad = make_model()["background"]["astrometric"]["else"]

# =============================================================================
# Load Data

snkmk: dict[str, Any]
try:
    snkmk = dict(snakemake.params)
except NameError:
    snkmk = {
        "load_from_static": False,
        "save_to_static": False,
        "epochs": 2_500,
        "diagnostic_plots": True,
    }

save_path = paths.data / "gd1"
save_path.mkdir(parents=True, exist_ok=True)
(save_path / "astro_flow").mkdir(parents=True, exist_ok=True)

static_path = paths.static / "gd1"
static_path.mkdir(parents=True, exist_ok=True)

if snkmk["load_from_static"] and (paths.static / "gd1").exists():
    # Load from static
    model_without_grad.load_state_dict(
        xp.load(static_path / "background_astrometric_model.pt")
    )

    # Save to data
    xp.save(
        model_without_grad.state_dict(),
        save_path / "background_astrometric_model.pt",
    )

    sys.exit(0)

figure_path = paths.scripts / "gd1" / "_diagnostics" / "astro_flow"
figure_path.mkdir(parents=True, exist_ok=True)


mws = galstreams.MWStreams()
gd1 = mws["GD-1-I21"]
gd1_sc = gd1.track.transform_to(gd1_frame)[::100]

spline_pmphi1 = InterpolatedUnivariateSpline(
    gd1_sc.phi1.value, gd1_sc.pm_phi1_cosphi2.value
)
spline_pmphi2 = InterpolatedUnivariateSpline(gd1_sc.phi1.value, gd1_sc.pm_phi2.value)

# =============================================================================


def diagnostic_plot(data: Data, prob: np.ndarray) -> plt.Figure:
    """Diagnostic plot.

    Parameters
    ----------
    data : Data
        The data.
    prob : np.ndarray
        The probability.

    Returns
    -------
    plt.Figure
        The figure.

    """
    psort = np.argsort(prob[off_stream])

    fig = plt.figure(figsize=(5, 13))

    gs = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[0.2, 10],
        hspace=0.01,
        left=0.15,
        right=0.98,
        top=0.965,
        bottom=0.03,
    )
    gs0 = gs[1].subgridspec(3, 1, hspace=0.04)

    ax0 = fig.add_subplot(gs[0])
    cbar = fig.colorbar(
        ScalarMappable(cmap=cmap_stream), cax=ax0, orientation="horizontal"
    )
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.text(0.5, 0.5, "Probability", ha="center", va="center", fontsize=12)

    ax1 = fig.add_subplot(
        gs0[0],
        xticklabels=[],
        ylabel=r"$\varpi$",
        ylim=(data["plx"].min(), data["plx"].max()),
    )
    ax1.scatter(data["phi1"][~off_stream], data["plx"][~off_stream], s=0.2, c="black")
    ax1.scatter(
        data["phi1"][off_stream][psort],
        data["plx"][off_stream][psort],
        s=0.2,
        c=prob[off_stream][psort],
        cmap=cmap_stream,
    )

    ax2 = fig.add_subplot(gs0[1], xticklabels=[], ylabel=r"$\mu_{\phi_1^*}$ - GD1")
    ax2.scatter(
        data["phi1"][~off_stream],
        data["pmphi1"][~off_stream] - spline_pmphi1(data["phi1"][~off_stream]),
        s=0.2,
        c="black",
    )
    ax2.scatter(
        data["phi1"][off_stream][psort],
        data["pmphi1"][off_stream][psort]
        - spline_pmphi1(data["phi1"][off_stream][psort]),
        s=0.2,
        c=prob[off_stream][psort],
        cmap=cmap_stream,
    )

    ax3 = fig.add_subplot(gs0[2], xlabel=r"$\phi_1$", ylabel=r"$\mu_{\phi_2}$ - GD1")
    ax3.scatter(
        data["phi1"][~off_stream],
        data["pmphi2"][~off_stream] - spline_pmphi2(data["phi1"][~off_stream]),
        s=0.2,
        c="black",
    )
    ax3.scatter(
        data["phi1"][off_stream][psort],
        data["pmphi2"][off_stream][psort]
        - spline_pmphi2(data["phi1"][off_stream][psort]),
        s=0.2,
        c=prob[off_stream][psort],
        cmap=cmap_stream,
    )
    return fig


# =============================================================================
# Train

# Make a copy of the model with gradients
# The network is shared with the original model
model = replace(model_without_grad, with_grad=True)

# Prerequisites for training
coord_names = model.indep_coord_names + model.coord_names
dataset = td.TensorDataset(
    data[coord_names].array[off_stream],
    where[coord_names].array[off_stream],
)
loader = td.DataLoader(
    dataset=dataset,
    batch_size=int(len(data) * 0.075),
    shuffle=True,
    num_workers=0,
    drop_last=True,
)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
optimizer.zero_grad()

# Train
for epoch in tqdm(range(snkmk["epochs"])):
    # Train batches
    for data_step_, where_step_ in loader:
        # Step
        data_step = sml.Data(data_step_, names=coord_names)
        where_step = sml.Data(where_step_, names=coord_names)

        # Reset gradients
        optimizer.zero_grad()

        # Compute loss
        mpars = model.unpack_params(model(data_step))
        loss = -model.ln_posterior_tot(mpars, data_step, where=where_step)

        # Backprop
        loss.backward()
        optimizer.step()

    # Save
    if (epoch % 100 == 0) or (epoch == snkmk["epochs"] - 1):
        xp.save(model.state_dict(), save_path / "astro_flow" / f"model_{epoch:05}.pt")
        xp.save(model.state_dict(), save_path / "background_astrometric_model.pt")

        # Diagnostic plots (not in the paper)
        if snkmk["diagnostic_plots"]:
            with xp.no_grad():
                mpars = model.unpack_params(model(data))
                prob = model.posterior(mpars, data, where=where).flatten()

            fig = diagnostic_plot(data, prob)
            fig.savefig(figure_path / f"epoch_{epoch:05}.png")
            plt.close(fig)


# ========================================================================
# Save

xp.save(model.state_dict(), save_path / "background_astrometric_model.pt")

if snkmk["save_to_static"]:
    xp.save(model.state_dict(), static_path / "background_astrometric_model.pt")
