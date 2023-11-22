"""Train photometry background flow."""

import sys
from dataclasses import replace
from typing import Any

import galstreams
import matplotlib.pyplot as plt
import numpy as np
import torch as xp
import torch.utils.data as td
from showyourwork.paths import user as user_paths
from torch import optim
from tqdm import tqdm

import stream_ml.pytorch as sml

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.pal5.datasets import data, off_stream, where
from scripts.pal5.frames import pal5_frame as frame
from scripts.pal5.model import background_pm_model as model_without_grad

# =============================================================================
# Load Data

snkmk: dict[str, Any]
try:
    snkmk = dict(snakemake.params)
except NameError:
    snkmk = {
        "load_from_static": False,
        "save_to_static": False,
        "epochs": 2_000,
        "diagnostic_plots": True,
    }

if snkmk["load_from_static"] and (paths.static / "pal5").exists():
    # Load the model from the static directory
    model_without_grad.load_state_dict(
        xp.load(paths.static / "pal5" / "background_pm_model.pt")
    )
    # Save the model to the data directory
    xp.save(
        model_without_grad.state_dict(),
        paths.data / "pal5" / "background_pm_model.pt",
    )
    sys.exit(0)


figure_path = paths.scripts / "pal5" / "_diagnostics" / "pm_flow"
figure_path.mkdir(parents=True, exist_ok=True)

# Training steps save directory
(paths.data / "pal5" / "pm_flow").mkdir(parents=True, exist_ok=True)

if snkmk["diagnostic_plots"]:
    allstreams = galstreams.MWStreams(implement_Off=True)
    pal5PW19 = allstreams["Pal5-PW19"].track.transform_to(frame)
    pal5I21 = allstreams["Pal5-I21"].track.transform_to(frame)

# =============================================================================
# Train

# Make a copy of the model with gradients
# The network is shared with the original model
model = replace(model_without_grad, with_grad=True)

# Prerequisites for training
coord_names = model.indep_coord_names + model.coord_names

dataset = td.TensorDataset(
    data[coord_names].array[off_stream], where[coord_names].array[off_stream]
)
batch_size = int(off_stream.sum() * 0.05)
loader = td.DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Train
for epoch in tqdm(range(snkmk["epochs"])):
    # Iterate over batches
    for data_step_, where_step_ in loader:
        # Convert to sml.Data
        data_step = sml.Data(data_step_, names=coord_names)
        where_step = sml.Data(where_step_, names=coord_names)

        # Reset gradients
        optimizer.zero_grad()

        # Compute loss
        mpars = model.unpack_params(model(data_step))
        loss = -model.ln_posterior_tot(mpars, data_step, where=where_step)

        # Back-propogate
        loss.backward()
        optimizer.step()

    # Save
    if epoch % 100 == 0:
        xp.save(
            model.state_dict(), paths.data / "pal5" / "pm_flow" / f"model_{epoch:04}.pt"
        )
        xp.save(model.state_dict(), paths.data / "pal5" / "background_pm_model.pt")

    # -----------------------------------------------------------
    # Diagnostic plots (not in the paper)

    if snkmk["diagnostic_plots"] and (
        (epoch % 100 == 0) or (epoch == snkmk["epochs"] - 1)
    ):
        with xp.no_grad():
            mpars = model.unpack_params(model(data))
            prob = model.posterior(mpars, data, where=where)
        psort = np.argsort(prob[off_stream])

        fig = plt.figure()
        ax = fig.add_subplot(
            111,
            xlabel=r"$\mu_{\phi_1}^*$ [mas/yr]",
            ylabel=r"$\mu_{\phi_2}$ [mas/yr]",
        )
        ax.plot(
            data["pmphi1"][~off_stream],
            data["pmphi2"][~off_stream],
            ls="none",
            marker=",",
            c="black",
            label="on-stream",
        )
        im = ax.scatter(
            data["pmphi1"][off_stream][psort],
            data["pmphi2"][off_stream][psort],
            s=0.1,
            c=prob[off_stream][psort],
            label="off-stream",
        )

        ax.plot(
            pal5PW19.pm_phi1_cosphi2,
            pal5PW19.pm_phi2,
            label="Pal5-PW19",
            c="tab:red",
            lw=2,
        )
        ax.plot(
            pal5I21.pm_phi1_cosphi2,
            pal5I21.pm_phi2,
            label="Pal5-I21",
            c="tab:purple",
            lw=2,
        )

        plt.colorbar(im, ax=ax)
        ax.legend()
        fig.savefig(figure_path / f"epoch_{epoch:05}.png")
        plt.close(fig)

# =============================================================================
# Save

xp.save(model.state_dict(), paths.data / "pal5" / "background_pm_model.pt")

if snkmk["save_to_static"]:
    xp.save(model.state_dict(), paths.static / "pal5" / "background_pm_model.pt")
