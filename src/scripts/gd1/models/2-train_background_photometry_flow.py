"""Train photometry background flow."""

import sys
from dataclasses import replace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch as xp
import torch.utils.data as td
from showyourwork.paths import user as user_paths
from torch import optim
from tqdm import tqdm

import stream_mapper.pytorch as sml

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.gd1.datasets import data, off_stream, where
from scripts.gd1.model import make_model

model_without_grad = make_model()["background"]["photometric"]

# =============================================================================
# Load Data

snkmk: dict[str, Any]
try:
    snkmk = dict(snakemake.params)
except NameError:
    snkmk = {
        "load_from_static": True,
        "save_to_static": False,
        "epochs": 2_000,
        "diagnostic_plots": True,
    }

save_path = paths.data / "gd1"
save_path.mkdir(parents=True, exist_ok=True)
(save_path / "phot_flow").mkdir(parents=True, exist_ok=True)

static_path = paths.static / "gd1"
static_path.mkdir(parents=True, exist_ok=True)

if snkmk["load_from_static"] and (paths.static / "gd1").exists():
    model_without_grad.load_state_dict(
        xp.load(static_path / "background_photometry_model.pt")
    )
    xp.save(
        model_without_grad.state_dict(),
        save_path / "background_photometry_model.pt",
    )
    sys.exit(0)


figure_path = paths.scripts / "gd1" / "_diagnostics" / "phot_flow"
figure_path.mkdir(parents=True, exist_ok=True)

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

# Train
for epoch in tqdm(range(snkmk["epochs"])):
    # Train batches
    for data_step_, where_step_ in loader:
        # Step data
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

    # -----------------------------------------------------------

    if (epoch % 100 == 0) or (epoch == snkmk["epochs"] - 1):
        xp.save(model.state_dict(), save_path / "phot_flow" / f"model_{epoch:05}.pt")
        xp.save(model.state_dict(), save_path / "background_photometry_model.pt")

        # Diagnostic plots (not in the paper)
        if snkmk["diagnostic_plots"]:
            with xp.no_grad():
                mpars = model.unpack_params(model(data))
                prob = model.posterior(mpars, data, where=where)

            psort = np.argsort(prob[off_stream])

            fig, ax = plt.subplots()
            ax.scatter(
                (data["g"] - data["r"])[~off_stream],
                data["g"][~off_stream],
                s=0.2,
                c="black",
            )
            im = ax.scatter(
                (data["g"] - data["r"])[off_stream][psort],
                data["g"][off_stream][psort],
                s=0.2,
                c=prob[off_stream][psort],
            )
            plt.colorbar(im, ax=ax)
            ax.set(xlim=(0, 0.8), ylim=(21, 13.5))
            fig.savefig(figure_path / f"epoch_{epoch:05}.png")
            plt.close(fig)


# =============================================================================
# Save

xp.save(model.state_dict(), save_path / "background_photometry_model.pt")

if snkmk["save_to_static"]:
    xp.save(model.state_dict(), static_path / "background_photometry_model.pt")
