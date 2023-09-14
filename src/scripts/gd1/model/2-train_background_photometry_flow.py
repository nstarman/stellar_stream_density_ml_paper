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

import stream_ml.pytorch as sml

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.as_posix())
# isort: split

from scripts.gd1.datasets import data, off_stream, where
from scripts.gd1.define_model import (
    background_photometric_model as model_without_grad,
)

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

if snkmk["load_from_static"]:
    model_without_grad.load_state_dict(
        xp.load(paths.static / "gd1" / "background_photometry_model.pt")
    )
    xp.save(
        model_without_grad.state_dict(),
        paths.data / "gd1" / "background_photometry_model.pt",
    )
    sys.exit(0)


figure_path = paths.figures / "gd1" / "diagnostic" / "phot_flow"
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
    dataset=dataset, batch_size=500, shuffle=True, num_workers=0, drop_last=True
)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Train
for epoch in tqdm(range(snkmk["epochs"])):
    for data_step_, where_step_ in loader:
        data_step = sml.Data(data_step_, names=coord_names)
        where_step = sml.Data(where_step_, names=coord_names)

        optimizer.zero_grad()

        mpars = model.unpack_params(model(data_step))
        loss = -model.ln_posterior_tot(mpars, data_step, where=where_step)

        loss.backward()
        optimizer.step()

    xp.save(model.state_dict(), paths.data / "gd1" / "background_photometry_model.pt")

    # -----------------------------------------------------------

    # Diagnostic plots (not in the paper)
    if snkmk["diagnostic_plots"] and (
        (epoch % 100 == 0) or (epoch == snkmk["epochs"] - 1)
    ):
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
        ax.set(xlim=(0, 0.8), ylim=(22, 13.5))
        fig.savefig(figure_path / f"epoch_{epoch:05}.png")
        plt.close(fig)

xp.save(model.state_dict(), paths.data / "gd1" / "background_photometry_model.pt")

if snkmk["save_to_static"]:
    xp.save(model.state_dict(), paths.static / "gd1" / "background_photometry_model.pt")
