"""Train parallax background flow."""

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
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.gd1.datasets_fullset import data, off_stream, where
from scripts.gd1.model import make_model

# =============================================================================
# Load Data

model_without_grad = make_model("fullset")["background"]["astrometric"]["plx"]

snkmk: dict[str, Any]
try:
    snkmk = dict(snakemake.params)
except NameError:
    snkmk = {
        "load_from_static": False,
        "save_to_static": False,
        "epochs": 1_000,
        "diagnostic_plots": True,
    }

save_path = paths.data / "gd1" / "fullset"
save_path.mkdir(parents=True, exist_ok=True)

static_path = paths.static / "gd1" / "fullset"
static_path.mkdir(parents=True, exist_ok=True)

if snkmk["load_from_static"]:
    model_without_grad.load_state_dict(
        xp.load(static_path / "background_parallax_model.pt")
    )
    xp.save(
        model_without_grad.state_dict(),
        save_path / "background_parallax_model.pt",
    )
    sys.exit(0)


figure_path = paths.scripts / "gd1" / "_diagnostics" / "fullset" / "plx_flow"
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
    for data_step_, where_step_ in loader:
        data_step = sml.Data(data_step_, names=coord_names)
        where_step = sml.Data(where_step_, names=coord_names)

        optimizer.zero_grad()

        mpars = model.unpack_params(model(data_step))
        loss = -model.ln_posterior_tot(mpars, data_step, where=where_step)

        loss.backward()
        optimizer.step()

        # Diagnostic plots (not in the paper)
        if snkmk["diagnostic_plots"] and (
            (epoch % 100 == 0) or (epoch == snkmk["epochs"] - 1)
        ):
            with xp.no_grad():
                mpars = model.unpack_params(model(data))
                prob = model.posterior(mpars, data, where=where).flatten()

            psort = np.argsort(prob[off_stream])

            fig = plt.figure()
            ax = fig.add_subplot(
                ylim=(data["plx"].min(), data["plx"].max()), rasterization_zorder=0
            )
            im = ax.hexbin(
                data["phi1"][off_stream][psort],
                data["plx"][off_stream][psort],
                C=prob[off_stream][psort],
                zorder=-10,
            )
            plt.colorbar(im, ax=ax)
            fig.savefig(figure_path / f"epoch_{epoch:05}.png")
            plt.close(fig)

xp.save(model.state_dict(), save_path / "background_parallax_model.pt")

if snkmk["save_to_static"]:
    xp.save(model.state_dict(), static_path / "background_parallax_model.pt")
