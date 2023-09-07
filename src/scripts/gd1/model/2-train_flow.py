"""Train photometry background flow."""

import sys
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

from scripts.gd1.datasets import data, off_stream
from scripts.gd1.define_model import (
    background_photometric_model as model,
)

# =============================================================================
# Load Data

snkmk: dict[str, Any]
try:
    snkmk = dict(snakemake.params)
except NameError:
    snkmk = {
        "load_from_static": True,
        "save_to_static": False,
        "epochs": 1_000,
    }

if snkmk["load_from_static"]:
    model.load_state_dict(xp.load(paths.static / "gd1" / "flow_model.pt"))
    xp.save(model.state_dict(), paths.data / "gd1" / "flow_model.pt")

    sys.exit(0)


figure_path = paths.figures / "gd1" / "diagnostic" / "flow"
figure_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Train

coord_names = model.indep_coord_names + model.coord_names
dataset = td.TensorDataset(data[coord_names].array[off_stream])
loader = td.DataLoader(dataset=dataset, batch_size=500, shuffle=True, num_workers=0)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Turn on gradients for training
object.__setattr__(model, "with_grad", True)

for epoch in tqdm(range(snkmk["epochs"])):
    for _step, (data_cur_,) in enumerate(loader):
        data_cur = sml.Data(data_cur_, names=coord_names)

        optimizer.zero_grad()

        mpars = model.unpack_params(model(data_cur))
        loss = -model.ln_posterior_tot(mpars, data_cur)

        loss.backward()
        optimizer.step()

        # Diagnostic plots (not in the paper)
        if snkmk["diagnostic_plots"] and (
            (epoch % 100 == 0) or (epoch == snkmk["epochs"] - 1)
        ):
            with xp.no_grad():
                mpars = model.unpack_params(model(data))
                prob = model.posterior(mpars, data)

            psort = np.argsort(prob[off_stream])

            fig, ax = plt.subplots()
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

# Turn off gradients for usage later
object.__setattr__(model, "with_grad", False)

xp.save(model.state_dict(), paths.data / "gd1" / "flow_model.pt")

if snkmk["save_to_static"]:
    xp.save(model.state_dict(), paths.static / "gd1" / "flow_model.pt")
