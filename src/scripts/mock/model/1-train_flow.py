"""Train photometry background flow."""

import sys
from pathlib import Path
from typing import Any

import asdf
import matplotlib.pyplot as plt
import torch as xp
import torch.utils.data as td
from torch import optim
from tqdm import tqdm

import stream_ml.pytorch as sml

# Add the parent directory to the path
sys.path.append(Path(__file__).parents[3].as_posix())
# isort: split

from scripts import paths
from scripts.mock.define_model import bkg_flow as model
from scripts.mock.define_model import flow_coords

# =============================================================================
# Load Data

snkmk: dict[str, Any]
try:
    snkmk = dict(snakemake.params)
except NameError:
    snkmk = {
        "load_from_static": False,
        "save_to_static": False,
        "diagnostic_plots": True,
        "epochs": 400,
        "batch_size": 500,
        "lr": 1e-3,
    }

with asdf.open(
    paths.data / "mock" / "data.asdf", lazy_load=False, copy_arrays=True
) as af:
    table = af["table"]

    data = sml.Data(**af["data"]).astype(xp.Tensor, dtype=xp.float32)
    where = sml.Data(**af["data"]).astype(xp.Tensor, dtype=xp.bool)
    scaler = sml.utils.StandardScaler(**af["scaler"]).astype(
        xp.Tensor, dtype=xp.float32
    )

    off_stream = af["off_stream"]


if snkmk["load_from_static"]:
    model.load_state_dict(xp.load(paths.static / "mock" / "flow_model.pt"))
    xp.save(model.state_dict(), paths.data / "mock" / "flow_model.pt")

    sys.exit(0)

figure_path = paths.figures / "mock" / "diagnostic" / "flow"
figure_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Training Parameters

loader = td.DataLoader(
    dataset=td.TensorDataset(data[flow_coords].array[off_stream]),
    batch_size=snkmk["batch_size"],
    shuffle=True,
    num_workers=0,
)

optimizer = optim.AdamW(list(model.parameters()), lr=snkmk["lr"])


# =============================================================================
# Train

# Turn on gradients for training
object.__setattr__(model, "with_grad", True)

for epoch in tqdm(range(snkmk["epochs"])):
    for _step, (data_cur_,) in enumerate(loader):
        data_cur = sml.Data(data_cur_, names=flow_coords)

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

            fig, ax = plt.subplots()
            im = ax.scatter(data["g-r"], data["g"], s=0.2, c=prob)
            plt.colorbar(im, ax=ax)
            fig.savefig(figure_path / f"epoch_{epoch:05}.png")
            plt.close(fig)

# Turn off gradients for running
object.__setattr__(model, "with_grad", False)

xp.save(model.state_dict(), paths.data / "mock" / "flow_model.pt")

if snkmk["save_to_static"]:
    xp.save(model.state_dict(), paths.static / "mock" / "flow_model.pt")
