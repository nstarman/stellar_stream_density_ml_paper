"""Train photometry background flow."""

import sys
from dataclasses import replace
from typing import Any

import asdf
import matplotlib.pyplot as plt
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

from scripts.mock2.model import bkg_flow as model_without_grad
from scripts.mock2.model import flow_coords

# =============================================================================
# Parameters

snkmk: dict[str, Any]
try:
    snkmk = dict(snakemake.params)
except NameError:
    snkmk = {
        "load_from_static": True,
        "save_to_static": True,
        "diagnostic_plots": True,
        "epochs": 2_000,
        "lr": 1e-3,
    }


if snkmk["load_from_static"] and (paths.static / "mock2").exists():
    model_without_grad.load_state_dict(
        xp.load(paths.static / "mock2" / "background_photometry_model.pt")
    )
    xp.save(
        model_without_grad.state_dict(),
        paths.data / "mock2" / "background_photometry_model.pt",
    )

    sys.exit(0)


# =============================================================================
# Load Data

with asdf.open(
    paths.data / "mock2" / "data.asdf", lazy_load=False, copy_arrays=True
) as af:
    table = af["table"]

    data = sml.Data(**af["data"]).astype(xp.Tensor, dtype=xp.float32)
    where = sml.Data(**af["where"]).astype(xp.Tensor, dtype=bool)
    scaler = sml.utils.StandardScaler(**af["scaler"]).astype(
        xp.Tensor, dtype=xp.float32
    )

    off_stream = af["off_stream"]

    stream_table = af["stream_table"]

figure_path = paths.scripts / "mock2" / "_diagnostics" / "flow"
figure_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Training Parameters

# Turn on gradients for training
model = replace(model_without_grad, with_grad=True)

data_offstream: sml.Data = data[off_stream]
where_offstream: sml.Data = where[off_stream]

loader = td.DataLoader(
    dataset=td.TensorDataset(
        data_offstream[flow_coords].array,
        where_offstream[flow_coords].array,
    ),
    batch_size=int(len(data_offstream) * 0.05),
    shuffle=True,
    num_workers=0,
)

optimizer = optim.AdamW(list(model.parameters()), lr=snkmk["lr"])


# =============================================================================
# Train

for epoch in tqdm(range(snkmk["epochs"])):
    # Iterate over batches
    for data_step_, where_step_ in loader:
        data_step = sml.Data(data_step_, names=flow_coords)
        where_step = sml.Data(where_step_, names=flow_coords)

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
                mpars = model.unpack_params(model(data_offstream))
                prob = model.posterior(mpars, data_offstream, where=where_offstream)

            fig, ax = plt.subplots()
            im = ax.scatter(data_offstream["g-r"], data_offstream["g"], s=0.2, c=prob)
            ax.scatter(stream_table["g-r"].value, stream_table["g"].value, s=0.1, c="k")
            plt.colorbar(im, ax=ax)
            fig.savefig(figure_path / f"epoch_{epoch:05}.png")
            plt.close(fig)


# =============================================================================
# Save

xp.save(model.state_dict(), paths.data / "mock2" / "background_photometry_model.pt")

if snkmk["save_to_static"]:
    xp.save(
        model.state_dict(), paths.static / "mock2" / "background_photometry_model.pt"
    )
