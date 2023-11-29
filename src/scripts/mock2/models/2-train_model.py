"""Train photometry background flow."""

import sys
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

import contextlib

from scripts import helper
from scripts.mock2.model import model
from scripts.mock2.models.helper import diagnostic_plot

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
        "T1": 2_100,
        "T2": 2_100,
        "T3": 2_100,
        "eta_min": 1e-4,
        "lr": 2e-3,
    }
snkmk["epochs"] = snkmk["T1"] + snkmk["T2"] + snkmk["T3"]


if snkmk["load_from_static"] and (paths.static / "mock2" / "model.pt").exists():
    model.load_state_dict(xp.load(paths.static / "mock2" / "model.pt"))
    xp.save(model.state_dict(), paths.data / "mock2" / "model.pt")

    sys.exit(0)

# =============================================================================
# Load Data

with asdf.open(
    paths.data / "mock2" / "data.asdf", lazy_load=False, copy_arrays=True
) as af:
    data = sml.Data(**af["data"]).astype(xp.Tensor, dtype=xp.float32)
    where = sml.Data(**af["where"]).astype(xp.Tensor, dtype=xp.bool)
    scaler = sml.utils.StandardScaler(**af["scaler"]).astype(
        xp.Tensor, dtype=xp.float32
    )
    table = af["table"]

# ensure the folders exist
(paths.data / "mock2").mkdir(exist_ok=True, parents=True)
(paths.static / "mock2").mkdir(exist_ok=True, parents=True)
diagnostic_path = paths.scripts / "mock2" / "_diagnostics" / "models"
diagnostic_path.mkdir(parents=True, exist_ok=True)
(paths.data / "mock2" / "models").mkdir(exist_ok=True, parents=True)

# =============================================================================
# Training Parameters

# Pre-trained Model component
with contextlib.suppress(KeyError):
    model["background"]["photometric"].load_state_dict(
        xp.load(paths.data / "mock2" / "background_photometry_model.pt")
    )

dataset = td.TensorDataset(data.array, where.array)
loader = td.DataLoader(
    dataset=dataset,
    batch_size=int(0.075 * len(data)),  # 7.5% of the data
    shuffle=True,
    num_workers=0,
)
optimizer = optim.AdamW(list(model.parameters()), lr=snkmk["lr"])

start_epoch = 0
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer,
    [
        optim.lr_scheduler.ConstantLR(optimizer, 1, total_iters=snkmk["T1"]),
        optim.lr_scheduler.ConstantLR(optimizer, 0.25, total_iters=snkmk["T2"]),
        optim.lr_scheduler.ConstantLR(optimizer, 0.05, total_iters=snkmk["T3"]),
    ],
    milestones=[snkmk["T1"], snkmk["T1"] + snkmk["T2"]],
    last_epoch=start_epoch,
)

epoch_iterator = tqdm(
    range(start_epoch, snkmk["epochs"]),
    dynamic_ncols=True,
    postfix={"lr": f"{scheduler.get_last_lr()[0]:.2e}", "loss": f"{0:.2e}"},
)
# load the model at the start epoch
if start_epoch > 0:
    model.load_state_dict(
        xp.load(paths.data / "mock2" / "models" / f"epoch_{start_epoch:05}.pt")
    )

# =============================================================================
# Train

lrs = []

model.train()
model.zero_grad()
for epoch in epoch_iterator:
    # Iterate over batches
    for data_step_, where_step_ in loader:
        data_step = sml.Data(data_step_, names=data.names)
        where_step = sml.Data(where_step_, names=data.names)

        pred = model(data_step)
        mpars = model.unpack_params(pred)
        loss = -model.ln_posterior(mpars, data_step, where=where_step).mean()

        if loss.isnan().any():
            raise ValueError

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        model.zero_grad()

    scheduler.step()

    epoch_iterator.set_postfix(
        {"lr": f"{scheduler.get_last_lr()[0]:.2e}", "loss": f"{loss:.2e}"}
    )
    lrs.append(scheduler.get_last_lr()[0])

    # Save
    if (epoch % 100 == 0) or (epoch == snkmk["epochs"] - 1):
        # up-to-date model
        xp.save(
            model.state_dict(), paths.data / "mock2" / "models" / f"epoch_{epoch:05}.pt"
        )
        xp.save(model.state_dict(), paths.data / "mock2" / "model.pt")

        # Diagnostic plots (not in the paper)
        if snkmk["diagnostic_plots"]:
            helper.manually_set_dropout(model, 0)

            with xp.no_grad():
                mpars = model.unpack_params(model(data))
                prob = model.posterior(mpars, data, where=where)

            plt.close("all")
            fig = diagnostic_plot(model, data, where, table)
            fig.savefig(diagnostic_path / f"epoch_{epoch:05}.png")
            plt.close(fig)

            helper.manually_set_dropout(model, 0.15)

    if (epoch % 20 == 0) or (epoch == snkmk["epochs"] - 1):
        # Plot the learning rate
        fig, ax = plt.subplots()
        ax.plot(lrs)
        fig.savefig(diagnostic_path / "lr.png")


# =============================================================================
# Save

xp.save(model.state_dict(), paths.data / "mock2" / "model.pt")

if snkmk["save_to_static"]:
    xp.save(model.state_dict(), paths.static / "mock2" / "model.pt")
