"""Train Pal5 model."""

import sys
from typing import Any

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

from scripts.helper import manually_set_dropout
from scripts.pal5.datasets import data, where
from scripts.pal5.model import model
from scripts.pal5.models.helper import diagnostic_plot

# =============================================================================
# Parameters

snkmk: dict[str, Any]
try:
    snkmk = dict(snakemake.params)
except NameError:
    snkmk = {
        "load_from_static": True,
        "save_to_static": False,
        "diagnostic_plots": True,
        # epoch milestones
        "epochs": 1_250 * 10,
        "lr": 1e-4,
        "weight_decay": 1e-8,
        "early_stopping": -1,
    }

save_path = paths.data / "pal5"
save_path.mkdir(parents=True, exist_ok=True)
(save_path / "models").mkdir(parents=True, exist_ok=True)

if snkmk["load_from_static"] and (paths.static / "pal5").exists():
    model.load_state_dict(xp.load(paths.static / "pal5" / "model.pt"))
    xp.save(model.state_dict(), save_path / "model.pt")

    sys.exit(0)

diagnostic_path = paths.scripts / "pal5" / "_diagnostics" / "models"
diagnostic_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Load saved model components

model["background"]["astrometric"]["pm"].load_state_dict(
    xp.load(save_path / "background_pm_model.pt")
)
# model.load_state_dict(xp.load(save_path / "model.pt"))  # start from most recent save

# =============================================================================
# Training Parameters

BATCH_SIZE = int(len(data) * 0.05)

dataset = td.TensorDataset(
    data.array,  # data
    where.array,  # TRUE where NOT missing
)

loader = td.DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True,  # drop rando last for better plotting
)

optimizer = optim.AdamW(
    list(model.parameters()), lr=snkmk["lr"], weight_decay=snkmk["weight_decay"]
)

scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1)  # constant 1e-4

# =============================================================================
# Train

num_steps = len(loader.dataset) // loader.batch_size
epoch: int = 0
epoch_iterator = tqdm(
    range(epoch, snkmk["epochs"]),
    dynamic_ncols=True,
    postfix={"lr": f"{scheduler.get_last_lr()[0]:.2e}", "loss": f"{0:.2e}"},
)
for epoch in epoch_iterator:
    # Step through batches
    for step_arr, step_where_ in loader:
        # Prepare
        step_data = sml.Data(step_arr, names=data.names)
        step_where = sml.Data(step_where_, names=data.names)

        # Forward Step
        pred = model(step_data)
        mpars = model.unpack_params(pred)
        if not pred.isfinite().all():
            raise ValueError

        # Evaluate loss
        loss_val = -model.ln_posterior_tot(mpars, step_data, where=step_where)
        if loss_val.isnan().any():
            raise ValueError

        # backward pass
        optimizer.zero_grad()
        loss_val.backward()

        # update weights
        optimizer.step()
        model.zero_grad()

    # step the scheduler
    scheduler.step()
    epoch_iterator.set_postfix(
        {"lr": f"{scheduler.get_last_lr()[0]:.2e}", "loss": f"{loss_val:.2e}"}
    )

    if snkmk["diagnostic_plots"] and (
        (epoch % 100 == 0) or (epoch == snkmk["epochs"] - 1)
    ):
        # turn dropout off
        model.eval()
        manually_set_dropout(model, 0)

        # Plot diagnostics
        fig = diagnostic_plot(model, data, where=where)
        fig.savefig(diagnostic_path / f"epoch_{epoch:05}.png")
        plt.close(fig)

        # turn dropout back on
        manually_set_dropout(model, 0.15)
        model.train()

        # Save state of the model
        xp.save(model.state_dict(), save_path / "model.pt")
        xp.save(model.state_dict(), save_path / "models" / f"model_{epoch:04d}.pt")


# Save final state of the model
xp.save(model.state_dict(), save_path / "models" / f"model_{epoch:04d}.pt")


# =============================================================================
# Save
# We institute early stopping to determine the "best" state of the model.

# Load the early stopping model
if snkmk["early_stopping"] > 0:
    model.load_state_dict(
        xp.load(save_path / "models" / f"model_{snkmk['early_stopping']:04d}.pt")
    )

# Save the model
xp.save(model.state_dict(), save_path / "model.pt")

if snkmk["save_to_static"]:
    xp.save(model.state_dict(), paths.static / "pal5" / "model.pt")
