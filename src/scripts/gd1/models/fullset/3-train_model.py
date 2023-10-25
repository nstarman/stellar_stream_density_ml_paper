"""Train GD-1 model."""

import sys
from typing import Any

import matplotlib.pyplot as plt
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

from scripts.gd1.datasets_fullset import data, where
from scripts.gd1.model import make_model
from scripts.gd1.models.helper import diagnostic_plot
from scripts.helper import manually_set_dropout

# =============================================================================
# Initial set up

snkmk: dict[str, Any]
try:
    snkmk = dict(snakemake.params)
except NameError:
    snkmk = {
        "load_from_static": False,
        "save_to_static": False,
        "diagnostic_plots": True,
        # hyperparameters
        "lr": 1e-3,
        "weight_decay": 1e-8,
        "eta_min": 1e-4,
        # epoch parameters
        "T_burn": 200,
        "T_cos": 300,
        "n_T": 3,
        "T_endburn": 10_000 - 200 - 300 * 3,
    }
    snkmk["epochs"] = (
        snkmk["T_burn"] + snkmk["T_cos"] * snkmk["n_T"] + snkmk["T_endburn"]
    )

save_path = paths.data / "gd1" / "fullset"
save_path.mkdir(parents=True, exist_ok=True)
(save_path / "models").mkdir(parents=True, exist_ok=True)

static_path = paths.static / "gd1" / "fullset"
static_path.mkdir(parents=True, exist_ok=True)

model = make_model("fullset")

if snkmk["load_from_static"]:
    model.load_state_dict(xp.load(static_path / "model.pt"))
    xp.save(model.state_dict(), save_path / "model.pt")

    sys.exit(0)

diagnostic_path = paths.scripts / "gd1" / "_diagnostics" / "fullset" / "models"
diagnostic_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Load saved model components

model.load_state_dict(xp.load(paths.data / "gd1" / "subset" / "model.pt"))
# overload with better background models
model["background"]["astrometric"]["plx"].load_state_dict(
    xp.load(save_path / "background_parallax_model.pt")
)
model["background"]["photometric"].load_state_dict(
    xp.load(save_path / "background_photometry_model.pt")
)


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

# scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=0.1)  # constant
# 1e-4
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer,
    [
        optim.lr_scheduler.ConstantLR(
            optimizer, 0.2, total_iters=snkmk["T_burn"] - 1
        ),  # converge
        optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=snkmk["T_cos"], eta_min=snkmk["eta_min"]
        ),
        optim.lr_scheduler.ConstantLR(
            optimizer, 0.1, total_iters=snkmk["T_endburn"]
        ),  # converge
    ],
    milestones=[
        snkmk["T_burn"],
        snkmk["T_burn"] + snkmk["n_T"] * snkmk["T_cos"],
    ],
)

# =============================================================================
# Train

num_steps = len(loader.dataset) // loader.batch_size
epoch: int = 0
epoch_iterator = tqdm(
    range(snkmk["epochs"]),
    dynamic_ncols=True,
    postfix={"lr": f"{scheduler.get_last_lr()[0]:.2e}", "loss": f"{0:.2e}"},
)
for epoch in epoch_iterator:
    for step_arr, step_where_ in loader:
        # Prepare
        step_data = sml.Data(step_arr, names=data.names)
        step_where = sml.Data(step_where_, names=data.names)

        # Forward Step
        pred = model(step_data)
        if not pred.isfinite().all():
            raise ValueError

        # Evaluate loss function
        mpars = model.unpack_params(pred)
        loss_val = -model.ln_posterior_tot(mpars, step_data, where=step_where)

        if loss_val.isnan().any():
            raise ValueError

        # backward pass
        optimizer.zero_grad()
        loss_val.backward()

        # update weights
        optimizer.step()
        model.zero_grad()

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

        fig = diagnostic_plot(model, data, where=where)
        fig.savefig(diagnostic_path / f"epoch_{epoch:05}.png")
        plt.close(fig)

        # turn dropout on
        manually_set_dropout(model, 0.15)
        model.train()

        xp.save(model.state_dict(), save_path / "model.pt")

        xp.save(
            model.state_dict(),
            save_path / "models" / f"model_{epoch:04d}.pt",
        )


# Save final state of the model
xp.save(model.state_dict(), save_path / "models" / f"model_{epoch:04d}.pt")


# =============================================================================
# Save

xp.save(model.state_dict(), save_path / "model.pt")

if snkmk["save_to_static"]:
    xp.save(model.state_dict(), static_path / "model.pt")
