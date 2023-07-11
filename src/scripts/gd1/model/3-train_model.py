"""Train photometry background flow."""

import sys
from pathlib import Path

import torch as xp
import torch.utils.data as td
from torch import optim
from tqdm import tqdm

import stream_ml.pytorch as sml

# isort: split
from data import data, where
from define_model import model

# Add the parent directory to the path
sys.path.append(Path(__file__).parent.parent.as_posix())
# isort: split

import paths  # noqa: E402

# =============================================================================

try:
    snkmkp = snakemake.params
except NameError:
    snkmkp = {
        "load_from_static": False,
        "epochs": 1_000,
    }

if snkmkp["load_from_static"]:
    model.load_state_dict(xp.load(paths.static / "gd1" / "model.pt"))
    xp.save(model.state_dict(), paths.data / "gd1" / "model.pt")

    sys.exit(0)


# =============================================================================
# Training Parameters

EPOCHS = 1250 * 10
BATCH_SIZE = int(len(data) * 0.075)

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-8

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

optimizer = optim.Adam(
    list(model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=0.1)  # constant 1e-4


# =============================================================================
# Train

num_steps = len(loader.dataset) // loader.batch_size
epoch: int = 0
epoch_iterator = tqdm(
    range(EPOCHS),
    dynamic_ncols=True,
    postfix={"lr": f"{scheduler.get_last_lr()[0]:.2e}", "loss": f"{0:.2e}"},
)
for _epoch in epoch_iterator:
    for _step, (step_arr, step_where_) in enumerate(loader):
        # Prepare
        step_data = sml.Data(step_arr, names=data.names)
        step_where = sml.Data(step_where_, names=data.names[1:])

        # Forward Step
        pred = model(step_data)
        if pred.isnan().any():
            raise ValueError

        mpars = model.unpack_params(pred)
        loss_val = -model.ln_posterior_tot(
            mpars,
            step_data,
            stream_astrometric_where=step_where,
            spur_astrometric_where=step_where,
            background_astrometric_where=step_where,
        )

        # backward pass
        optimizer.zero_grad()
        loss_val.backward()

        # update weights
        optimizer.step()
        model.zero_grad()  # ?

    scheduler.step()
    epoch_iterator.set_postfix(
        {"lr": f"{scheduler.get_last_lr()[0]:.2e}", "loss": f"{loss_val:.2e}"}
    )

xp.save(model.state_dict(), paths.static / "gd1" / "model.pt")
