"""Train photometry background flow."""

import sys
from pathlib import Path

import asdf
import matplotlib.pyplot as plt
import torch as xp
import torch.utils.data as td
from torch import optim
from tqdm import tqdm

import stream_ml.pytorch as sml

# isort: split
from define_model import bkg_flow as model
from define_model import flow_coords

# Add the parent directory to the path
sys.path.append(Path(__file__).parent.parent.parent.as_posix())
# isort: split

import paths  # noqa: E402

# =============================================================================
# Load Data

try:
    snkmkp = snakemake.params
except NameError:
    snkmkp = {
        "load_from_static": False,
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
    scaler = sml.utils.StandardScaler(**af["scaler"]).astype(
        xp.Tensor, dtype=xp.float32
    )

    off_stream = af["off_stream"]


if snkmkp["load_from_static"]:
    model.load_state_dict(xp.load(paths.static / "mock" / "flow_model.pt"))
    xp.save(model.state_dict(), paths.data / "mock" / "flow_model.pt")

    sys.exit(0)

(paths.figures / "mock" / "diagnostic" / "flow").mkdir(parents=True, exist_ok=True)

# =============================================================================
# Training Parameters

loader = td.DataLoader(
    dataset=td.TensorDataset(data[flow_coords].array[off_stream]),
    batch_size=snkmkp["batch_size"],
    shuffle=True,
    num_workers=0,
)

optimizer = optim.AdamW(list(model.parameters()), lr=snkmkp["lr"])


# =============================================================================
# Train

# Turn on gradients for training
object.__setattr__(model, "with_grad", True)  # noqa: FBT003

for epoch in tqdm(range(snkmkp["epochs"])):
    for _step, (data_cur_,) in enumerate(loader):
        data_cur = sml.Data(data_cur_, names=flow_coords)

        optimizer.zero_grad()

        mpars = model.unpack_params(model(data_cur))
        loss = -model.ln_posterior_tot(mpars, data_cur)

        loss.backward()
        optimizer.step()

        # Diagnostic plots (not in the paper)
        if snkmkp["diagnostic_plots"] and (
            (epoch % 100 == 0) or (epoch == snkmkp["epochs"] - 1)
        ):
            with xp.no_grad():
                mpars = model.unpack_params(model(data))
                prob = model.posterior(mpars, data)

            fig, ax = plt.subplots()
            im = ax.scatter(data["g-r"].flatten(), data["g"].flatten(), s=0.2, c=prob)
            plt.colorbar(im, ax=ax)
            fig.savefig(
                paths.figures / "mock" / "diagnostic" / "flow" / f"epoch_{epoch:05}.png"
            )
            plt.close(fig)

# Turn off gradients for running
object.__setattr__(model, "with_grad", False)  # noqa: FBT003

xp.save(model.state_dict(), paths.static / "mock" / "flow_model.pt")
xp.save(model.state_dict(), paths.data / "mock" / "flow_model.pt")
