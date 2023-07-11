"""Train photometry background flow."""

import sys
from pathlib import Path

import torch as xp
import torch.utils.data as td
from torch import optim
from tqdm import tqdm

import stream_ml.pytorch as sml

# isort: split
from data import data
from define_model import background_photometric_model as bkg_flow

# Add the parent directory to the path
sys.path.append(Path(__file__).parent.parent.as_posix())
# isort: split

import paths  # noqa: E402

# =============================================================================
# Load Data

try:
    snkmkp = snakemake.params
except NameError:
    snkmkp = {
        "load_from_static": True,
        "epochs": 1_000,
    }

if snkmkp["load_from_static"]:
    bkg_flow.load_state_dict(xp.load(paths.static / "gd1" / "flow_model.pt"))
    xp.save(bkg_flow.state_dict(), paths.data / "gd1" / "flow_model.pt")

    sys.exit(0)


# =============================================================================
# Train

off_stream = (data["phi2"] < -1.5) | (data["phi2"] > 2)

coord_names = (*bkg_flow.indep_coord_names, *bkg_flow.coord_names)
dataset = td.TensorDataset(data[coord_names].array[off_stream])
loader = td.DataLoader(dataset=dataset, batch_size=500, shuffle=True, num_workers=0)
optimizer = optim.Adam(bkg_flow.parameters(), lr=1e-3)

# Turn on gradients for training
object.__setattr__(bkg_flow, "with_grad", True)  # noqa: FBT003

for _epoch in tqdm(range(snkmkp["epochs"])):
    for _step, (data_cur_,) in enumerate(loader):
        data_cur = sml.Data(data_cur_, names=coord_names)

        optimizer.zero_grad()

        mpars = bkg_flow.unpack_params(bkg_flow(data_cur))
        loss = -bkg_flow.ln_posterior_tot(mpars, data_cur)

        loss.backward()
        optimizer.step()

# Turn off gradients for usage later
object.__setattr__(bkg_flow, "with_grad", False)  # noqa: FBT003

xp.save(bkg_flow.state_dict(), paths.data / "gd1" / "flow_model.pt")
xp.save(bkg_flow.state_dict(), paths.static / "gd1" / "flow_model.pt")
