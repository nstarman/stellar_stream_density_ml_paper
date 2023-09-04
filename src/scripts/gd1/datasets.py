"""Import data."""

import sys
from pathlib import Path

import asdf
import torch as xp
from astropy.table import QTable

import stream_ml.pytorch as sml

# Add the parent directory to the path
sys.path.append(Path(__file__).parents[2].as_posix())
# isort: split

from scripts import paths

# =============================================================================
# Load data table

with asdf.open(
    paths.data / "gd1" / "info.asdf", lazy_load=False, copy_arrays=True
) as af:
    sel = af["mask"]
    names = tuple(af["names"])
    renamer = af["renamer"]

table = QTable.read(paths.data / "gd1" / "gaia_ps1_xm.asdf")[sel]


# =============================================================================
# Make Data object

data = sml.Data.from_format(
    table, fmt="astropy.table", names=names, renamer=renamer
).astype(xp.Tensor, dtype=xp.float32)

where = sml.Data(
    (~xp.isnan(data.array)),  # True where NOT missing
    names=data.names,
)

# TODO: it would be nice to keep this as NaN
data.array[xp.isnan(data.array)] = 0.0  # set missing data to zero


# =============================================================================
# Off-stream selection

off_stream = (data["phi2"] < -1.7) | (data["phi2"] > 2)


# =============================================================================
# Save temp file

pth = paths.data / "gd1" / "data.tmp"
if not pth.exists():
    with pth.open("w") as f:
        f.write("hack for snakemake")
