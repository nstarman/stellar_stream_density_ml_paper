"""Import data."""

import sys
from pathlib import Path

import asdf
import torch as xp
from astropy.table import QTable

import stream_ml.pytorch as sml

# isort: split
# Add the parent directory to the path
sys.path.append(Path(__file__).parent.parent.parent.as_posix())
# isort: split

import paths  # noqa: E402

# =============================================================================
# Load data and model

with asdf.open(
    paths.data / "gd1" / "info.asdf", lazy_load=False, copy_arrays=True
) as af:
    sel = af["mask"]
    names = tuple(af["names"])
    renamer = af["renamer"]

table = QTable.read(paths.data / "gd1" / "gaia_ps1_xm.asdf")[sel]

data = sml.Data.from_format(
    table, fmt="astropy.table", names=names, renamer=renamer
).astype(xp.Tensor, dtype=xp.float32)

where = sml.Data(
    (~xp.isnan(data.array)[:, 1:]),  # True where NOT missing
    names=data.names[1:],
)

# TODO: it would be nice to keep this as NaN
data.array[xp.isnan(data.array)] = 0.0  # set missing data to zero


# -----------------------------------------------------------------------------

pth = paths.data / "gd1" / "data.tmp"
if not pth.exists():
    with pth.open("w") as f:
        f.write("hack for snakemake")
