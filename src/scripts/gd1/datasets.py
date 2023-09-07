"""Import data."""

import sys
from pathlib import Path

import asdf
import astropy.units as u
import numpy as np
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

# TODO: where should this go?
# We set photoometrics with G_gaia > 20 to NaN
completeness_mask = table["gaia_g"] > 20 * u.mag
table["g0"][completeness_mask] = np.nan
table["r0"][completeness_mask] = np.nan
table["i0"][completeness_mask] = np.nan
table["z0"][completeness_mask] = np.nan
table["y0"][completeness_mask] = np.nan

# =============================================================================
# Make Data object

data = sml.Data.from_format(
    table, fmt="astropy.table", names=names, renamer=renamer
).astype(xp.Tensor, dtype=xp.float32)

where = sml.Data(
    (~xp.isnan(data.array)),  # True where NOT missing
    names=data.names,
)

# TODO: it would be nice to keep this as NaN.
# Need to set missing data to some value, even though it's ignored, for the
# gradient
data.array[~where.array] = xp.asarray(
    np.repeat(np.nanmedian(data.array, axis=0, keepdims=True), len(data), axis=0)[
        ~where.array
    ]
)


# =============================================================================
# Off-stream selection

off_stream = (data["phi2"] < -1.7) | (data["phi2"] > 2)


# =============================================================================
# Save temp file

pth = paths.data / "gd1" / "data.tmp"
if not pth.exists():
    with pth.open("w") as f:
        f.write("hack for snakemake")
