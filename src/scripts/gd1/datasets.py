"""Import data."""

import pathlib

import asdf
import astropy.units as u
import numpy as np
import torch as xp
from astropy.table import QTable

import stream_mapper.pytorch as sml

from ..syw import user as user_paths

paths = user_paths(pathlib.Path(__file__).parents[3])

###############################################################################

# Info
with asdf.open(
    paths.static / "gd1" / "info.asdf", lazy_load=False, copy_arrays=True
) as af:
    sel = af["mask"]
    names = tuple(af["names"])
    renamer = af["renamer"]

# Tables
table = QTable.read(paths.static / "gd1" / "gaia_ps1_xm.asdf")[sel]
masks = QTable.read(paths.static / "gd1" / "masks.asdf")[sel]

# =============================================================================

completeness_mask = table["gaia_g"] > 20 * u.mag
table["g0"][completeness_mask] = np.nan
table["r0"][completeness_mask] = np.nan
table["i0"][completeness_mask] = np.nan
table["z0"][completeness_mask] = np.nan
table["y0"][completeness_mask] = np.nan

data = sml.Data.from_format(
    table, fmt="astropy.table", names=names, renamer=renamer
).astype(xp.Tensor, dtype=xp.float32)

# True where NOT missing
where = sml.Data((~xp.isnan(data.array)), names=data.names)

# TODO: it would be nice to keep this as NaN.
# Need to set missing data to some value for the gradient.
data.array[~where.array] = xp.asarray(
    np.repeat(np.nanmedian(data.array, axis=0, keepdims=True), len(data), axis=0)
)[~where.array]

# Off-stream selection
# This will select the off-stream region (it's the opposite of the mask).
off_stream = ~masks["offstream"]
