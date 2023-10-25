"""Import data."""


import asdf
import astropy.units as u
import numpy as np
import torch as xp
from astropy.table import QTable
from showyourwork.paths import user as user_paths

import stream_ml.pytorch as sml

paths = user_paths()

# Info
with asdf.open(
    paths.data / "gd1" / "info.asdf", lazy_load=False, copy_arrays=True
) as af:
    sel = af["subset"]["mask"]
    names = tuple(af["names"])
    renamer = af["renamer"]

# Tables
table = QTable.read(paths.data / "gd1" / "gaia_ps1_xm.asdf")[sel]
masks = QTable.read(paths.data / "gd1" / "masks.asdf")[sel]

# =============================================================================
# Subset

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
off_stream = (data["phi2"] < -1.7) | (data["phi2"] > 2)
