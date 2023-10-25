"""Define data masks for GD-1."""

import shutil
import sys

import asdf
import astropy.units as u
import matplotlib.path as mpath
import numpy as np
from astropy.table import QTable
from showyourwork.paths import user as user_paths

paths = user_paths()

##############################################################################
# PARAMETERS

SAVE_LOC = paths.data / "gd1" / "masks.asdf"

snkmk: dict[str, bool]
try:
    snkmk = snakemake.params
except NameError:
    snkmk = {"load_from_static": False, "save_to_static": False}


if snkmk["load_from_static"]:
    shutil.copyfile(paths.static / "gd1" / "masks.asdf", SAVE_LOC)

    sys.exit(0)

# Gaia Data
table = QTable.read(paths.data / "gd1" / "gaia_ps1_xm.asdf")


##############################################################################
# Masks

masks_table = QTable()

# =============================================================================
# Proper motion

# Mask edges
pm_edges = QTable.read(paths.data / "gd1" / "pm_edges.ecsv")
pm_edges.add_index("label", unique=True)

# Tight
pm_tight = pm_edges.loc["tight"]
masks_table["pm_tight"] = (
    (table["pm_phi1"] > pm_tight["pm_phi1_min"])
    & (table["pm_phi1"] < pm_tight["pm_phi1_max"])
    & (table["pm_phi2"] > pm_tight["pm_phi2_min"])
    & (table["pm_phi2"] < pm_tight["pm_phi2_max"])
)

# Medium
pm_medium = pm_edges.loc["medium"]
masks_table["pm_medium"] = (
    (table["pm_phi1"] > pm_medium["pm_phi1_min"])
    & (table["pm_phi1"] < pm_medium["pm_phi1_max"])
    & (table["pm_phi2"] > pm_medium["pm_phi2_min"])
    & (table["pm_phi2"] < pm_medium["pm_phi2_max"])
)

# =============================================================================
# Parallax

masks_table["neg_parallax"] = table["parallax"] > 0 * u.mas


# =============================================================================
# Photometry

with asdf.open(
    paths.data / "gd1" / "isochrone.asdf", lazy_load=False, copy_arrays=True
) as af:
    iso_medium = af["isochrone_medium"]

mags = np.c_[table["g0"] - table["i0"], table["g0"]]

masks_table["phot_medium"] = mpath.Path(iso_medium, readonly=True).contains_points(mags)


# =============================================================================

masks_table.write(SAVE_LOC)

if snkmk["save_to_static"]:
    shutil.copyfile(SAVE_LOC, paths.static / "gd1" / "masks.asdf")
