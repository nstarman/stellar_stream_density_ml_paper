"""Define data masks for GD-1."""

import shutil
import sys

import asdf
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

# Loose
pm_loose = pm_edges.loc["loose"]
masks_table["pm_loose"] = (
    (table["pm_phi1"] > pm_loose["pm_phi1_min"])
    & (table["pm_phi1"] < pm_loose["pm_phi1_max"])
    & (table["pm_phi2"] > pm_loose["pm_phi2_min"])
    & (table["pm_phi2"] < pm_loose["pm_phi2_max"])
)


# =============================================================================
# Photometry


with asdf.open(
    paths.data / "gd1" / "isochrone.asdf", lazy_load=False, copy_arrays=True
) as af:
    isochrone = af["isochrone"]
    iso_tight = af["isochrone_tight"]
    iso_medium = af["isochrone_medium"]
    iso_loose = af["isochrone_loose"]

mags = np.c_[table["g0"] - table["i0"], table["g0"]]

# Tight
cmd_tight = mpath.Path(iso_tight, readonly=True).contains_points(mags)
masks_table["cmd_tight"] = cmd_tight

# Medium
cmd_med = mpath.Path(iso_medium, readonly=True).contains_points(mags)
masks_table["cmd_medium"] = cmd_med


# Loose
cmd_loose = mpath.Path(iso_loose, readonly=True).contains_points(mags)
masks_table["cmd_loose"] = cmd_loose


# =============================================================================

masks_table.write(SAVE_LOC)

if snkmk["save_to_static"]:
    shutil.copyfile(SAVE_LOC, paths.static / "gd1" / "masks.asdf")
