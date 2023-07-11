"""Setup."""

import shutil
import sys
from pathlib import Path

import asdf
import matplotlib.path as mpath
import numpy as np
from astropy.table import QTable

# isort: split

sys.path.append(Path(__file__).parent.parent.parent.as_posix())
# isort: split

import paths  # noqa: E402

##############################################################################
# PARAMETERS

SAVE_LOC = paths.data / "gd1" / "masks.asdf"

try:
    snkmkp = snakemake.params
except NameError:
    snkmkp = {"load_from_static": True}


if snkmkp["load_from_static"]:
    shutil.copyfile(paths.static / "gd1" / "masks.asdf", SAVE_LOC)

    sys.exit(0)


##############################################################################
# Read tables

# Mask edges
pm_edges = QTable.read(paths.data / "gd1" / "pm_edges.ecsv")
pm_edges.add_index("label", unique=True)

with asdf.open(
    paths.data / "gd1" / "isochrone.asdf", lazy_load=True, copy_arrays=True
) as af:
    isochrone = af["isochrone"]
    iso_tight = af["iso_tight"]
    iso_medium = af["iso_medium"]
    iso_loose = af["iso_loose"]

# Gaia Data
table = QTable.read(paths.data / "gd1" / "gaia_ps1_xm.asdf")

##############################################################################
# Masks


masks_table = QTable()

# =============================================================================
# Proper motion

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
