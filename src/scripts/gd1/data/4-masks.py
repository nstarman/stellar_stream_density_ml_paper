"""Define data masks for GD-1."""

import shutil
import sys

import asdf
import astropy.units as u
import galstreams
import matplotlib.path as mpath
import numpy as np
from astropy.table import QTable
from showyourwork.paths import user as user_paths

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.gd1.frames import gd1_frame
from scripts.helper import make_path, make_vertices

##############################################################################
# PARAMETERS

SAVE_LOC = paths.data / "gd1" / "masks.asdf"

snkmk: dict[str, bool]
try:
    snkmk = snakemake.params
except NameError:
    snkmk = {"load_from_static": True, "save_to_static": False}


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

# Tight
path_pmphi1 = make_path(pm_edges["phi1"], pm_edges["pm_phi1_cosphi2_tight"])
path_pmphi2 = make_path(pm_edges["phi1"], pm_edges["pm_phi2_tight"])

masks_table["pm_tight"] = path_pmphi1.contains_points(
    np.c_[table["phi1"].value, table["pm_phi1"].value]
) & path_pmphi2.contains_points(np.c_[table["phi1"].value, table["pm_phi2"].value])

# Medium
path_pmphi1 = make_path(pm_edges["phi1"], pm_edges["pm_phi1_cosphi2_medium"])
path_pmphi2 = make_path(pm_edges["phi1"], pm_edges["pm_phi2_medium"])

masks_table["pm_medium"] = path_pmphi1.contains_points(
    np.c_[table["phi1"].value, table["pm_phi1"].value]
) & path_pmphi2.contains_points(np.c_[table["phi1"].value, table["pm_phi2"].value])


# =============================================================================
# Parallax

masks_table["neg_parallax"] = table["parallax"] > 0 * u.mas

# =============================================================================
# Parallax

masks_table["low_phi2"] = table["phi2"] > -5 * u.deg


# =============================================================================
# Photometry

with asdf.open(
    paths.data / "gd1" / "isochrone.asdf", lazy_load=False, copy_arrays=True
) as af:
    iso_medium = af["isochrone_medium"]

mags = np.c_[table["g0"] - table["i0"], table["g0"]]

masks_table["phot_medium"] = mpath.Path(iso_medium, readonly=True).contains_points(mags)

# =============================================================================
# Offstream

# Get track from galstreams
mws = galstreams.MWStreams()
gd1 = mws["GD-1-I21"]
gd1_sc = gd1.track.transform_to(gd1_frame)[::100]

# Make path from track
verts = make_vertices(gd1_sc.phi1, gd1_sc.phi2, 1 * u.deg)
path_phi12 = make_path(verts[:, 0], verts[:, 1])

# Offstream mask
offstream = path_phi12.contains_points(np.c_[table["phi1"].value, table["phi2"].value])
masks_table["offstream"] = offstream

# =============================================================================

masks_table.write(SAVE_LOC)

if snkmk["save_to_static"]:
    shutil.copyfile(SAVE_LOC, paths.static / "gd1" / "masks.asdf")
