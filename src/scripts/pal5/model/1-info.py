"""Pal-5 Model Info."""

import sys
from pathlib import Path

import asdf
import numpy as np
from astropy.table import QTable

# isort: split
import stream_ml.pytorch as sml

sys.path.append(Path(__file__).parents[3].as_posix())
# isort: split

from scripts import paths

##############################################################################

try:
    snkmk = snakemake.params
except NameError:
    snkmk = {
        "pm_mask": "pm_tight",
        "phot_mask": "cmd_medium",
    }


##############################################################################

# Read tables
table = QTable.read(paths.data / "pal5" / "gaia_ps1_xm.asdf")
masks = QTable.read(paths.data / "pal5" / "masks.asdf")

# Info file
af = asdf.AsdfFile()

# -----------------------------------------------------------------------------
# Mask
# TODO: move this to the data files

sel = (
    # (table["parallax"] > 0 * u.milliarcsecond)  # TODO: allow negative parallax
    # & masks[snkmk["phot_mask"]]
    masks["M5"]
    & masks["things"]
    & masks[snkmk["pm_mask"]]
)
table = table[sel]

# Save mask
af["mask"] = sel


# ----------------------------------------------------
# Data

af["renamer"] = {
    "source_id": "id",
    # ------
    "ra": "ra",
    "ra_error": "ra_err",
    "dec": "dec",
    "dec_error": "dec_err",
    "parallax": "plx",
    "parallax_error": "plx_err",
    "pmra": "pmra",
    "pmra_error": "pmra_err",
    "pmdec": "pmdec",
    "pmdec_error": "pmdec_err",
    "ra_dec_corr": "ra_dec_corr",
    "ra_parallax_corr": "ra_plx_corr",
    "ra_pmra_corr": "ra_pmra_corr",
    "ra_pmdec_corr": "ra_pmdec_corr",
    "dec_parallax_corr": "dec_plx_corr",
    "dec_pmra_corr": "dec_pmra_corr",
    "dec_pmdec_corr": "dec_pmdec_corr",
    "parallax_pmra_corr": "plx_pmra_corr",
    "parallax_pmdec_corr": "plx_pmdec_corr",
    "pmra_pmdec_corr": "pmra_pmdec_corr",
    "bp_rp": "bp_rp",
    # 'gaia_g',
    # 'gaia_g_ferror',
    # 'gaia_bp',
    # 'gaia_bp_ferror',
    # 'gaia_rp',
    # 'gaia_rp_ferror',
    # 'ruwe',
    # 'ag_gspphot',
    # 'ebpminrp_gspphot',
    # 'original_ext_source_id',
    # 'gaia_ps1_angular_distance',
    # 'number_of_neighbours',
    # 'number_of_mates',
    # 'ps1_g',
    "ps1_g_error": "g_err",
    # 'ps1_r',
    "ps1_r_error": "r_err",
    # 'ps1_i',
    "ps1_i_error": "i_err",
    # 'ps1_z',
    "ps1_z_error": "z_err",
    # 'ps1_y',
    "ps1_y_error": "y_err",
    # 'ps1_n_detections',
    "phi1": "phi1",
    "phi1_error": "phi1_err",
    "phi2": "phi2",
    "phi2_error": "phi2_err",
    "pm_phi1": "pmphi1",
    "pm_phi1_error": "pmphi1_err",
    "pm_phi2": "pmphi2",
    "pm_phi2_error": "pmphi2_err",
    "pmphi1_pmphi2_corr": "pmphi1_pmphi2_corr",
    "g0": "g",
    "r0": "r",
    "i0": "i",
    "z0": "z",
    "y0": "y",
}

af["names"] = (
    "phi1",
    "phi1_error",
    "phi2",
    "phi2_error",
    "parallax",
    "parallax_error",
    "pm_phi1",
    "pm_phi1_error",
    "pm_phi2",
    "pm_phi2_error",
    "pmphi1_pmphi2_corr",
    "g0",
    "ps1_g_error",
    "r0",
    "ps1_r_error",
)

data = sml.Data.from_format(
    table, fmt="astropy.table", names=af["names"], renamer=af["renamer"]
)


# -----------------------------------------------------------------------------
# Coordinate Bounds

af["coord_bounds"] = {k: (np.nanmin(data[k]), np.nanmax(data[k])) for k in data.names}

# -----------------------------------------------------------------------------
# Scaling for ML

scaler = sml.utils.StandardScaler.fit(data, names=data.names)

af["scaler"] = {"mean": scaler.mean, "scale": scaler.scale, "names": scaler.names}

# -----------------------------------------------------------------------------

af.write_to(paths.data / "pal5" / "info.asdf")
af.close()
