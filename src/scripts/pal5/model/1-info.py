"""Pal-5 Model Info."""

from dataclasses import asdict

import asdf
import numpy as np
from astropy.table import QTable
from showyourwork.paths import user as user_paths

# isort: split
import stream_ml.pytorch as sml

paths = user_paths()

##############################################################################

try:
    snkmk = snakemake.params
except NameError:
    snkmk = {
        "pm_mask": "pm_med_icrs",
        "phot_mask": "phot_15",
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
    masks["M5"]
    & masks["things"]
    & masks[snkmk["pm_mask"]]
    & masks[snkmk["phot_mask"]]
)
table = table[sel]

# Save mask
af["mask_info"] = {"pm_mask": snkmk["pm_mask"], "phot_mask": snkmk["phot_mask"]}
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
    # "ps1_g_error": "g_err",
    # # 'ps1_r',
    # "ps1_r_error": "r_err",
    # # 'ps1_i',
    # "ps1_i_error": "i_err",
    # # 'ps1_z',
    # "ps1_z_error": "z_err",
    # # 'ps1_y',
    # "ps1_y_error": "y_err",
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
    "g0_error": "g_err",
    "r0": "r",
    "r0_error": "r_err",
    "i0": "i",
    "i0_error": "i_err",
    "z0": "z",
    "z0_error": "z_err",
    "y0": "y",
    "y0_error": "y_err",
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
    "g0_error",
    "r0",
    "r0_error",
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
af["scaler"] = asdict(scaler)

# -----------------------------------------------------------------------------

af.write_to(paths.data / "pal5" / "info.asdf")
af.close()
