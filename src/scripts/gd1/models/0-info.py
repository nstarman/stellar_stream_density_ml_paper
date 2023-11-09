"""GD-1 model info."""


from dataclasses import asdict

import asdf
import numpy as np
from astropy.table import QTable
from showyourwork.paths import user as user_paths

import stream_ml.pytorch as sml
from stream_ml.pytorch.utils import StandardScaler

paths = user_paths()

##############################################################################

# Read tables
table = QTable.read(paths.data / "gd1" / "gaia_ps1_xm.asdf")
masks = QTable.read(paths.data / "gd1" / "masks.asdf")

# Info file
af = asdf.AsdfFile()

# -----------------------------------------------------------------------------
# Mask

sel = (
    masks["pm_tight"] & masks["phot_medium"] & masks["neg_parallax"] & masks["low_phi2"]
)

# Save mask
af["mask_info"] = {
    "pm_mask": "pm_tight",
    "phot_mask": "phot_medium",
    "plx_mask": "neg_parallax",
    "phi2_mask": "low_phi2",
}
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

data_sub = sml.Data.from_format(
    table[sel], fmt="astropy.table", names=af["names"], renamer=af["renamer"]
)

# -----------------------------------------------------------------------------
# Coordinate Bounds

af["coord_bounds"] = {
    k: (np.nanmin(data_sub[k]), np.nanmax(data_sub[k])) for k in data_sub.names
}

# -----------------------------------------------------------------------------
# Scaling for ML
# We only scale for the full dataset

scaler = StandardScaler.fit(data_sub, names=data_sub.names)

af["scaler"] = asdict(scaler)

# -----------------------------------------------------------------------------

(paths.data / "gd1").mkdir(exist_ok=True, parents=True)
af.write_to(paths.data / "gd1" / "info.asdf")
af.close()
