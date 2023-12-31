"""Query data from Gaia, crossmatched with PS-1."""

import shutil
import sys
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

import asdf
import astropy.coordinates as coords
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astroquery.gaia import Gaia
from astroquery.utils.tap.model.job import Job
from showyourwork.paths import user as user_paths
from tqdm import tqdm

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split
from scripts.gd1.frames import gd1_frame as frame
from scripts.helper import a_as_b

##############################################################################
# Parameters

SAVE_LOC = paths.data / "gd1" / "gaia_ps1_xm_polygons.asdf"

snkmk: dict[str, Any]
try:
    snkmk = dict(snakemake.params)
except NameError:
    snkmk = {"load_from_static": True, "save_to_static": True}


if snkmk["load_from_static"] and (paths.static / "gd1").exists():
    shutil.copyfile(paths.static / "gd1" / "gaia_ps1_xm_polygons.asdf", SAVE_LOC)
    sys.exit(0)
else:
    GAIA_LOGIN = Path(paths.static / "gaia.login").expanduser()
    Gaia.login(credentials_file=GAIA_LOGIN, verbose=False)

Gaia.ROW_LIMIT = -1  # no limit

##############################################################################


PHI1_EDGES = (
    np.concatenate((np.arange(-100, -80, 2), np.arange(-80, 30 + 5, 5))) * u.deg
)
PHI2_BOUNDS = (-9, 5) * u.deg
PLX_BOUNDS = (-10, 1.0) * u.milliarcsecond
BP_RP_BOUNDS = (-1, 3) * u.mag
GMAG_BOUNDS = (0, 30) * u.mag
IMAG_BOUNDS = (0, 30) * u.mag


gaia_cols = {
    "source_id": None,
    # Astrometry
    "ra": None,
    "ra_error": None,
    "dec": None,
    "dec_error": None,
    "parallax": None,
    "parallax_error": None,
    "pmra": None,
    "pmra_error": None,
    "pmdec": None,
    "pmdec_error": None,
    #   correlations
    **{
        f"{a}_{b}_corr": None
        for a, b in combinations(("ra", "dec", "parallax", "pmra", "pmdec"), r=2)
    },
    # Photometry
    "bp_rp": None,
    "phot_g_mean_mag": "gaia_g",
    "phot_g_mean_flux_over_error": "gaia_g_ferror",
    "phot_bp_mean_mag": "gaia_bp",
    "phot_bp_mean_flux_over_error": "gaia_bp_ferror",
    "phot_rp_mean_mag": "gaia_rp",
    "phot_rp_mean_flux_over_error": "gaia_rp_ferror",
    "ruwe": None,
    "ag_gspphot": None,
    "ebpminrp_gspphot": None,
    # For Zero Point Correction
    "nu_eff_used_in_astrometry": "nu_eff",
    "pseudocolour": None,
    "ecl_lat": None,
    "astrometric_params_solved": None,
}

xmatch_cols = {
    "original_ext_source_id": None,
    "angular_distance": "gaia_ps1_angular_distance",
    "number_of_neighbours": None,
    "number_of_mates": None,
}

ps1_cols: dict[str, str | None] = {
    "g_mean_psf_mag": "ps1_g",
    "g_mean_psf_mag_error": "ps1_g_error",
    "r_mean_psf_mag": "ps1_r",
    "r_mean_psf_mag_error": "ps1_r_error",
    "i_mean_psf_mag": "ps1_i",
    "i_mean_psf_mag_error": "ps1_i_error",
    "z_mean_psf_mag": "ps1_z",
    "z_mean_psf_mag_error": "ps1_z_error",
    "y_mean_psf_mag": "ps1_y",
    "y_mean_psf_mag_error": "ps1_y_error",
    "n_detections": "ps1_n_detections",
}

base_query = """
SELECT {gaia_columns}, {xmatch_columns}, {panstarrs_columns}
FROM gaiadr3.gaia_source as G
JOIN gaiadr3.panstarrs1_best_neighbour AS xm USING (source_id)
JOIN gaiadr2.panstarrs1_original_valid AS PS1
   ON xm.original_ext_source_id = PS1.obj_id
WHERE
        CONTAINS(
            POINT('ICRS', G.ra, G.dec),
            POLYGON('ICRS',
                    {{c[0].ra.degree}}, {{c[0].dec.degree}},
                    {{c[1].ra.degree}}, {{c[1].dec.degree}},
                    {{c[2].ra.degree}}, {{c[2].dec.degree}},
                    {{c[3].ra.degree}}, {{c[3].dec.degree}})
        ) = 1
    AND G.parallax BETWEEN {plx_bounds[0].value} AND {plx_bounds[1].value}
    AND G.phot_g_mean_mag BETWEEN {gmag_bounds[0].value} AND {gmag_bounds[1].value}
    AND G.bp_rp BETWEEN {bp_rp[0].value} AND {bp_rp[1].value}
""".format(
    gaia_columns=a_as_b(gaia_cols, "G."),
    xmatch_columns=a_as_b(xmatch_cols, "xm."),
    panstarrs_columns=a_as_b(ps1_cols, "PS1."),
    plx_bounds=PLX_BOUNDS,
    bp_rp=BP_RP_BOUNDS,
    gmag_bounds=GMAG_BOUNDS,
)
base_query = base_query[1:]  # remove leading newline

# =============================================================================

af = asdf.AsdfFile()

# Load in base metadata
af["base_query"] = base_query
af["query_box"] = {
    "phi1_edges": PHI1_EDGES,
    "phi2_bounds": PHI2_BOUNDS,
    "plx_bounds": PLX_BOUNDS,
    "bp_rp_bounds": BP_RP_BOUNDS,
    "gmag_bounds": GMAG_BOUNDS,
    "imag_bounds": IMAG_BOUNDS,
}
af["frame"] = f"{frame.__class__.__module__}.{frame.__class__.__name__}"

# setup for the jobs
icrs_frame = coords.ICRS()

# Start the jobs
jobs: dict[str, tuple[str, Job]] = {}
for i, (phi1a, phi1b) in tqdm(
    enumerate(pairwise(PHI1_EDGES)), total=len(PHI1_EDGES) - 1
):
    # Construct the query
    vertices = frame.realize_frame(
        coords.UnitSphericalRepresentation(
            lon=u.Quantity([phi1a, phi1a, phi1b, phi1b]),  # ll, ul, ur, lr
            lat=u.Quantity([*PHI2_BOUNDS, *PHI2_BOUNDS[::-1]]),
        )
    )
    vertices_icrs = vertices.transform_to(icrs_frame)
    query = base_query.format(c=vertices_icrs)

    # Perform the query & save the results
    name = f"GD1_{i:02d}"
    job = Gaia.launch_job_async(query, name=name, background=True)
    jobs[name] = (query, job)

# Collect results
for i, (query, job) in tqdm(enumerate(jobs.values()), total=len(jobs)):
    tbl = job.get_results()

    # Add metadata
    tbl.meta["query"] = query
    tbl.meta["query_box"] = {
        "phi1_edges": (PHI1_EDGES[i], PHI1_EDGES[i + 1]),
        "phi2_bounds": PHI2_BOUNDS,
        "plx_bounds": PLX_BOUNDS,
        "bp_rp_bounds": BP_RP_BOUNDS,
        "gmag_bounds": GMAG_BOUNDS,
        "imag_bounds": IMAG_BOUNDS,
    }
    tbl.meta["frame"] = f"{frame.__class__.__module__}.{frame.__class__.__name__}"

    # Add to ASDF
    af[f"polygon-{i:02d}"] = tbl

    # Save the ASDF
    af.write_to(SAVE_LOC)

af.close()


if snkmk["save_to_static"]:
    shutil.copyfile(SAVE_LOC, paths.static / "gd1" / "gaia_ps1_xm_polygons.asdf")


##############################################################################
# Diagnostic plot

_p2_bnds = PHI2_BOUNDS.value

fig, axs = plt.subplots(2, 1, figsize=(10, 7))

for i, (p1a, p1b) in enumerate(pairwise(PHI1_EDGES.value)):
    axs[1].plot(
        [p1a, p1a, p1b, p1b, p1a], [*_p2_bnds, *_p2_bnds[::-1], _p2_bnds[0]], c="k"
    )
    axs[1].annotate(f"box {i}", (np.mean([p1a, p1b]) - 4, np.mean(_p2_bnds)))

    vx = frame.realize_frame(
        coords.UnitSphericalRepresentation(
            lon=u.Quantity([p1a, p1a, p1b, p1b], u.deg),  # ll, ul, ur, lr
            lat=u.Quantity([*PHI2_BOUNDS, *PHI2_BOUNDS[::-1]], u.deg),
        )
    ).transform_to(icrs_frame)
    axs[0].plot(
        [*vx.ra.degree, vx.ra[0].degree], [*vx.dec.degree, vx.dec[0].degree], c="k"
    )
    axs[0].annotate(f"box {i}", (np.mean(vx.ra.degree) - 4, np.mean(vx.dec.degree)))

axs[1].set_ylim(-12, 6)
axs[1].set_aspect("equal")
axs[0].set_aspect("equal")

fig.savefig(paths.scripts / "gd1" / "_diagnostics" / "query_boxes.png", dpi=300)
