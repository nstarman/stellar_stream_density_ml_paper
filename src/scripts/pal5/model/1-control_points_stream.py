"""Plot results."""

import sys

import astropy.units as u
import galstreams
import numpy as np
from astropy.coordinates import Distance
from astropy.table import QTable
from interpolated_coordinates import InterpolatedSkyCoord
from matplotlib import pyplot as plt
from showyourwork.paths import user as user_paths

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.pal5.datasets import data, masks
from scripts.pal5.frames import pal5_frame as frame

try:
    snkmk = snakemake.params
except NameError:
    snkmk = {
        "diagnostic_plots": True,
    }


table = QTable(
    np.empty((11, 7)),
    names=("phi1", "phi2", "w_phi2", "distance", "w_distance", "distmod", "w_distmod"),
    units=(u.deg, u.deg, u.deg, u.kpc, u.kpc, u.mag, u.mag),
)

# Adding in regularly spaced points from :mod:`galstreams`
allstreams = galstreams.MWStreams(implement_Off=True)
pal5 = allstreams["Pal5-I21"]
track = pal5.track.transform_to(frame)[::-1]
itrack = InterpolatedSkyCoord(track, affine=track.phi1)

x = np.linspace(track.phi1.min(), track.phi1.max(), len(table) - 1)
nans = np.full_like(x.value, np.nan)

#       ([progenitor], [stream])
table["phi1"] = np.concatenate(([0], x)) << u.deg

# phi2
table["phi2"] = np.concatenate(([0], itrack(x).phi2)) << u.deg
table["w_phi2"] = np.concatenate(([0.1], np.full_like(x.value, 1))) << u.deg
# distance
table["distance"] = np.concatenate(([np.nan], itrack(x).distance)) << u.kpc
table["w_distance"] = np.concatenate(([np.nan], np.full_like(x.value, 1.5))) << u.kpc
# -> distmod
table["distmod"] = Distance(table["distance"]).distmod << u.mag
table["w_distmod"] = (
    np.abs(
        Distance(table["distance"] + table["w_distance"]).distmod
        - Distance(table["distance"] - table["w_distance"]).distmod
    )
    / 2
) << u.mag

# pmphi1
prog_pmphi1s = data["pmphi1"][~masks["Pal5"]]
table["pmphi1"] = np.concatenate(([np.nanmean(prog_pmphi1s)], nans)) << u.mas / u.yr
table["w_pmphi1"] = np.concatenate(([np.nanstd(prog_pmphi1s)], nans)) << u.mas / u.yr

# pmphi2
prog_pmphi2s = data["pmphi2"][~masks["Pal5"]]
table["pmphi2"] = np.concatenate(([np.nanmean(prog_pmphi2s)], nans)) << u.mas / u.yr
table["w_pmphi2"] = np.concatenate(([np.nanstd(prog_pmphi2s)], nans)) << u.mas / u.yr

# Save
table.write(paths.data / "pal5" / "control_points_stream.ecsv", overwrite=True)


##############################################################################
# Diagnostic plots

if not snkmk["diagnostic_plots"]:
    sys.exit(0)


fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(xlabel=r"$\phi_1$ [deg]", ylabel=r"$d$ [kpc]")

trackPW19 = allstreams["Pal5-PW19"].track.transform_to(frame)
ax.plot(trackPW19.phi1, trackPW19.distance, label="PW-19")

trackI21 = allstreams["Pal5-I21"].track.transform_to(frame)
ax.plot(trackI21.phi1, trackI21.distance, label="I21")

# From https://doi.org/10.3847/1538-4357/ab5afe
# covers 3 degree wide bins of which these are the centers
bonaca_19_phi1 = [-13.5, -10.5, -7.5, -4.5, -1.5, 1.5, 4.5, 7.5, 10.5] * u.deg
# * u.kpc
bonaca_19_distances = (
    22.5 * u.kpc + [1.1, 0.9, 0.6, 0.4, 0.2, 0, -1.5, -3.6, -5.5] * u.kpc
)
ax.plot(bonaca_19_phi1, bonaca_19_distances, label="Bonaca+19")
ax.legend(loc="best", fontsize=8)
fig.savefig(
    paths.scripts / "pal5" / "_diagnostics" / "control_points_distance_confusion.png",
    dpi=300,
)
