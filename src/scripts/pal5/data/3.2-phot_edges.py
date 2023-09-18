"""Define the photometric data mask boundaries for GD-1."""

import sys
from dataclasses import asdict

import asdf
import astropy.units as u
import brutus.filters
import brutus.seds
import numpy as np
import shapely
from astropy.coordinates import Distance
from astropy.table import QTable
from matplotlib import pyplot as plt
from showyourwork.paths import user as user_paths

import stream_ml.pytorch as sml

paths = user_paths()

try:
    snkmk = snakemake.params
except NameError:
    snkmk = {
        "diagnostic_plots": True,
    }


##############################################################################

filters = brutus.filters.ps[:-2]  # (g, r)

# Build isochrone
# From https://arxiv.org/pdf/1910.00592.pdf
# 11.5 Gyr MIST isochrone with [Fe/H] = -1.3 (Choi et al. 2016) between 20 < g < 23.7
abs_mags, _, _ = brutus.seds.Isochrone(
    filters=filters,
    nnfile=(paths.data / "brutus" / "nn_c3k.h5").resolve(),
    mistfile=(paths.data / "brutus" / "MIST_1.2_iso_vvcrit0.0.h5").resolve(),
).get_seds(
    eep=np.linspace(202, 600, 5000),
    apply_corr=True,
    feh=-1.3,
    dist=10,  # absolute magnitudes
    loga=np.log10(11.5e9),
)
abs_mags = abs_mags[np.all(np.isfinite(abs_mags), axis=1)]
mags_data = sml.Data(abs_mags, names=[n.removeprefix("PS_") for n in filters])

af = asdf.AsdfFile()
af["isochrone_data"] = asdict(mags_data)
af["isochrone_data"].pop("_n2k")


# Shift to distance of Palomar 5
# Palomar 5 has a distance gradient, so it's important that a simple
# photometric selection is wide enough to capture the full stream, including
# the more distant parts. See the diagnostic plot below.
app_mags = abs_mags + Distance(20 * u.kpc).distmod.value
isochrone = shapely.LineString(app_mags[:, :2])
iso_buffer = isochrone.buffer(0.15)

# Masks
af["isochrone_"] = np.c_[isochrone.xy]
af["isochrone_15"] = isochrone_15 = np.c_[iso_buffer.exterior.xy]

af.write_to(paths.data / "pal5" / "isochrone.asdf")
af.close()


###############################################################################
# Diagnostic plot

if not snkmk["diagnostic_plots"]:
    sys.exit(0)

# Load data table. This is big!
table = QTable.read(paths.data / "pal5" / "gaia_ps1_xm.asdf")


fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(
    xlabel="g - r [mag]", ylabel="g [mag]", xlim=(-0.5, 1.5), ylim=(21, 12)
)

ax.plot(table["g0"] - table["r0"], table["g0"], c="k", ls="none", marker=",")
ax.plot(
    abs_mags[:, 0] - abs_mags[:, 1],
    abs_mags[:, 0] + Distance(18 * u.kpc).distmod.value,
    c="r",
    ls="--",
    lw=1,
)
ax.plot(
    abs_mags[:, 0] - abs_mags[:, 1],
    abs_mags[:, 0] + Distance(20 * u.kpc).distmod.value,
    c="r",
    lw=2,
    label="Isochrone (11.5 Gyr, [Fe/H] = -1.3, 20 kpc)",
)
ax.plot(
    abs_mags[:, 0] - abs_mags[:, 1],
    abs_mags[:, 0] + Distance(23 * u.kpc).distmod.value,
    c="r",
    ls="--",
    lw=1,
)
ax.plot(
    isochrone_15[:, 0] - isochrone_15[:, 1],
    isochrone_15[:, 0],
    c="r",
    lw=1,
    label="Isochrone buffer",
)
ax.legend(loc="upper left", fontsize=8)
fig.savefig(paths.scripts / "pal5" / "_diagnostics" / "phot_edges.png", dpi=300)
