"""Define the photometric data mask boundaries for GD-1."""

from dataclasses import asdict

import asdf
import astropy.units as u
import brutus.filters
import brutus.seds
import numpy as np
import shapely
from astropy.coordinates import Distance
from showyourwork.paths import user as user_paths

import stream_ml.pytorch as sml

paths = user_paths()

##############################################################################

filters = brutus.filters.ps[:-2]  # (g, r)

# Build isochrone
abs_mags, _, _ = brutus.seds.Isochrone(
    filters=filters,
    nnfile=(paths.data / "brutus" / "nn_c3k.h5").resolve(),
    mistfile=(paths.data / "brutus" / "MIST_1.2_iso_vvcrit0.0.h5").resolve(),
).get_seds(
    eep=np.linspace(202, 600, 5000),
    apply_corr=True,
    feh=-1.2,
    # feh=-1.35,
    # dist=7.8e3,
    dist=10,
    loga=np.log10(12e9),
)
abs_mags = abs_mags[np.all(np.isfinite(abs_mags), axis=1)]
mags_data = sml.Data(abs_mags, names=[n.removeprefix("PS_") for n in filters])

app_mags = abs_mags + Distance(7.8 * u.kpc).distmod.value
isochrone = shapely.LineString(np.c_[app_mags[:, 0] - app_mags[:, 2], app_mags[:, 0]])
# iso_tight = isochrone.buffer(0.1)
iso_medium = isochrone.buffer(0.3)
# iso_loose = isochrone.buffer(0.5)


af = asdf.AsdfFile()
af["isochrone_data"] = asdict(mags_data)
af["isochrone_data"].pop("_n2k")
# Masks
af["isochrone_"] = np.c_[isochrone.xy]
af["isochrone_medium"] = np.c_[iso_medium.exterior.xy]

af.write_to(paths.data / "gd1" / "isochrone.asdf")
af.close()
