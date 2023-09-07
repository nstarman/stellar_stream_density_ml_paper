"""Setup."""


import asdf
import brutus.filters
import brutus.seds
import numpy as np
import shapely
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
    feh=-1.35,
    dist=7.8e3,
    loga=np.log10(12e9),
)
abs_mags = abs_mags[np.all(np.isfinite(abs_mags), axis=1)]

mags_data = sml.Data(abs_mags, names=[n.removeprefix("PS_") for n in filters])

isochrone = shapely.LineString(np.c_[abs_mags[:, 0] - abs_mags[:, 2], abs_mags[:, 0]])
iso_tight = isochrone.buffer(0.1)
iso_medium = isochrone.buffer(0.3)
iso_loose = isochrone.buffer(0.5)


af = asdf.AsdfFile()
af["isochrone"] = np.c_[isochrone.xy]
af["isochrone_data"] = {"value": mags_data.array, "names": mags_data.names}
# Masks
af["isochrone_tight"] = np.c_[iso_tight.exterior.xy]
af["isochrone_medium"] = np.c_[iso_medium.exterior.xy]
af["isochrone_loose"] = np.c_[iso_loose.exterior.xy]

af.write_to(paths.data / "gd1" / "isochrone.asdf")
af.close()
