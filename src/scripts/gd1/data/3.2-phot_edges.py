"""Setup."""

import sys
from pathlib import Path

import asdf
import brutus.filters
import brutus.seds
import numpy as np
import shapely

# isort: split

sys.path.append(Path(__file__).parent.parent.parent.as_posix())
# isort: split

import paths  # noqa: E402

##############################################################################


# Build isochrone
mags, _, _ = brutus.seds.Isochrone(
    filters=brutus.filters.ps[:-2],  # (g, r)
    nnfile=(paths.data / "brutus" / "nn_c3k.h5").resolve(),
    mistfile=(paths.data / "brutus" / "MIST_1.2_iso_vvcrit0.0.h5").resolve(),
).get_seds(
    eep=np.linspace(202, 600, 5000),
    apply_corr=True,
    feh=-1.35,
    dist=7.8e3,
    loga=np.log10(12e9),
)
mags = mags[np.all(np.isfinite(mags), axis=1)]

isochrone = shapely.LineString(np.c_[mags[:, 0] - mags[:, 2], mags[:, 0]])
iso_tight = isochrone.buffer(0.1)
iso_medium = isochrone.buffer(0.3)
iso_loose = isochrone.buffer(0.5)


af = asdf.AsdfFile()
af["isochrone"] = np.c_[isochrone.xy]
af["isochrone_tight"] = np.c_[iso_tight.exterior.xy]
af["isochrone_medium"] = np.c_[iso_medium.exterior.xy]
af["isochrone_loose"] = np.c_[iso_loose.exterior.xy]
af.write_to(paths.data / "gd1" / "isochrone.asdf")
af.close()
