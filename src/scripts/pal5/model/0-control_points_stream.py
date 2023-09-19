"""Plot results."""


import sys

import astropy.units as u
import galstreams
import numpy as np
from astropy.table import QTable
from interpolated_coordinates import InterpolatedSkyCoord
from showyourwork.paths import user as user_paths

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.pal5.frames import pal5_frame as frame

table = QTable(
    np.empty((11, 3)),
    names=("phi1", "phi2", "w_phi2"),
    units=(u.deg, u.deg, u.deg),
    meta={},
)

# Adding in regularly spaced points from :mod:`galstreams`
pal5 = galstreams.MWStreams()["Pal5-PW19"]
track = pal5.track.transform_to(frame)
itrack = InterpolatedSkyCoord(track, affine=track.phi1)

x = np.linspace(track.phi1.min(), track.phi1.max(), len(table) - 1)

# ([progenitor], [stream])
table["phi1"] = np.concatenate(([0], x))
table["phi2"] = np.concatenate(([0], itrack(x).phi2))
table["w_phi2"] = np.concatenate(([0.1], np.zeros_like(x.value) + 1)) << u.deg


table.write(paths.data / "pal5" / "control_points_stream.ecsv", overwrite=True)


##############################################################################
