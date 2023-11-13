"""Off-stream selection."""

import sys

import galstreams
import numpy as np
import shapely.geometry as sg
from showyourwork.paths import user as user_paths

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.pal5.frames import pal5_frame

###############################################################################

pal5 = galstreams.MWStreams()["Pal5-PW19"]
track = pal5.track.transform_to(pal5_frame)
footprint = sg.LineString(np.c_[track.phi1.degree, track.phi2.degree]).buffer(1.5)

with paths.data / "pal5" / "footprint.npz" as f:
    np.savez_compressed(f, footprint=footprint.exterior.xy)
