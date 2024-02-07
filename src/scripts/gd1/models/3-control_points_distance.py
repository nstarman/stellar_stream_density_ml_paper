"""GD-1 distance control points."""

import astropy.units as u
import numpy as np
from astropy.coordinates import Distance
from astropy.table import QTable
from showyourwork.paths import user as user_paths

paths = user_paths()

##############################################################################


table = QTable(
    # fmt: off
    rows=[
        [-70.0 * u.deg, 14.5 * u.mag, 0.3],
        [-40.0 * u.deg, 14.4 * u.mag, 0.3],
        [0.0 * u.deg, 15 * u.mag, 0.3],
    ],
    # fmt: on
    names=("phi1", "distmod", "w_distmod"),
    units=(u.deg, u.mag, u.mag),
    meta={},
)

# Convert to parallax
table["parallax"] = Distance(distmod=table["distmod"]).parallax
table["w_parallax"] = (
    np.abs(
        (Distance(distmod=table["distmod"] + table["w_distmod"]).parallax)
        - (Distance(distmod=table["distmod"] - table["w_distmod"]).parallax)
    )
    / 2
    * 1.1
)

# Convert to distance
table["distance"] = Distance(distmod=table["distmod"])
table["w_distance"] = (
    np.abs(
        (Distance(distmod=table["distmod"] + table["w_distmod"]))
        - (Distance(distmod=table["distmod"] - table["w_distmod"]))
    )
    / 2
)

# Save
table.write(paths.data / "gd1" / "control_points_distance.ecsv", overwrite=True)
