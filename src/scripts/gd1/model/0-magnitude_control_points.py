"""Plot results."""

import sys
from pathlib import Path

import astropy.units as u
from astropy.table import QTable

sys.path.append(Path(__file__).parents[3].as_posix())
# isort: split

from scripts import paths

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

table.write(paths.data / "gd1" / "magnitude_control_points.ecsv", overwrite=True)
