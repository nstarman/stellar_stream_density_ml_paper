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
        [0.0 * u.deg, 0.0 * u.deg, 0.1 * u.deg],  # the progenitor
    ],
    # fmt: on
    names=("phi1", "phi2", "w_phi2"),
    units=(u.deg, u.deg, u.deg),
    meta={},
)

table.write(paths.data / "pal5" / "stream_control_points.ecsv", overwrite=True)
