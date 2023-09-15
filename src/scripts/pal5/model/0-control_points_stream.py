"""Plot results."""

import sys

import astropy.units as u
from astropy.table import QTable
from showyourwork.paths import user as user_paths

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split


table = QTable(
    # fmt: off
    rows=[
        [0 * u.deg, 0.0 * u.deg, 0.1 * u.deg],  # the progenitor
        [7 * u.deg, 1.5 * u.deg, 0.75 * u.deg],
    ],
    # fmt: on
    names=("phi1", "phi2", "w_phi2"),
    units=(u.deg, u.deg, u.deg),
    meta={},
)

table.write(paths.data / "pal5" / "control_points_stream.ecsv", overwrite=True)
