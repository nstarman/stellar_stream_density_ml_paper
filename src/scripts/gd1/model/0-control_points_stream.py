"""GD1 stream control points."""


import astropy.units as u
import numpy as np
from astropy.table import QTable
from showyourwork.paths import user as user_paths

paths = user_paths()

##############################################################################

MAS_YR = u.mas / u.yr

table = QTable(
    # fmt: off
    rows=[
        [-70.0 * u.deg, np.NaN * u.deg, np.NaN * u.deg, -11.2 * MAS_YR, 1 * MAS_YR],  # noqa: E501
        [-60.0 * u.deg, -0.8 * u.deg, 1.75 * u.deg, -12.5 * MAS_YR, 0.75 * MAS_YR],  # noqa: E501
        [-50.0 * u.deg, -0.0 * u.deg, 1.75 * u.deg, -13.3 * MAS_YR, 0.75 * MAS_YR],  # noqa: E501
        [-40.0 * u.deg, -0.0 * u.deg, 1.75 * u.deg, -13.3 * MAS_YR, 0.75 * MAS_YR],  # noqa: E501
        [-30.0 * u.deg, -0.0 * u.deg, 1.75 * u.deg, -12.6 * MAS_YR, 0.75 * MAS_YR],  # noqa: E501
        [-10. * u.deg, np.NaN * u.deg, np.NaN * u.deg, -10 * MAS_YR, 1.75 * MAS_YR],
    ],
    # fmt: on
    names=("phi1", "phi2", "w_phi2", "pm_phi1", "w_pm_phi1"),
    units=(u.deg, u.deg, u.deg, MAS_YR, MAS_YR),
    meta={},
)

table.write(paths.data / "gd1" / "control_points_stream.ecsv", overwrite=True)
