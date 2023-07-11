"""Plot results."""

import sys
from pathlib import Path

import astropy.units as u
from astropy.table import QTable

sys.path.append(Path(__file__).parent.parent.parent.as_posix())
# isort: split

import paths  # noqa: E402

##############################################################################


table = QTable(
    # fmt: off
    rows=[
        [-80.0 * u.deg, -2.24 * u.deg, 1.75 * u.deg, -9.9 * u.mas / u.yr, 0.75 * u.mas / u.yr],  # noqa: E501
        [-70.0 * u.deg, -1.3 * u.deg, 1.75 * u.deg, -11.1 * u.mas / u.yr, 0.75 * u.mas / u.yr],  # noqa: E501
        [-60.0 * u.deg, -0.8 * u.deg, 1.75 * u.deg, -12.5 * u.mas / u.yr, 0.75 * u.mas / u.yr],  # noqa: E501
        [-50.0 * u.deg, -0.0 * u.deg, 1.75 * u.deg, -13.3 * u.mas / u.yr, 0.75 * u.mas / u.yr],  # noqa: E501
        [-40.0 * u.deg, -0.0 * u.deg, 1.75 * u.deg, -13.3 * u.mas / u.yr, 0.75 * u.mas / u.yr],  # noqa: E501
        [-30.0 * u.deg, -0.0 * u.deg, 1.75 * u.deg, -12.6 * u.mas / u.yr, 0.75 * u.mas / u.yr],  # noqa: E501
        [-20.0 * u.deg, 0.09 * u.deg, 1.75 * u.deg, -11.5 * u.mas / u.yr, 0.75 * u.mas / u.yr],  # noqa: E501
    ],
    # fmt: on
    names=("phi1", "phi2", "w_phi2", "pm_phi1", "w_pm_phi1"),
    units=(u.deg, u.deg, u.deg, u.mas / u.yr, u.mas / u.yr),
    meta={},
)

table.write(paths.data / "gd1" / "stream_control_points.ecsv", overwrite=True)
