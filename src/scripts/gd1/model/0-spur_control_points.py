"""Plot results."""

import sys
from pathlib import Path

import astropy.units as u
from astropy.table import QTable

sys.path.append(Path(__file__).parent.parent.parent.as_posix())
# isort: split

import paths  # noqa: E402

##############################################################################


# Values from Price-Whelan and Bonaca 2018
table = QTable(
    # fmt: off
    rows=[
        [-35.0 * u.deg, 1.3 * u.deg, 0.85 * u.deg, -12.95 * u.mas / u.yr, 3 * u.mas / u.yr],  # noqa: E501
        [-30.0 * u.deg, 1.3 * u.deg, 0.85 * u.deg, -12.6 * u.mas / u.yr, 3 * u.mas / u.yr],  # noqa: E501
        [-20 * u.deg, 1.5 * u.deg, 0.85 * u.deg, -11.5 * u.mas / u.yr, 5 * u.mas / u.yr],  # noqa: E501
    ],
    # fmt: on
    names=("phi1", "phi2", "w_phi2", "pm_phi1", "w_pm_phi1"),
    units=(u.deg, u.deg, u.deg, u.mas / u.yr, u.mas / u.yr),
    meta={},
)

table.write(paths.data / "gd1" / "spur_control_points.ecsv", overwrite=True)
