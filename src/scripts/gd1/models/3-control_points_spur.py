"""GD1 spur control points."""


import astropy.units as u
from astropy.table import QTable
from showyourwork.paths import user as user_paths

paths = user_paths()

##############################################################################


# Values from Price-Whelan and Bonaca 2018
table = QTable(
    # fmt: off
    rows=[
        [-35.0 * u.deg, 1.3 * u.deg, 0.85 * u.deg, -12.95 * u.mas / u.yr, 2.0 * u.mas / u.yr],  # noqa: E501
        [-30.0 * u.deg, 1.3 * u.deg, 0.85 * u.deg, -12.6 * u.mas / u.yr, 2.0 * u.mas / u.yr],  # noqa: E501
        [-20 * u.deg, 1.5 * u.deg, 0.85 * u.deg, -11.5 * u.mas / u.yr, 2.0 * u.mas / u.yr],  # noqa: E501
    ],
    # fmt: on
    names=("phi1", "phi2", "w_phi2", "pm_phi1", "w_pm_phi1"),
    units=(u.deg, u.deg, u.deg, u.mas / u.yr, u.mas / u.yr),
    meta={},
)

table.write(paths.data / "gd1" / "control_points_spur.ecsv", overwrite=True)
