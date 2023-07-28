"""PM-edges."""

import sys
from pathlib import Path

import astropy.units as u
from astropy.table import QTable

sys.path.append(Path(__file__).parents[3].as_posix())
# isort: split

from scripts import paths

##############################################################################

table = QTable(
    rows=[
        ["tight", *(-15, -10) * u.mas / u.yr, *(-4.5, -2) * u.mas / u.yr],
        ["medium", *(-15, -5) * u.mas / u.yr, *(-5, -1) * u.mas / u.yr],
        ["loose", *(-15, -3.5) * u.mas / u.yr, *(-6, -1) * u.mas / u.yr],
    ],
    names=("label", "pm_phi1_min", "pm_phi1_max", "pm_phi2_min", "pm_phi2_max"),
    dtype=(str, float, float, float, float),
    units=(None, u.mas / u.yr, u.mas / u.yr, u.mas / u.yr, u.mas / u.yr),
    meta={
        "pm_phi1_min": "$\\mu_{\\phi_1}\\cos{\\phi_2}$, not reflex corrected",
        "pm_phi1_max": "$\\mu_{\\phi_1}\\cos{\\phi_2}$, not reflex corrected",
        "pm_phi2_min": "$\\mu_{\\phi_2}$, not reflex corrected",
        "pm_phi2_max": "$\\mu_{\\phi_2}$, not reflex corrected",
    },
)
table.add_index("label", unique=True)

table.write(paths.data / "gd1" / "pm_edges.ecsv")
