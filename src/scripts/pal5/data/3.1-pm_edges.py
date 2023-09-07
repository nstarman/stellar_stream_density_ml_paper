"""PM-edges."""

import astropy.units as u
from astropy.table import QTable
from showyourwork.paths import user as user_paths

paths = user_paths()

table = QTable(
    rows=[["tight_icrs", *(-3.5, -2) * u.mas / u.yr, *(-3.5, -2) * u.mas / u.yr]],
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

table.write(paths.data / "pal5" / "pm_edges.ecsv")
