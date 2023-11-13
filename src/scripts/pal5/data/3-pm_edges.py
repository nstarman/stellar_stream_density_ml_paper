"""PM-edges."""

import astropy.units as u
from astropy.table import QTable
from showyourwork.paths import user as user_paths

paths = user_paths()

##############################################################################

table = QTable(
    rows=[
        ["med_icrs", *(-4, -1) * u.mas / u.yr, *(-4, -1) * u.mas / u.yr],
    ],
    names=("label", "pm_ra_min", "pm_ra_max", "pm_dec_min", "pm_dec_max"),
    dtype=(str, float, float, float, float),
    units=(None, u.mas / u.yr, u.mas / u.yr, u.mas / u.yr, u.mas / u.yr),
    meta={
        "pm_ra_min": "$\\mu_{\\phi_1}\\cos{\\phi_2}$, not reflex corrected",
        "pm_ra_max": "$\\mu_{\\phi_1}\\cos{\\phi_2}$, not reflex corrected",
        "pm_dec_min": "$\\mu_{\\phi_2}$, not reflex corrected",
        "pm_dec_max": "$\\mu_{\\phi_2}$, not reflex corrected",
    },
)
table.add_index("label", unique=True)

table.write(paths.data / "pal5" / "pm_edges.ecsv", overwrite=True)
