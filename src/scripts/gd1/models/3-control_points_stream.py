"""GD1 stream control points."""

import sys

import astropy.units as u
import galstreams
import numpy as np
from astropy.table import QTable
from scipy.interpolate import InterpolatedUnivariateSpline as IUS  # noqa: N817
from showyourwork.paths import user as user_paths

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.gd1.frames import gd1_frame as frame

##############################################################################

# Galstreams
mws = galstreams.MWStreams(verbose=False, implement_Off=False)
gd1 = mws["GD-1-I21"].track.transform_to(frame)

# Stream Control table
table = QTable()

# phi1
phi1s = np.arange(-90, 10 + 10, 10)
table["phi1"] = phi1s * u.deg
table.add_index("phi1")

# phi2
table["phi2"] = IUS(gd1.phi1.wrap_at("180d").degree, gd1.phi2.degree)(phi1s) * u.deg
table["w_phi2"] = 1.75 * u.deg

# pm-phi1
MAS_YR = u.mas / u.yr
table["pm_phi1"] = np.NaN * MAS_YR
table["w_pm_phi1"] = np.NaN * MAS_YR

table.loc[-70]["pm_phi1", "w_pm_phi1"] = (-11.2, 1) * MAS_YR
table.loc[-60]["pm_phi1", "w_pm_phi1"] = (-12.5, 0.80) * MAS_YR
table.loc[-50]["pm_phi1", "w_pm_phi1"] = (-13.3, 0.75) * MAS_YR
table.loc[-40]["pm_phi1", "w_pm_phi1"] = (-13.3, 0.75) * MAS_YR
table.loc[-30]["pm_phi1", "w_pm_phi1"] = [-12.6, 0.75] * MAS_YR
table.loc[-10]["pm_phi1", "w_pm_phi1"] = (-10.0, 1.75) * MAS_YR


table.add_row([5 * u.deg, np.NaN * u.deg, np.NaN * u.deg, -7.4 * MAS_YR, 1.75 * MAS_YR])

# re-sort
table.sort("phi1")

# save
table.write(paths.data / "gd1" / "control_points_stream.ecsv", overwrite=True)
