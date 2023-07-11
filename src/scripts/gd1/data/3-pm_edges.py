"""Define the proper motion data mask boundaries for GD-1."""

import sys

import astropy.units as u
import galstreams
from astropy.table import QTable
from showyourwork.paths import user as user_paths

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.gd1.frames import gd1_frame
from scripts.helper import make_vertices

##############################################################################

mws = galstreams.MWStreams()
gd1 = mws["GD-1-I21"]
gd1_sc = gd1.track.transform_to(gd1_frame)[::100]


path_pmphi1_tight = make_vertices(gd1_sc.phi1.degree, gd1_sc.pm_phi1_cosphi2.value, 4)
path_pmphi2_tight = make_vertices(gd1_sc.phi1.degree, gd1_sc.pm_phi2.value, 2)

path_pmphi1_medium = make_vertices(gd1_sc.phi1.degree, gd1_sc.pm_phi1_cosphi2.value, 6)
path_pmphi2_medium = make_vertices(gd1_sc.phi1.degree, gd1_sc.pm_phi2.value, 3)

table = QTable()
table["phi1"] = path_pmphi1_tight[:, 0] * u.deg
table["pm_phi1_cosphi2_tight"] = path_pmphi1_tight[:, 1] * u.mas / u.yr
table["pm_phi2_tight"] = path_pmphi2_tight[:, 1] * u.mas / u.yr
table["pm_phi1_cosphi2_medium"] = path_pmphi1_medium[:, 1] * u.mas / u.yr
table["pm_phi2_medium"] = path_pmphi2_medium[:, 1] * u.mas / u.yr

table.write(paths.data / "gd1" / "pm_edges.ecsv", overwrite=True)
