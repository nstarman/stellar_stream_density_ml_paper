"""Combine data fields into one dataset."""


from astropy.table import QTable
from showyourwork.paths import user as user_paths

paths = user_paths()

table = QTable.read(paths.data / "gd1" / "gaia_ps1_xm.asdf")

with (paths.output / "gd1" / "ndata_variable.txt").open("w") as f:
    f.write(f"{len(table):,}")
