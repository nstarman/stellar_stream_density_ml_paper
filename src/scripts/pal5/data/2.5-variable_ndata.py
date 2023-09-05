"""Combine data fields into one dataset."""

from astropy.table import QTable
from showyourwork.paths import user as Paths

paths = Paths()

table = QTable.read(paths.data / "pal5" / "gaia_ps1_xm.asdf")

with (paths.output / "pal5" / "ndata_variable.txt").open("w") as f:
    f.write(f"{len(table):,}")
