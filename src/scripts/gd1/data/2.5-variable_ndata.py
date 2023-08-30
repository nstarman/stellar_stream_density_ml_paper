"""Combine data fields into one dataset."""

import sys
from pathlib import Path

from astropy.table import QTable

sys.path.append(Path(__file__).parents[3].as_posix())
# isort: split

from scripts import paths

table = QTable.read(paths.data / "gd1" / "gaia_ps1_xm.asdf")

with (paths.output / "gd1" / "ndata_variable.txt").open("w") as f:
    f.write(f"{len(table):,}")
