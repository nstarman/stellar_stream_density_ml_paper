"""Mist files for brutus."""

import pathlib
import shutil
import sys
from typing import Any

import dustmaps.bayestar
import dustmaps.config

# Add the parent directory to the path
sys.path.append(pathlib.Path(__file__).parents[2].as_posix())
# isort: split

from scripts import paths

##############################################################################
# Parameters

snkmk: dict[str, Any]
try:
    snkmk = dict(snakemake.params)
except NameError:
    snkmk = {"load_from_static": True, "save_to_static": False}


##############################################################################


DUSTMAPS_DIR = paths.data / "dustmaps"
DUSTMAPS_DIR.mkdir(exist_ok=True)

dustmaps.config.config["data_dir"] = DUSTMAPS_DIR.as_posix()

if snkmk["load_from_static"]:
    shutil.copyfile(
        paths.static / "dustmaps" / "bayestar2019.h5",
        DUSTMAPS_DIR / "bayestar" / "bayestar2019.h5",
    )
else:
    dustmaps.bayestar.fetch(version="bayestar2019")

if snkmk["save_to_static"]:
    shutil.copyfile(
        DUSTMAPS_DIR / "bayestar" / "bayestar2019.h5",
        paths.static / "dustmaps" / "bayestar2019.h5",
    )
