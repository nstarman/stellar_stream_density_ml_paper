"""Mist files for brutus."""

import pathlib
import shutil
import sys

import dustmaps.bayestar
import dustmaps.config

# Add the parent directory to the path
sys.path.append(pathlib.Path(__file__).parent.parent.as_posix())
# isort: split

import paths  # noqa: E402

##############################################################################
# Parameters

try:
    snkmkp = snakemake.params
except NameError:
    snkmkp = {"load_from_static": True}


##############################################################################


DUSTMAPS_DIR = paths.data / "dustmaps"
DUSTMAPS_DIR.mkdir(exist_ok=True)

dustmaps.config.config["data_dir"] = DUSTMAPS_DIR.as_posix()

if snkmkp["load_from_static"]:
    shutil.copyfile(
        paths.static / "dustmaps" / "bayestar2019.h5",
        DUSTMAPS_DIR / "bayestar" / "bayestar2019.h5",
    )
else:
    dustmaps.bayestar.fetch(version="bayestar2019")
