"""Mist files for brutus."""

import pathlib
import shutil
import sys
from typing import Any

import brutus.utils

# Add the parent directory to the path
sys.path.append(pathlib.Path(__file__).parents[2].as_posix())
# isort: split

from scripts import paths

##############################################################################

snkmk: dict[str, Any]
try:
    snkmk = dict(snakemake.params)
except NameError:
    snkmk = {"load_from_static": True, "save_to_static": False}


if snkmk["load_from_static"]:
    shutil.copyfile(
        paths.static / "brutus" / "nn_c3k.h5",
        paths.data / "brutus" / "nn_c3k.h5",
        follow_symlinks=True,
    )
    sys.exit(0)


brutus.utils.fetch_nns(target_dir=paths.data / "brutus", model="c3k")

if snkmk["save_to_static"]:
    shutil.copyfile(
        paths.data / "brutus" / "nn_c3k.h5",
        paths.static / "brutus" / "nn_c3k.h5",
        follow_symlinks=True,
    )
