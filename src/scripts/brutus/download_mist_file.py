"""Download MIST files for brutus."""

import shutil
import sys
from typing import Any

import brutus.utils
from showyourwork.paths import user as user_paths

paths = user_paths()

##############################################################################

snkmk: dict[str, Any]
try:
    snkmk = dict(snakemake.params)
except NameError:
    snkmk = {"load_from_static": True, "save_to_static": False}


if snkmk["load_from_static"]:
    shutil.copyfile(
        paths.static / "brutus" / "MIST_1.2_iso_vvcrit0.0.h5",
        paths.data / "brutus" / "MIST_1.2_iso_vvcrit0.0.h5",
        follow_symlinks=True,
    )
    sys.exit(0)

brutus.utils.fetch_isos(target_dir=paths.data / "brutus", iso="MIST_1.2_vvcrit0.0")

if snkmk["save_to_static"]:
    shutil.copyfile(
        paths.data / "brutus" / "MIST_1.2_iso_vvcrit0.0.h5",
        paths.static / "brutus" / "MIST_1.2_iso_vvcrit0.0.h5",
        follow_symlinks=True,
    )
