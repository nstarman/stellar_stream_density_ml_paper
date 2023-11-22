"""Download Nerual Network files for brutus."""

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


if snkmk["load_from_static"] and (paths.static / "brutus").exists():
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
