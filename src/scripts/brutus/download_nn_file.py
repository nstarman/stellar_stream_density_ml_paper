"""Mist files for brutus."""

import pathlib
import shutil
import sys

import brutus.utils

# Add the parent directory to the path
sys.path.append(pathlib.Path(__file__).parent.parent.as_posix())
# isort: split

import paths  # noqa: E402

##############################################################################

try:
    snkmkp = snakemake.params
except NameError:
    snkmkp = {"load_from_static": True}


if snkmkp["load_from_static"]:
    shutil.copyfile(
        paths.static / "brutus" / "nn_c3k.h5",
        paths.data / "brutus" / "nn_c3k.h5",
        follow_symlinks=True,
    )
    sys.exit(0)


brutus.utils.fetch_nns(target_dir=paths.data / "brutus", model="c3k")
shutil.copyfile(
    paths.data / "brutus" / "nn_c3k.h5",
    paths.static / "brutus" / "nn_c3k.h5",
    follow_symlinks=True,
)
