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
        paths.static / "brutus" / "MIST_1.2_iso_vvcrit0.0.h5",
        paths.data / "brutus" / "MIST_1.2_iso_vvcrit0.0.h5",
        follow_symlinks=True,
    )
    sys.exit(0)

brutus.utils.fetch_isos(target_dir=paths.data / "brutus", iso="MIST_1.2_vvcrit0.0")
shutil.copyfile(
    paths.data / "brutus" / "MIST_1.2_iso_vvcrit0.0.h5",
    paths.static / "brutus" / "MIST_1.2_iso_vvcrit0.0.h5",
    follow_symlinks=True,
)
