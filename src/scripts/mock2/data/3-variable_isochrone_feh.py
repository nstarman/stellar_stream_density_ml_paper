"""write variable 'isochrone_feh_variable.txt' to disk."""

import asdf
from showyourwork.paths import user as user_paths

paths = user_paths()
(paths.output / "mock2").mkdir(parents=True, exist_ok=True)


with asdf.open(
    paths.data / "mock2" / "data.asdf", lazy_load=False, copy_arrays=True
) as af, (paths.output / "mock2" / "isochrone_feh_variable.txt").open("w") as f:
    f.write(f"{af['isochrone_feh']:g}~dex")
