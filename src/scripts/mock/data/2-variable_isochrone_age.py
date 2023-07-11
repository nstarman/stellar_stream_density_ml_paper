"""write variable 'isochrone_age_variable.txt' to disk."""

import asdf
from showyourwork.paths import user as user_paths

paths = user_paths()
(paths.output / "mock").mkdir(parents=True, exist_ok=True)

# =============================================================================
# Load Data

with asdf.open(
    paths.data / "mock" / "data.asdf", lazy_load=False, copy_arrays=True
) as af, (paths.output / "mock" / "isochrone_age_variable.txt").open("w") as f:
    f.write(f"{af['isochrone_age'].value:g} {af['isochrone_age'].unit}")
