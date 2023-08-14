"""write variable 'nstream_variable.txt' to disk."""

import asdf
from showyourwork.paths import user as Paths  # noqa: N812

paths = Paths()

# =============================================================================
# Load Data

with asdf.open(
    paths.data / "mock" / "data.asdf", lazy_load=False, copy_arrays=True
) as af:
    n_stream = af["n_stream"]

with (paths.data / "mock" / "nstream_variable.txt").open("w") as f:
    f.write(f"{n_stream}")
