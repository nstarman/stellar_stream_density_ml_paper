"""write variable 'nstream_variable.txt' to disk."""

import asdf
from showyourwork.paths import user as Paths  # noqa: N812

paths = Paths()

# =============================================================================
# Load Data

with asdf.open(
    paths.data / "mock" / "data.asdf", lazy_load=False, copy_arrays=True
) as af, (paths.data / "mock" / "nstream_variable.txt").open("w") as f:
    f.write(f"{af['n_stream']}")
