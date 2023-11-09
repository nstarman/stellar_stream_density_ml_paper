"""write variable 'nspur_variable.txt' to disk."""

from astropy.table import QTable
from showyourwork.paths import user as user_paths

paths = user_paths()

save_path = paths.data / "gd1"
save_path.mkdir(parents=True, exist_ok=True)

output_path = paths.output / "gd1" / "nspur"
output_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Load Data

lik_tbl = QTable.read(save_path / "membership_likelhoods.ecsv")

is_strm = (lik_tbl["spur (50%)"] > 0.80) & (lik_tbl["spur.ln-weight"].mean(1) > -6)

with (output_path / "nspur_variable.txt").open("w") as f:
    f.write(f"${sum(is_strm):d}$")

# TODO: get these into the DAG, but can't handle folders yet

with (output_path / "posterior_percentile.txt").open("w") as f:
    f.write(r"$50$th")

with (output_path / "minimum_membership_probability.txt").open("w") as f:
    f.write(r"$80\%$")
