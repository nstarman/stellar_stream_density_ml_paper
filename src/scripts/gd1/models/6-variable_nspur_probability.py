"""write variable 'nspur_variable.txt' to disk."""

from showyourwork.paths import user as user_paths

paths = user_paths()

output_path = paths.output / "gd1" / "nspur"
output_path.mkdir(parents=True, exist_ok=True)

with (output_path / "minimum_membership_probability.txt").open("w") as f:
    f.write(r"$80\%$")
