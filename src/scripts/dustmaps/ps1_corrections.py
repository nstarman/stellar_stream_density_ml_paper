"""Mist files for brutus."""

import json

from showyourwork.paths import user as user_paths

paths = user_paths()


factors = {
    "g": 3.158,
    "r": 2.617,
    "i": 1.971,
    "z": 1.549,
    "y": 1.263,
}

with (paths.data / "dustmaps" / "ps1_corrections.json").open(mode="w") as f:
    json.dump(factors, f)
