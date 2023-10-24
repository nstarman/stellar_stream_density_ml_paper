"""Plot GD1 Likelihoods."""

import sys
from io import StringIO

import numpy as np
from astropy.table import QTable, Table
from astropy.units import Quantity
from showyourwork.paths import user as user_paths

from stream_ml.core.typing import Array

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

(paths.output / "gd1").mkdir(exist_ok=True, parents=True)

# =============================================================================

cps_stream = QTable.read(paths.data / "gd1" / "control_points_stream.ecsv")
cps_spur = QTable.read(paths.data / "gd1" / "control_points_spur.ecsv")
cps_dist = QTable.read(paths.data / "gd1" / "control_points_distance.ecsv")


def low_high(vs: Array, es: Array) -> list[str]:
    """Process to +/- error string."""
    vs = Quantity(vs).value
    es = Quantity(es).value
    return [
        (
            "${\\color{gray}\\big(}_{"
            + lower
            + "}^{"
            + ("" if len(lower) == len(upper) else "\\phantom{+}")
            + upper
            + "}{\\color{gray}\\big)}$"
            if not np.isnan(v)
            else ""
        )
        for v, lower, upper in (
            (v, f"{v - e:0.2f}", f"{v + e:0.2f}") for v, e in zip(vs, es, strict=True)
        )
    ]


table = Table()

table["component"] = ["stream"] * len(cps_stream) + ["spur"] * len(cps_spur)
table[r"$\phi_1 \,[\rm{\degree}]$"] = np.concatenate(
    (cps_stream["phi1"], cps_spur["phi1"])
)
table[r"$\phi_2 \,[\rm{\degree}]$"] = low_high(
    cps_stream["phi2"], cps_stream["w_phi2"]
) + low_high(cps_spur["phi2"], cps_spur["w_phi2"])
table[r"$\mu_{\phi_1} \,[\frac{\rm{mas}}{\rm{yr}}]$"] = low_high(
    cps_stream["pm_phi1"], cps_stream["w_pm_phi1"]
) + low_high(cps_spur["pm_phi1"], cps_spur["w_pm_phi1"])
table[r"$\mu \,[\rm{mag}]$"] = " " * 74
table[r"$(\simeq \parallax \,\unit{mas})$"] = " " * 60

for r in cps_dist:
    idx = (table[r"$\phi_1 \,[\rm{\degree}]$"] == r["phi1"]) & (
        table["component"] == "stream"
    )

    if np.any(idx):
        table[r"$\mu \,[\rm{mag}]$"][idx] = low_high([r["distmod"]], [r["w_distmod"]])[
            0
        ]
        table[r"$(\simeq \parallax \,\unit{mas})$"][idx] = low_high(
            [r["parallax"]], [r["w_parallax"]]
        )[0]

    else:
        table.add_row(
            {
                "$\\phi_1 \\,[\rm{\\degree}]$": r["phi1"],
                r"$\mu \,[\rm{mag}]$": low_high([r["distmod"]], [r["w_distmod"]])[0],
                r"$(\simeq \parallax \,\unit{mas})$": low_high(
                    [r["parallax"]], [r["w_parallax"]]
                )[0],
            }
        )

table.sort(r"$\phi_1 \,[\rm{\degree}]$")

# -----------------------------------------------------------------------------
# Write the table

preamble = r"""
\centering
\setlength{\tabcolsep}{0pt}
\newcommand\capitem{\\$\phantom{+}\ast$\ }
"""

caption = r"""%
    Stream Track Regions Prior: %
    This table includes all the region priors used to guide the model towards
    the known stream track. The model will converge to the region $_{\rm
    minimum}^{\rm maximum}$.
    See \autoref{sub:methods:priors:track_region_prior} for details.
"""

write_kwargs = {
    "format": "ascii.latex",
    "overwrite": True,
    "caption": caption,
    "latexdict": {
        "tabletype": "table",
        "preamble": preamble[1:-1],
        "col_align": r"@{}r<{\hspace{7pt}}*{4}{r<{\hspace{7pt}}}l<{\hspace{7pt}}@{}",
        "header_start": r"\toprule",
        "header_end": r"\midrule",
        "data_end": r"\bottomrule\bottomrule",
    },
}

# Serialize to a string
out = StringIO()
table.write(out, **write_kwargs)
lines = out.getvalue().split("\n")

# remove units
lines.pop(lines.index(r" & $\mathrm{{}^{\circ}}$ &  &  &  &  \\"))

# color rows
lines.insert(lines.index("\\end{tabular}") + 1, "}")
index = lines.index("\\toprule")
lines.insert(index - 1, r"{\rowcolors{2}{gray!10}{white!10}")
lines.insert(index - 1, r"\label{tab:gd1_track_prior}")

# Save to disk
with (paths.output / "gd1" / "control_points.tex").open("w") as f:
    f.write("\n".join(lines))
