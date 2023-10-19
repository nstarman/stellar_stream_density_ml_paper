"""Pal5 control points."""

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

(paths.output / "pal5").mkdir(exist_ok=True, parents=True)

# =============================================================================

cps_stream = QTable.read(paths.data / "pal5" / "control_points_stream.ecsv")


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

table[r"$\phi_1 \,[\rm{\degree}]$"] = cps_stream["phi1"].value
table[r"$\phi_2 \,[\rm{\degree}]$"] = low_high(cps_stream["phi2"], cps_stream["w_phi2"])
table[r"$\mu_{\phi_1} \,[\frac{\rm{mas}}{\rm{yr}}]$"] = low_high(
    cps_stream["pmphi1"], cps_stream["w_pmphi1"]
)
table[r"$\mu_{\phi_2} \,[\frac{\rm{mas}}{\rm{yr}}]$"] = low_high(
    cps_stream["pmphi2"], cps_stream["w_pmphi2"]
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
    Stream Track Regions Priors for Pal\,5: %
    This table includes all the region priors used to guide the model towards
    the known stream track. The model will converge to the region $_{\rm
    minimum}^{\rm maximum}$. The regions are determined by the stream track
    from \package{galstreams} \citep{Mateu2022}, with a width significantly
    larger than the stream width. In the kinematics only the progenitor is used
    to guide the model.
"""

write_kwargs = {
    "format": "ascii.latex",
    "overwrite": True,
    "caption": caption,
    "latexdict": {
        "tabletype": "table",
        "preamble": preamble[1:-1],
        "col_align": r"@{}*{4}{r<{\hspace{7pt}}}@{}",
        "header_start": r"\toprule",
        "header_end": r"\midrule",
        "data_end": r"\bottomrule\bottomrule",
    },
    "formats": {
        r"$\phi_1 \,[\rm{\degree}]$": "%0.2f",
    },
}

# Serialize to a string
out = StringIO()
table.write(out, **write_kwargs)
lines = out.getvalue().split("\n")

# color rows
lines.insert(lines.index("\\end{tabular}") + 1, "}")
index = lines.index("\\toprule")
lines.insert(index - 1, r"{\rowcolors{2}{gray!10}{white!10}")
lines.insert(index - 1, r"\label{tab:pal5_track_prior}")

# Save to disk
with (paths.output / "pal5" / "control_points.tex").open("w") as f:
    f.write("\n".join(lines))
