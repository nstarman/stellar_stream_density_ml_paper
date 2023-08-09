"""Plot GD1 Likelihoods."""

import sys
from pathlib import Path

import numpy as np
from astropy.table import QTable

# Add the parent directory to the path
sys.path.append(Path(__file__).parents[3].as_posix())
# isort: split

from scripts import paths
from scripts.gd1.datasets import table as data_table

# =============================================================================

member_liks = QTable.read(paths.data / "gd1" / "membership_likelhoods.ecsv")

sel = (member_liks["stream (95%)"] > 0.8) | (member_liks["spur (95%)"] > 0.8)

table = QTable()
table["Source ID"] = data_table["source_id"][sel]

# Astrometry
table[r"$\alpha$ [$\mathrm{{}^{\circ}}$]"] = data_table["ra"][sel].to_value("deg")
table[r"$\delta$ [$\mathrm{{}^{\circ}}$]"] = data_table["dec"][sel].to_value("deg")
table[r"$\mu_{\alpha}^{*}$ [$\frac{\rm{mas}}{\rm{yr}}$]"] = [
    f"${v:0.2f} \\pm {e:0.2f}$"
    for v, e in zip(
        data_table["pmra"][sel].to_value("mas/yr"),
        data_table["pmra_error"][sel].to_value("mas/yr"),
        strict=True,
    )
]
table[r"$\mu_{\delta}$ [$\frac{\rm{mas}}{\rm{yr}}$]"] = [
    rf"${v:0.2f} \pm {e:0.2f}$"
    for v, e in zip(
        data_table["pmdec"][sel].to_value("mas/yr"),
        data_table["pmdec_error"][sel].to_value("mas/yr"),
        strict=True,
    )
]

# Photometry
table["g [mag]"] = [
    rf"${v:0.2f} \pm {e:0.2f}$"
    for v, e in zip(
        data_table["g0"][sel].to_value("mag"),
        data_table["ps1_g_error"][sel].to_value("mag"),
        strict=True,
    )
]
table["r [mag]"] = [
    rf"${v:0.2f} \pm {e:0.2f}$"
    for v, e in zip(
        data_table["r0"][sel].to_value("mag"),
        data_table["ps1_r_error"][sel].to_value("mag"),
        strict=True,
    )
]


# Likelihoods
def process(value: float, minus: float, plus: float, /) -> str:
    """Process to value +/- error string."""
    dm = np.round(minus - value, 2)
    dp = np.round(plus - value, 2)

    if value == 0 and dm == 0 and dp == 0:
        return ""
    return "".join(
        (f"${np.round(value, 2):0.2f}", "^{", f"{dp:+0.2f}", "}_{", f"{dm:+0.2f}", "}$")
    )


# fmt: off
table[r"$\mathcal{L}_{\rm stream}$"] = [
    process(v, m, p)
    for (v, m, p) in zip(
        member_liks["stream (MLE)"][sel],
        member_liks["stream (5%)"][sel],
        member_liks["stream (95%)"][sel],
        strict=True
    )
]
table[r"$\mathcal{L}_{\rm spur}$"] = [
    process(v, m, p)
    for (v, m, p) in zip(
        member_liks["spur (MLE)"][sel],
        member_liks["spur (5%)"][sel],
        member_liks["spur (95%)"][sel],
        strict=True
    )
]
# fmt: off

table[:20].write(
    paths.output / "gd1_members.tex",
    format="ascii.latex",
    overwrite=True,
    caption="Membership Table.",
    latexdict={
        "tabletype": "table*",
        "preamble": r"\centering",
        "col_align": r"@{}rcccccccc@{}",
        "header_start": "\n".join(  # noqa: FLY002
            (
                r"\toprule",
                r"& \multicolumn{4}{c}{Gaia} & \multicolumn{2}{c}{PS-1} & \multicolumn{2}{c}{Likelihood}\\",  # noqa: E501
                r"\cmidrule(lr){2-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}"
            )
        ),
        "header_end": r"\midrule",
        "data_end": r"\bottomrule",
    },
    formats={
        r"$\alpha$ [$\mathrm{{}^{\circ}}$]": "%0.2f",
        r"$\delta$ [$\mathrm{{}^{\circ}}$]": "%0.2f",
    },
)
