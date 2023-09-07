"""Plot Pal5 Likelihoods."""

import sys

import numpy as np
from astropy.table import QTable
from numpy.lib.recfunctions import structured_to_unstructured
from showyourwork.paths import user as user_paths

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.as_posix())
# isort: split

from scripts.pal5.datasets import table as data_table

(paths.output / "pal5").mkdir(exist_ok=True, parents=True)
rng = np.random.default_rng(42)

# =============================================================================

member_liks = QTable.read(paths.data / "pal5" / "membership_likelhoods.ecsv")

sel = member_liks["stream (95%)"] > 0.8

table = QTable()
table[r"\texttt{source\_id}"] = data_table["source_id"][sel]

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

table[r"${\rm dim}(\boldsymbol{x})$"] = np.sum(
    ~np.isnan(
        structured_to_unstructured(
            data_table[["ra", "dec", "pmra", "pmdec", "g0", "r0"]][sel].as_array()
        )
    ),
    1,
)


# Likelihoods
def process(value: float, minus: float, plus: float, /) -> str:
    """Process to value +/- error string."""
    dm = np.round(minus - value, 2)
    dp = np.round(plus - value, 2)
    v = np.round(value, 2)

    if v == 0 and dm == 0 and dp == 0:
        return "---"
    if dm == 0 and dp == 0:
        return f"${v:0.2f}$"
    return "".join((f"${v:0.2f}", "_{", f"{dm:+0.2f}", "}^{", f"{dp:+0.2f}", "}$"))


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
table[r"$\mathcal{L}_{\rm background}$"] = [
    process(v, m, p)
    for (v, m, p) in zip(
        member_liks["bkg (MLE)"][sel],
        member_liks["bkg (5%)"][sel],
        member_liks["bkg (95%)"][sel],
        strict=True
    )
]
# fmt: off

# -----------------------------------------------------------------------------
# Write the table

preamble = r"""
\centering
\small
\addtolength{\tabcolsep}{-1pt}
\newcommand\capitem{\\$\phantom{+}\ast$\ }
"""

caption = r"""Subset of Membership Table.
\\ This table includes a selection of candidate member stars for the GD-1
stream, based on the membership likelihoods.  For each star we include the Gaia
DR3 source ID and astrometric solution, the Pan-STARRS1 photometry, and the
membership likelihoods for the stream and background. The likelihoods are
computed using the trained model described in \autoref{ssub:trained_pal5} and we
include a quality flag ${\rm dim}(\boldsymbol{x})$, indicating the number of
features used by the model.  For most stars all features are measured. We use
dropout regularization to estimate the uncertainty in the likelihoods, and
report the 5\% and 95\% quantiles of the distribution, as well as the
dropout-disabled maximum-likelihood estimate (MLE) of the likelihood.  For
convenience we round the likelihoods to 2 decimal places, and only show the
value and uncertainty when it is non-zero.\\ \textit{The full table is
available online.}
"""

write_kwargs = {
    "format": "ascii.latex",
    "overwrite": True,
    "caption": caption,
    "latexdict": {
        "tabletype": "table*",
        "preamble": preamble[1:],
        "col_align": r"@{}r*{4}{c}*{2}{c}c*{2}{l}@{}",
        "header_start": "\n".join(  # noqa: FLY002
            (
                r"\toprule",
                r"\multicolumn{5}{c}{Gaia} & \multicolumn{2}{c}{PS-1} & \multicolumn{1}{c}{} & \multicolumn{3}{c}{Likelihood (${\rm MLE}_{5\%}^{95\%}$)}\\",  # noqa: E501
                r"\cmidrule(lr){1-5} \cmidrule(lr){6-7} \cmidrule(lr){9-11}"
            )
        ),
        "header_end": r"\midrule",
        "data_end": r"\bottomrule\bottomrule",
    },
    "formats": {
        r"$\alpha$ [$\mathrm{{}^{\circ}}$]": "%0.2f",
        r"$\delta$ [$\mathrm{{}^{\circ}}$]": "%0.2f",
    },
}

table.write(paths.output / "pal5" / "member_table_full.tex", **write_kwargs)
