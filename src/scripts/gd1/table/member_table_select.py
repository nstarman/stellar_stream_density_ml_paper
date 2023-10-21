"""Plot GD1 Likelihoods."""

import sys
from io import StringIO

import numpy as np
from astropy.table import QTable
from numpy.lib.recfunctions import structured_to_unstructured
from showyourwork.paths import user as user_paths

from stream_ml.core.typing import Array

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split
from scripts.gd1.datasets import table as data_table

(paths.output / "gd1").mkdir(exist_ok=True, parents=True)
rng = np.random.default_rng(42)

# =============================================================================

member_liks = QTable.read(paths.data / "gd1" / "membership_likelhoods.ecsv")

sel = (member_liks["stream (95%)"] > 0.8) | (member_liks["spur (95%)"] > 0.8)

# Select some rows
rows = []
# we work within the table selection
strm_mle = member_liks["stream (MLE)"][sel]
spur_mle = member_liks["spur (MLE)"][sel]
# 1 row with highest probability
rows.append(np.argmax(strm_mle))
# 5 rows with a probability > 0.9
prob_idx = np.where(strm_mle > 0.9)[0]
subselect = rng.choice(np.arange(len(prob_idx)), size=5, replace=False, shuffle=False)
rows.extend(prob_idx[subselect])
# 4 rows with a probability < 0.75
prob_idx = np.where(strm_mle < 0.75)[0]
subselect = rng.choice(np.arange(len(prob_idx)), size=4, replace=False, shuffle=False)
rows.extend(prob_idx[subselect])
# 1 row with highest probability > 0.99
rows.append(np.argmax(spur_mle))
# 1 rows with a probability > 0.9 and low stream probability
prob_idx = np.where(
    (strm_mle < 0.75) & (spur_mle > 0.9) & (spur_mle != spur_mle.max())
)[0]
subselect = rng.choice(np.arange(len(prob_idx)), size=1, replace=False, shuffle=False)
rows.extend(prob_idx[subselect])
# # 3 rows with non-zero joint probability  # TODO
# prob_idx = np.where((strm_mle > 0.1) & (spur_mle > 0.6))[0]
# subselect = rng.choice(np.arange(len(prob_idx)), size=3, replace=False, shuffle=False)
# rows.extend(prob_idx[subselect])


def process_lines(vs: Array, es: Array) -> list[str]:
    """Process to value +/- error string."""
    return [
        (rf"${v:0.2f} \pm {e:0.2f}$" if not np.isnan(v) else "---")
        for v, e in zip(vs, es, strict=True)
    ]


table = QTable()
# yada yada the source_id column
# table[r"\texttt{source\_id}"] = [f"...{str(sid)[-4:]}" for sid in table[r"\texttt{source\_id}"]]  # noqa: E501
table[r"\texttt{source\_id}"] = ["---"] * len(rows)

# Astrometry
table[r"$\alpha$ [$\mathrm{{}^{\circ}}$]"] = data_table["ra"][sel][rows].to_value("deg")
table[r"$\delta$ [$\mathrm{{}^{\circ}}$]"] = data_table["dec"][sel][rows].to_value(
    "deg"
)
table[r"$\mu_{\alpha}^{*}$ [$\frac{\rm{mas}}{\rm{yr}}$]"] = process_lines(
    data_table["pmra"][sel][rows].to_value("mas/yr"),
    data_table["pmra_error"][sel][rows].to_value("mas/yr"),
)

table[r"$\mu_{\delta}$ [$\frac{\rm{mas}}{\rm{yr}}$]"] = process_lines(
    data_table["pmdec"][sel][rows].to_value("mas/yr"),
    data_table["pmdec_error"][sel][rows].to_value("mas/yr"),
)

table[r"$\varpi$ [\rm{mas}]"] = process_lines(
    data_table["parallax"][sel][rows].to_value("mas"),
    data_table["parallax_error"][sel][rows].to_value("mas"),
)

# Photometry
table["g [mag]"] = process_lines(
    data_table["g0"][sel][rows].to_value("mag"),
    data_table["g0_error"][sel][rows].to_value("mag"),
)
table["r [mag]"] = process_lines(
    data_table["r0"][sel][rows].to_value("mag"),
    data_table["r0_error"][sel][rows].to_value("mag"),
)

table[r"${\rm dim}(\boldsymbol{x})$"] = np.sum(
    ~np.isnan(
        structured_to_unstructured(
            data_table[["ra", "dec", "parallax", "pmra", "pmdec", "g0", "r0"]][sel][
                rows
            ].as_array()
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
        member_liks["stream (MLE)"][sel][rows],
        member_liks["stream (5%)"][sel][rows],
        member_liks["stream (95%)"][sel][rows],
        strict=True
    )
]
table[r"$\mathcal{L}_{\rm spur}$"] = [
    process(v, m, p)
    for (v, m, p) in zip(
        member_liks["spur (MLE)"][sel][rows],
        member_liks["spur (5%)"][sel][rows],
        member_liks["spur (95%)"][sel][rows],
        strict=True
    )
]
table[r"$\mathcal{L}_{\rm background}$"] = [
    process(v, m, p)
    for (v, m, p) in zip(
        member_liks["bkg (MLE)"][sel][rows],
        member_liks["bkg (5%)"][sel][rows],
        member_liks["bkg (95%)"][sel][rows],
        strict=True
    )
]
# fmt: off

# -----------------------------------------------------------------------------
# Write the table

preamble = r"""
\centering
\small
\setlength{\tabcolsep}{0pt}
\newcommand\capitem{\\$\phantom{+}\ast$\ }
"""

caption = r"""Subset of GD-1 Membership Table.
\\
This table includes a selection of candidate member stars for the GD-1 stream,
based on the membership likelihoods.  For each star we include the Gaia DR3
source ID and astrometric solution, the Pan-STARRS1 photometry, and the
membership likelihoods for the stream, spur, and background.  The likelihoods
are computed using the trained model described in
\autoref{sub:results_gd1:results} and we include a quality flag ${\rm
dim}(\boldsymbol{x})$, indicating the number of features used by the model.  For
most stars all features are measured.  We use dropout regularization to estimate
the uncertainty in the likelihoods, and report the 5\% and 95\% quantiles of the
distribution, as well as the dropout-disabled maximum-likelihood estimate (MLE)
of the likelihood.
\\
We include as interesting cases:
    \capitem{} 1 star with the highest MLE for the stream,
    \capitem{} 5 stars with high stream MLE ($\mathcal{L}^{(S)}_{\rm MLE} > 0.9$),
    \capitem{} 4 stars with low stream MLE, but whose 95\% likelihood is high
               ($\mathcal{L}^{(S)}_{\rm MLE} < 0.75, \mathcal{L}^{(S)}_{\rm 95\%} > 0.8$),
    \capitem{} 1 star with the maximum MLE for the spur,
    \capitem{} 1 star with high spur MLE and low stream MLE
               ($\mathcal{L}^{(spur)}_{\rm MLE} > 0.9, \mathcal{L}^{(S)}_{\rm MLE} < 0.75$),
\\
For convenience we round the likelihoods to 2 decimal places, and only show the
value and uncertainty when it is non-zero.
\\
\textit{The full table, including source ids, is available online.}
"""  # noqa: E501
#     \capitem{} and 3 stars with significant MLE for both the stream and spur
#                ($\mathcal{L}^{(S)}_{\rm MLE} > 0.1, \mathcal{L}^{(spur)}_{\rm MLE} > 0.7$).  # noqa: E501

write_kwargs = {
    "format": "ascii.latex",
    "overwrite": True,
    "caption": caption,
    "latexdict": {
        "tabletype": "table*",
        "preamble": preamble[1:],
        "col_align": (
            r"@{}"
            r"c<{\hspace{7pt}}"  # source_id
            r"*{5}{>{\footnotesize}c<{\hspace{7pt}}}"  # astrometry
            r"*{2}{>{\footnotesize}c<{\hspace{7pt}}}"  # photometry
            r"c<{\hspace{7pt}}"  # dim
            r"*{3}{l<{\hspace{7pt}}}"  # likelihoods
            r"@{}"
        ),
        "header_start": "\n".join(  # noqa: FLY002
            (
                r"\toprule",
                r"\multicolumn{6}{c}{Gaia} & \multicolumn{2}{c}{PS-1} & \multicolumn{1}{c}{} & \multicolumn{3}{c}{Membership Likelihood (${\rm MLE}_{\phantom{0}5\%}^{95\%}$)}\\",  # noqa: E501
                r"\cmidrule(lr){1-6} \cmidrule(lr){7-8} \cmidrule(lr){10-12}"
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

# -----------------------------------------------------------------------------
# Save the table

# Serialize to a string
out = StringIO()
table.write(out, **write_kwargs)
lines = out.getvalue().split("\n")

# color the lines
end = lines.index("\\bottomrule\\bottomrule")
lines.insert(end - 2, r"\rowcolor{gray!7}")
lines.insert(end - 3, r"\rowcolor{gray!7}")
lines.insert(end - 4, r"\rowcolor{gray!7}")
lines.insert(end - 5, r"\rowcolor{gray!7}")
lines.insert(end - 11, r"\rowcolor{gray!7}")

# Save to disk
with (paths.output / "gd1" / "member_table_select.tex").open("w") as f:
    f.write("\n".join(lines))
