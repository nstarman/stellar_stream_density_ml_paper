"""Plot GD1 Likelihoods."""

import sys
from pathlib import Path

import numpy as np
from astropy.table import QTable
from numpy.lib.recfunctions import structured_to_unstructured

# Add the parent directory to the path
sys.path.append(Path(__file__).parents[3].as_posix())
# isort: split

from scripts import paths
from scripts.gd1.datasets import table as data_table

(paths.output / "gd1").mkdir(exist_ok=True, parents=True)
rng = np.random.default_rng(42)

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
        return ""
    if dm == 0 and dp == 0:
        return f"${v:0.2f}$"
    return "".join((f"${v:0.2f}", "^{", f"{dp:+0.2f}", "}_{", f"{dm:+0.2f}", "}$"))


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

# -----------------------------------------------------------------------------
# Write the table

caption = r"""Subset of Membership Table.
\\
This table includes a selection of stars with high membership likelihoods for
the GD-1 stream.
We include:
%
1 star with the maximum likelihood for the stream,
%
5 stars  $(\mathcal{L}^{(S)}_{\rm MLE}) > 0.9$,
%
4 stars with $\mathcal{L}^{(S)}_{\rm MLE}) < 0.75, \mathcal{L}^{(S)}_{\rm 95\%}) > 0.8$,
%
1 star with the maximum likelihood for the spur,
%
1 star with $\mathcal{L}^{(spur)}_{\rm MLE}) > 0.9, \mathcal{L}^{(S)}_{\rm MLE}) < 0.75$,
%
and 3 stars with significant likelihoods for both the stream and spur -- $\mathcal{L}^{(S)}_{\rm MLE}) > 0.1, \mathcal{L}^{(spur)}_{\rm MLE}) > 0.7$
\\
We also include a quality flag ${\rm dim}(\boldsymbol{x})$, indicating the number of
features used by the model.
The full table is available online.
"""  # noqa: E501

write_kwargs = {
    "format": "ascii.latex",
    "overwrite": True,
    "caption": caption,
    "latexdict": {
        "tabletype": "table*",
        "preamble": r"\centering",
        "col_align": r"@{}rcccccccll@{}",
        "header_start": "\n".join(  # noqa: FLY002
            (
                r"\toprule",
                r"& \multicolumn{4}{c}{Gaia} & \multicolumn{2}{c}{PS-1} & \multicolumn{1}{c}{} & \multicolumn{2}{c}{Likelihood}\\",  # noqa: E501
                r"\cmidrule(lr){2-5} \cmidrule(lr){6-7} \cmidrule(lr){9-10}"
            )
        ),
        "header_end": r"\midrule",
        "data_end": r"\bottomrule",
    },
    "formats": {
        r"$\alpha$ [$\mathrm{{}^{\circ}}$]": "%0.2f",
        r"$\delta$ [$\mathrm{{}^{\circ}}$]": "%0.2f",
    },
}

# Save some of the rows for the paper
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
prob_idx = np.where((strm_mle < 0.75) & (spur_mle > 0.9) & (spur_mle != spur_mle.max()))[0]  # noqa: E501
subselect = rng.choice(np.arange(len(prob_idx)), size=1, replace=False, shuffle=False)
rows.extend(prob_idx[subselect])
# 3 rows with non-zero joint probability
prob_idx = np.where((strm_mle > 0.1) & (spur_mle > 0.7))[0]
subselect = rng.choice(np.arange(len(prob_idx)), size=3, replace=False, shuffle=False)
rows.extend(prob_idx[subselect])

paper_table = table[rows]
paper_table.write(paths.output / "gd1" / "select_members.tex", **write_kwargs)

# Save the full table for online publication
table.write(paths.output / "gd1" / "gd1_members.tex", **write_kwargs)
