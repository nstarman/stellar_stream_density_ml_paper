"""Get Likelihoods."""

import copy as pycopy
import sys

import numpy as np
import torch as xp
from astropy.table import QTable
from showyourwork.paths import user as user_paths
from tqdm import tqdm

import stream_ml.pytorch as sml
from stream_ml.core import Params

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from numpy.typing import NDArray

from scripts.helper import (
    detect_significant_changes_in_width,
    manually_set_dropout,
    recursive_iterate,
)
from scripts.pal5.datasets import data, table, where
from scripts.pal5.model import model

# =============================================================================
# Load model

model = pycopy.deepcopy(model)
model.load_state_dict(xp.load(paths.data / "pal5" / "model.pt"))
model = model.eval()


# =============================================================================


def postprocess(data: sml.Data, mpars: Params) -> NDArray[np.bool_]:
    """Postprocess the model parameters."""
    clean = mpars["stream.ln-weight",] > -10  # everything has weight > 0

    indices = detect_significant_changes_in_width(
        data["phi1"],
        mpars,
        coords=[
            ("stream.astrometric.phi2", "ln-sigma"),
            ("stream.astrometric.pmphi1", "ln-sigma"),
            ("stream.astrometric.pmphi2", "ln-sigma"),
        ],
        threshold=3_000,
    )
    clean &= data["phi1"] > (data["phi1"][indices[0]] + 0.5)  # note the munge

    return clean


# =============================================================================
# Variations

N = 250

stream_weights = np.empty((len(data), N))
stream_probs = np.empty((len(data), N))
bkg_probs = np.empty((len(data), N))
with xp.no_grad():
    # turn dropout on
    model = model.train()
    manually_set_dropout(model, 0.15)
    # evaluate the model
    ldmpars = [model.unpack_params(model(data)) for _ in tqdm(range(N))]
    # mpars
    dmpars = Params(recursive_iterate(ldmpars, ldmpars[0], reduction=lambda x: x))

    for i, mpars in enumerate(ldmpars):
        # Postprocess
        # clean = postprocess(data, mpars)
        clean = np.ones(len(data), dtype=bool)

        # Weights
        stream_weights[:, i] = mpars["stream.ln-weight",]
        stream_weights[~clean, i] = -100

        stream_lnlik = model.component_ln_posterior("stream", mpars, data, where=where)
        bkg_lnlik = model.component_ln_posterior("background", mpars, data, where=where)
        # tot_lnlik = model.ln_posterior(mpars, data, where=where)  # TODO
        tot_lnlik = xp.logaddexp(stream_lnlik, bkg_lnlik)

        # Store, applying the postprocessing
        bkg_probs[:, i] = xp.exp(bkg_lnlik - tot_lnlik)
        bkg_probs[~clean, i] = 1

        stream_probs[:, i] = xp.exp(stream_lnlik - tot_lnlik)
        stream_probs[~clean, i] = 0

    # turn dropout back off
    manually_set_dropout(model, 0)
    model = model.eval()

# =============================================================================
# MLE

with xp.no_grad():
    # Evaluate the model
    mpars = model.unpack_params(model(data))

    # Likelihoods
    stream_lnlik = model.component_ln_posterior("stream", mpars, data, where=where)
    bkg_lnlik = model.component_ln_posterior("background", mpars, data, where=where)
    # tot_lnlik = model.ln_posterior(mpars, data, where=where)  # TODO
    tot_lnlik = xp.logaddexp(stream_lnlik, bkg_lnlik)

    # clean = postprocess(data, mpars)
    clean = np.ones(len(data), dtype=bool)

    # Probabilities
    bkg_prob = xp.exp(bkg_lnlik - tot_lnlik)
    bkg_prob[clean] = 1

    stream_prob = xp.exp(stream_lnlik - tot_lnlik)
    stream_prob[~clean] = 0


# =============================================================================

lik_tbl = QTable()
lik_tbl["source id"] = table["source_id"]
lik_tbl["bkg (MLE)"] = bkg_prob.numpy()
lik_tbl["bkg (5%)"] = np.percentile(bkg_probs, 5, axis=1)
lik_tbl["bkg (50%)"] = np.percentile(bkg_probs, 50, axis=1)
lik_tbl["bkg (95%)"] = np.percentile(bkg_probs, 95, axis=1)
lik_tbl["stream.ln-weight"] = stream_weights
lik_tbl["stream (MLE)"] = stream_prob.numpy()
lik_tbl["stream (5%)"] = np.percentile(stream_probs, 5, axis=1)
lik_tbl["stream (50%)"] = np.percentile(stream_probs, 50, axis=1)
lik_tbl["stream (95%)"] = np.percentile(stream_probs, 95, axis=1)

lik_tbl.write(paths.data / "pal5" / "membership_likelhoods.ecsv", overwrite=True)
