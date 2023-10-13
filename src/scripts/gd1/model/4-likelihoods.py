"""Compute and save a table of the per-star likelihoods."""

import copy as pycopy
import sys

import numpy as np
import torch as xp
from astropy.table import QTable
from showyourwork.paths import user as user_paths
from tqdm import tqdm

from stream_ml.pytorch.params import Params

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.gd1.datasets import data, table, where
from scripts.gd1.define_model import model
from scripts.helper import (
    manually_set_dropout,
    recursive_iterate,
)

# =============================================================================
# Load model

model = pycopy.deepcopy(model)
model.load_state_dict(xp.load(paths.data / "gd1" / "model" / "model_12499.pt"))
model = model.eval()


# =============================================================================
# Variations

N = 250

stream_weights = np.empty((len(data), N))
stream_probs = np.empty((len(data), N))
spur_weights = np.empty((len(data), N))
spur_probs = np.empty((len(data), N))
allstream_probs = np.empty((len(data), N))
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
        # Weights
        stream_weights[:, i] = mpars["stream.ln-weight",]

        spur_weights[:, i] = mpars["spur.ln-weight",]

        bkg_lnlik = model.component_ln_posterior("background", mpars, data, where=where)
        stream_lnlik = model.component_ln_posterior("stream", mpars, data, where=where)
        spur_lnlik = model.component_ln_posterior("spur", mpars, data, where=where)
        # tot_lnlik = model.ln_posterior(mpars, data, where=where)  # TODO
        tot_lnlik = xp.logsumexp(xp.stack((stream_lnlik, spur_lnlik, bkg_lnlik), 1), 1)

        # Store, applying the postprocessing
        bkg_probs[:, i] = xp.exp(bkg_lnlik - tot_lnlik)
        stream_probs[:, i] = xp.exp(stream_lnlik - tot_lnlik)
        spur_probs[:, i] = xp.exp(spur_lnlik - tot_lnlik)

        allstream_probs[:, i] = xp.exp(
            xp.logaddexp(stream_lnlik, spur_lnlik) - tot_lnlik
        )

    # turn dropout back off
    manually_set_dropout(model, 0)
    model = model.eval()


bkg_prob_percentiles = np.c_[
    np.percentile(bkg_probs, 5, axis=1),
    np.percentile(bkg_probs, 50, axis=1),
    np.percentile(bkg_probs, 95, axis=1),
]
stream_prob_percentiles = np.c_[
    np.percentile(stream_probs, 5, axis=1),
    np.percentile(stream_probs, 50, axis=1),
    np.percentile(stream_probs, 95, axis=1),
]
spur_prob_percentiles = np.c_[
    np.percentile(spur_probs, 5, axis=1),
    np.percentile(spur_probs, 50, axis=1),
    np.percentile(spur_probs, 95, axis=1),
]
allstream_prob_percentiles = np.c_[
    np.percentile(allstream_probs, 5, axis=1),
    np.percentile(allstream_probs, 50, axis=1),
    np.percentile(allstream_probs, 95, axis=1),
]


# =============================================================================
# MLE

with xp.no_grad():
    # turn dropout on
    model = model.eval()
    manually_set_dropout(model, 0)

    # Evaluate the model
    mpars = model.unpack_params(model(data))

    # Likelihoods
    bkg_lnlik = model.component_ln_posterior("background", mpars, data, where=where)
    stream_lnlik = model.component_ln_posterior("stream", mpars, data, where=where)
    spur_lnlik = model.component_ln_posterior("spur", mpars, data, where=where)
    # tot_lnlik = model.ln_posterior(mpars, data, where=where)  # TODO
    tot_lnlik = xp.logsumexp(xp.stack((stream_lnlik, spur_lnlik, bkg_lnlik), 1), 1)

    # Probabilities
    bkg_prob = xp.exp(bkg_lnlik - tot_lnlik)
    stream_prob = xp.exp(stream_lnlik - tot_lnlik)
    spur_prob = xp.exp(spur_lnlik - tot_lnlik)
    allstream_prob = xp.exp(xp.logaddexp(stream_lnlik, spur_lnlik) - tot_lnlik)


# =============================================================================

lik_tbl = QTable()
lik_tbl["source id"] = table["source_id"]

lik_tbl["bkg (MLE)"] = bkg_prob.numpy()
lik_tbl["bkg (5%)"] = bkg_prob_percentiles[:, 0]
lik_tbl["bkg (50%)"] = bkg_prob_percentiles[:, 1]
lik_tbl["bkg (95%)"] = bkg_prob_percentiles[:, 1]

lik_tbl["stream.ln-weight"] = stream_weights
lik_tbl["stream (MLE)"] = stream_prob.numpy()
lik_tbl["stream (5%)"] = stream_prob_percentiles[:, 0]
lik_tbl["stream (50%)"] = stream_prob_percentiles[:, 1]
lik_tbl["stream (95%)"] = stream_prob_percentiles[:, 1]

lik_tbl["spur.ln-weight"] = spur_weights
lik_tbl["spur (MLE)"] = spur_prob.numpy()
lik_tbl["spur (5%)"] = spur_prob_percentiles[:, 0]
lik_tbl["spur (50%)"] = spur_prob_percentiles[:, 1]
lik_tbl["spur (95%)"] = spur_prob_percentiles[:, 1]

lik_tbl["allstream (MLE)"] = allstream_prob.numpy()
lik_tbl["allstream (5%)"] = allstream_prob_percentiles[:, 0]
lik_tbl["allstream (50%)"] = allstream_prob_percentiles[:, 1]
lik_tbl["allstream (95%)"] = allstream_prob_percentiles[:, 1]

lik_tbl.write(paths.data / "gd1" / "membership_likelhoods.ecsv", overwrite=True)
