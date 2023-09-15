"""Get Likelihoods."""

import sys

import numpy as np
import torch as xp
from astropy.table import QTable
from showyourwork.paths import user as user_paths
from tqdm import tqdm

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.helper import manually_set_dropout
from scripts.pal5.datasets import data, table, where
from scripts.pal5.define_model import model

# =============================================================================

# Load model
model.load_state_dict(xp.load(paths.data / "pal5" / "model.pt"))
model = model.eval()


# =============================================================================
# Variations

N = 250

stream_probs = xp.empty((len(data), N))
bkg_probs = xp.empty((len(data), N))
with xp.no_grad():
    model.train()
    manually_set_dropout(model, 0.15)

    for i in tqdm(range(N), total=N):
        mpars = model.unpack_params(model(data))
        stream_lik = model.component_posterior("stream", mpars, data, where=where)
        bkg_lik = model.component_posterior("background", mpars, data, where=where)
        tot_lik = model.posterior(mpars, data, where=where)

        stream_probs[:, i] = stream_lik / tot_lik
        bkg_probs[:, i] = bkg_lik / tot_lik

stream_prob_percentiles = np.c_[
    np.percentile(stream_probs, 5, axis=1),
    np.percentile(stream_probs, 95, axis=1),
]
bkg_prob_percentiles = np.c_[
    np.percentile(bkg_probs, 5, axis=1),
    np.percentile(bkg_probs, 95, axis=1),
]

model.eval()
manually_set_dropout(model, 0.0)


# =============================================================================
# MLE

with xp.no_grad():
    mpars = model.unpack_params(model(data))

    stream_lik = model.component_ln_posterior("stream", mpars, data, where=where)
    bkg_lik = model.component_ln_posterior("background", mpars, data, where=where)
    tot_lik = model.ln_posterior(mpars, data, where=where)

stream_prob = xp.exp(stream_lik - tot_lik)
bkg_prob = xp.exp(bkg_lik - tot_lik)

# =============================================================================

lik_tbl = QTable()
lik_tbl["source id"] = table["source_id"]
lik_tbl["stream (5%)"] = stream_prob_percentiles[:, 0]
lik_tbl["stream (MLE)"] = stream_prob.numpy()
lik_tbl["stream (95%)"] = stream_prob_percentiles[:, 1]
lik_tbl["bkg (5%)"] = bkg_prob_percentiles[:, 0]
lik_tbl["bkg (MLE)"] = bkg_prob.numpy()
lik_tbl["bkg (95%)"] = bkg_prob_percentiles[:, 1]

lik_tbl.write(paths.data / "pal5" / "membership_likelhoods.ecsv", overwrite=True)
