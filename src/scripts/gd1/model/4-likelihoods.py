"""Get Likelihoods."""

import sys
from pathlib import Path

import numpy as np
import torch as xp
from astropy.table import QTable
from tqdm import tqdm

# Add the parent directory to the path
sys.path.append(Path(__file__).parents[3].as_posix())
# isort: split

from scripts import paths
from scripts.gd1.datasets import data, table, where
from scripts.gd1.model.define_model import model
from scripts.helper import manually_set_dropout

# =============================================================================

# Load model
model.load_state_dict(xp.load(paths.data / "gd1" / "model.pt"))
model = model.eval()


# =============================================================================
# Variations

N = 250

stream_probs = xp.empty((len(data), N))
spur_probs = xp.empty((len(data), N))
with xp.no_grad():
    model.train()
    manually_set_dropout(model, 0.15)

    for i in tqdm(range(N), total=N):
        mpars = model.unpack_params(model(data))
        stream_lik = model.component_posterior(
            "stream", mpars, data, stream_astrometric_where=where
        )
        spur_lik = model.component_posterior(
            "spur", mpars, data, spur_astrometric_where=where
        )
        bkg_lik = model.component_posterior(
            "background", mpars, data, background_astrometric_where=where
        )
        tot_lik = model.posterior(
            mpars,
            data,
            stream_astrometric_where=where,
            spur_astrometric_where=where,
            background_astrometric_where=where,
        )

        stream_probs[:, i] = stream_lik / tot_lik
        spur_probs[:, i] = spur_lik / tot_lik

stream_prob_percentiles = np.c_[
    np.percentile(stream_probs, 5, axis=1),
    np.percentile(stream_probs, 95, axis=1),
]
spur_prob_percentiles = np.c_[
    np.percentile(spur_probs, 5, axis=1),
    np.percentile(spur_probs, 95, axis=1),
]

model.eval()
manually_set_dropout(model, 0.0)


# =============================================================================
# MLE

with xp.no_grad():
    mpars = model.unpack_params(model(data))

    stream_lik = model.component_posterior(
        "stream", mpars, data, stream_astrometric_where=where
    )
    spur_lik = model.component_posterior(
        "spur", mpars, data, spur_astrometric_where=where
    )
    bkg_lik = model.component_posterior(
        "background", mpars, data, background_astrometric_where=where
    )
    tot_lik = model.posterior(
        mpars,
        data,
        stream_astrometric_where=where,
        spur_astrometric_where=where,
        background_astrometric_where=where,
    )

stream_prob = stream_lik / tot_lik
spur_prob = spur_lik / tot_lik

# =============================================================================

lik_tbl = QTable()
lik_tbl["source id"] = table["source_id"]
lik_tbl["stream (5%)"] = stream_prob_percentiles[:, 0]
lik_tbl["stream (MLE)"] = stream_prob.numpy()
lik_tbl["stream (95%)"] = stream_prob_percentiles[:, 1]
lik_tbl["spur (5%)"] = spur_prob_percentiles[:, 0]
lik_tbl["spur (MLE)"] = spur_prob.numpy()
lik_tbl["spur (95%)"] = spur_prob_percentiles[:, 1]

lik_tbl.write(paths.data / "gd1" / "membership_likelhoods.ecsv")
