"""write variable 'isochrone_age_variable.txt' to disk."""

import sys

import asdf
import torch as xp
from showyourwork.paths import user as user_paths

import stream_mapper.pytorch as sml
from stream_mapper.core import WEIGHT_NAME

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.mock.model import model
from scripts.mock.models import helper

(paths.output / "mock").mkdir(parents=True, exist_ok=True)

# =============================================================================
# Load Data

# Load model
model.load_state_dict(xp.load(paths.data / "mock" / "model.pt"))

# Load data
with asdf.open(paths.data / "mock" / "data.asdf") as af:
    data = sml.Data(**af["data"]).astype(xp.Tensor, dtype=xp.float32)
    where = sml.Data(**af["where"]).astype(xp.Tensor, dtype=xp.bool)
    table = af["table"]


# Evaluate model
with xp.no_grad():
    helper.manually_set_dropout(model, 0)
    mpars = model.unpack_params(model(data))

    stream_lik = model.component_posterior("stream", mpars, data, where=where)
    tot_lik = model.posterior(mpars, data, where=where)

stream_prob = stream_lik / tot_lik
stream_prob[(stream_prob > 0.4) & (mpars[(f"stream.{WEIGHT_NAME}",)] < -4)] = 0

true_stream = table["label"] == "stream"
numstream = sum(true_stream)
falseident = sum((stream_prob.numpy() > 0.8) & ~true_stream) / numstream * 100

with (paths.output / "mock" / "falseident_variable.txt").open("w") as f:
    f.write(f"{falseident:2g}")
