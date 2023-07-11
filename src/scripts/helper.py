"""Exposes common paths useful for manipulating datasets and generating figures."""

from scipy.interpolate import CubicSpline
from torch import nn

from stream_ml.core import ModelAPI
from stream_ml.core.typing import Array, ArrayNamespace
from stream_ml.core.utils.funcs import pairwise_distance
from stream_ml.pytorch.builtin.compat import FlowModel


def isochrone_spline(mags: Array, *, xp: ArrayNamespace[Array]) -> CubicSpline:
    """Return a spline interpolating the isochrone's coordinates."""
    pdist = pairwise_distance(mags, axis=0, xp=xp)
    gamma = xp.concatenate((xp.asarray([0]), pdist.cumsum()))
    gamma = gamma / gamma[-1]  # gamma in [0, 1]
    return CubicSpline(gamma, mags)


def manually_set_dropout(model: ModelAPI, p: float) -> tuple[ModelAPI]:
    """Manually set dropout.

    For use when pytorch ``.eval()`` doesn't work
    """
    for m in model.children():
        if isinstance(m, nn.Dropout):
            m.p = p
        elif isinstance(m, FlowModel):
            pass
        else:
            manually_set_dropout(m, p)
    return model
