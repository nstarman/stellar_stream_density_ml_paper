"""Exposes common paths useful for manipulating datasets and generating figures."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import torch as xp
from scipy.interpolate import CubicSpline
from torch import nn

from stream_ml.core.utils.funcs import pairwise_distance
from stream_ml.pytorch.builtin.compat._flow import _FlowModel

if TYPE_CHECKING:
    from types import FunctionType

    from matplotlib.colors import LinearSegmentedColormap

    from stream_ml.core import ModelAPI
    from stream_ml.core.params import Params
    from stream_ml.core.typing import Array, ArrayNamespace


def p2alpha(p: Array, /, minval: float = 0.1) -> Array:
    """Convert probability to alpha."""
    out = minval + (1 - minval) * np.where(  # avoid NaN for p=0
        p == p.max(), 1, (p - p.min()) / (p.max() - p.min())
    )
    return np.clip(out, minval, 1)


def color_by_probable_member(
    *pandcmaps: tuple[Array, LinearSegmentedColormap]
) -> np.ndarray:
    """Color by the most probable member."""
    # probabilities
    ps = np.stack(tuple(p[0] for p in pandcmaps), 1)
    # colors
    cs = np.stack(tuple(cmap(p) for p, cmap in pandcmaps), 0)
    # color by most probable
    return cs[np.argmax(ps, 1), np.arange(len(ps))]


def isochrone_spline(mags: Array, *, xp: ArrayNamespace[Array]) -> CubicSpline:
    """Return a spline interpolating the isochrone's coordinates."""
    pdist = pairwise_distance(mags, axis=0, xp=xp)
    gamma = xp.concatenate((xp.asarray([0]), pdist.cumsum(0)))
    gamma = gamma / gamma[-1]  # gamma in [0, 1]
    return CubicSpline(gamma, mags)


def manually_set_dropout(model: ModelAPI, p: float) -> tuple[ModelAPI]:
    """Manually set dropout.

    For use when pytorch ``.eval()`` doesn't work
    """
    for m in model.children():
        if isinstance(m, nn.Dropout):
            m.p = p
        elif isinstance(m, _FlowModel):
            pass
        else:
            manually_set_dropout(m, p)
    return model


def a_as_b(cols: dict[str, str | None], /, prefix: str) -> str:
    """Convert a dictionary of column names to a string of "a as b" pairs."""
    return ", ".join(
        tuple(prefix + (k if v is None else f"{k} as {v}") for k, v in cols.items())
    )


def recursive_iterate(
    dmpars: list[Params[str, Any]],
    structure: dict[str, Any],
    _prefix: str = "",
    *,
    reduction: FunctionType = partial(xp.mean, axis=1),
) -> dict[str, Any]:
    """Recursively iterate and compute the mean of each parameter."""
    out = dict[str, Any]()
    _prefix = _prefix.lstrip(".")
    for k, v in structure.items():
        if isinstance(v, Mapping):
            out[k] = recursive_iterate(
                dmpars, v, _prefix=f"{_prefix}.{k}", reduction=reduction
            )
            continue

        key: tuple[str] | tuple[str, str] = (f"{_prefix}", k) if _prefix else (k,)
        out[k] = reduction(xp.stack([mp[key] for mp in dmpars], 1))

    return out
