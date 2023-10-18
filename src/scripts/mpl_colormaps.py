"""Matplotlib colormaps."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ============================================================================

stream_cmap1 = LinearSegmentedColormap(
    "Stream1",
    {
        "red": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [0.5, 1.0, 1.0],
                [0.75, 1.0, 1.0],
                [1.0, 0.5, 0.5],
            ]
        ),
        "green": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [0.5, 0.75, 0.75],
                [0.75, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        "blue": np.array(
            [
                [0.0, 0.3, 0.3],
                [0.25, 1.0, 1.0],
                [0.5, 1.0, 1.0],
                [0.75, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        "alpha": np.array(
            [
                [0.0, 1.0, 1.0],
                [0.25, 1.0, 1.0],
                [0.5, 1.0, 1.0],
                [0.75, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ),
    },
)
mpl.colormaps.register(cmap=stream_cmap1)

# ============================================================================

stream_cmap2 = LinearSegmentedColormap(
    "Stream2",
    {
        "red": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [0.75, 1, 1],
                [1.0, 0.5, 0.5],
            ]
        ),
        "green": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [0.5, 0.75, 0.75],
                [0.75, 1, 1],
                [1.0, 0.5, 0.5],
            ]
        ),
        "blue": np.array(
            [
                [0.0, 0.3, 0.3],
                [0.25, 1.0, 1.0],
                [0.5, 0.75, 0.75],
                [0.75, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        "alpha": np.array(
            [
                [0.0, 1.0, 1.0],
                [0.25, 1.0, 1.0],
                [0.5, 1.0, 1.0],
                [0.75, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ),
    },
)
mpl.colormaps.register(cmap=stream_cmap2)
