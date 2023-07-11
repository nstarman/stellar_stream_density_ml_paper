"""Train photometry background flow."""

import sys
from collections.abc import Callable
from dataclasses import dataclass

import asdf
import numpy as np
import torch as xp
import zuko
from scipy.interpolate import CubicSpline
from showyourwork.paths import user as user_paths

import stream_mapper.pytorch as sml
from stream_mapper.core import WEIGHT_NAME
from stream_mapper.core.typing import ArrayNamespace
from stream_mapper.pytorch.builtin import Parallax2DistMod
from stream_mapper.pytorch.params import ModelParameter, ModelParameters
from stream_mapper.pytorch.params.bounds import ClippedBounds, SigmoidBounds
from stream_mapper.pytorch.params.scaler import StandardLnWidth, StandardLocation
from stream_mapper.pytorch.typing import Array

paths = user_paths()


# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.helper import isochrone_spline

# =============================================================================
# Load Data

with asdf.open(
    paths.data / "mock" / "data.asdf", lazy_load=False, copy_arrays=True
) as af:
    stream_abs_mags = af["stream_abs_mags"]
    gamma_mass = af["gamma_mass"]

    data = sml.Data(**af["data"]).astype(xp.Tensor, dtype=xp.float32)
    where = sml.Data(**af["where"]).astype(xp.Tensor, dtype=xp.bool)
    scaler = sml.utils.StandardScaler(**af["scaler"]).astype(
        xp.Tensor, dtype=xp.float32
    )
    object.__setattr__(scaler, "names", tuple(scaler.names))
    coord_bounds = {k: tuple(v) for k, v in af["coord_bounds"].items()}


# =============================================================================
# Make Data

coord_astrometric_names = ("phi2", "parallax")
coord_photometric_names = ()
phot_names = ("g", "r")
coord_names = coord_astrometric_names + coord_photometric_names

coord_astrometric_bounds = {k: coord_bounds[k] for k in coord_astrometric_names}
coord_photometric_bounds = {k: coord_bounds[k] for k in coord_photometric_names}
phot_bounds = {k: coord_bounds[k] for k in phot_names}


# =============================================================================
# Background Model

bkg_phi2_model = sml.builtin.Uniform(
    data_scaler=scaler,
    indep_coord_names=("phi1",),
    coord_names=("phi2",),
    coord_bounds={"phi2": coord_bounds["phi2"]},
    params=ModelParameters(),
)


bkg_plx_model = sml.builtin.Exponential(
    net=sml.nn.sequential(
        data=1, hidden_features=32, layers=3, features=1, dropout=0.15
    ),
    data_scaler=scaler,
    indep_coord_names=("phi1",),
    coord_names=("parallax",),
    coord_bounds={"parallax": coord_bounds["parallax"]},
    params=ModelParameters(
        {
            "parallax": {
                "slope": ModelParameter(bounds=SigmoidBounds(15.0, 25.0), scaler=None)
            }
        }
    ),
)


# -----------------------------------------------------------------------------

flow_coords = ("phi1", "g", "r")

flow_scaler = scaler[flow_coords]  # slice the StandardScaler
bkg_flow = sml.builtin.compat.ZukoFlowModel(
    net=zuko.flows.MAF(features=2, context=1, transforms=4, hidden_features=[4] * 4),
    jacobian_logdet=-xp.log(xp.prod(flow_scaler.scale[1:])),
    data_scaler=flow_scaler,
    coord_names=phot_names,
    coord_bounds=phot_bounds,
    params=ModelParameters(),
    with_grad=False,
)

# -----------------------------------------------------------------------------

background_model = sml.IndependentModels(
    {
        "astrometric": sml.IndependentModels(
            {"phi2": bkg_phi2_model, "parallax": bkg_plx_model}
        ),
        "photometric": bkg_flow,
    }
)


# =============================================================================
# Stream Model

stream_astrometric_model = sml.builtin.Normal(
    net=sml.nn.sequential(
        data=1,
        hidden_features=64,
        layers=5,
        features=2 * len(coord_astrometric_names),
        dropout=0.15,
    ),
    data_scaler=scaler,
    coord_names=coord_astrometric_names,
    coord_bounds=coord_astrometric_bounds,
    params=ModelParameters(
        {
            "phi2": {
                "mu": ModelParameter(
                    bounds=SigmoidBounds(*coord_bounds["phi2"]),
                    scaler=StandardLocation.from_data_scaler(scaler, "phi2"),
                ),
                "ln-sigma": ModelParameter(
                    bounds=SigmoidBounds(-2.4, -0.5),
                    scaler=StandardLnWidth.from_data_scaler(scaler, "phi2", xp=xp),
                ),
            },
            "parallax": {
                "mu": ModelParameter(
                    bounds=SigmoidBounds(*coord_bounds["parallax"]),
                    scaler=StandardLocation.from_data_scaler(scaler, "parallax"),
                ),
                "ln-sigma": ModelParameter(
                    bounds=SigmoidBounds(-9.0, -0.5),
                    scaler=StandardLnWidth.from_data_scaler(scaler, "parallax", xp=xp),
                ),
            },
        }
    ),
)


isochrone_spl = isochrone_spline(stream_abs_mags[:, :2].value, xp=np)
mass_of_gamma = CubicSpline(gamma_mass[:, 0], gamma_mass[:, 1])


@dataclass(frozen=True)
class Pal5StreamMassFunction:
    """Stream mass function."""

    mass_of_gamma: Callable[[float | Array], Array]

    def __post_init__(self) -> None:
        """Post init."""
        self._mmin: Array
        self._mmax: Array
        object.__setattr__(self, "_mmin", xp.asarray(self.mass_of_gamma([0])))
        object.__setattr__(self, "_mmax", xp.asarray(self.mass_of_gamma([1])))

    @property
    def _ln_pdf_norm(self) -> Array:
        """Normalization for the PDF."""
        return xp.log(xp.asarray(2)) + xp.log(xp.sqrt(self._mmax) - xp.sqrt(self._mmin))

    def __call__(
        self, gamma: Array, _: sml.Data[Array], *, xp: ArrayNamespace
    ) -> Array:
        """Return the PDF."""
        return -0.5 * xp.log(xp.asarray(self.mass_of_gamma(gamma))) - self._ln_pdf_norm


stream_isochrone_model = sml.builtin.IsochroneMVNorm(
    net=None,
    data_scaler=scaler,
    # coordinates
    coord_names=coord_photometric_names,
    coord_bounds=coord_photometric_bounds,
    # photometry
    phot_names=phot_names,
    phot_err_names=tuple(f"{k}_err" for k in phot_names),
    phot_apply_dm=(True, True),
    phot_bounds=phot_bounds,
    # isochrone
    gamma_edges=xp.linspace(isochrone_spl.x.min(), isochrone_spl.x.max(), 50),
    isochrone_spl=isochrone_spl,
    isochrone_err_spl=None,
    stream_mass_function=Pal5StreamMassFunction(mass_of_gamma=mass_of_gamma),
    # params
    params=ModelParameters(),
)

stream_model = sml.IndependentModels(
    {"astrometric": stream_astrometric_model, "photometric": stream_isochrone_model},
    unpack_params_hooks=(
        Parallax2DistMod(
            astrometric_coord="astrometric.parallax",
            photometric_coord="photometric.distmod",
        ),
    ),
)


# =============================================================================

model = sml.MixtureModel(
    {
        "stream": stream_model,
        "background": background_model,
    },
    net=sml.nn.sequential(
        data=1, hidden_features=64, layers=4, features=1, dropout=0.15
    ),
    data_scaler=scaler,
    params=ModelParameters(
        {
            f"stream.{WEIGHT_NAME}": ModelParameter(
                bounds=SigmoidBounds(-10.0, -0.05), scaler=None
            ),
            f"background.{WEIGHT_NAME}": ModelParameter(
                bounds=ClippedBounds(-3.2, 0.0), scaler=None
            ),
        }
    ),
)
