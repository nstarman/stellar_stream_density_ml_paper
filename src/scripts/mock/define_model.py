"""Train photometry background flow."""


import asdf
import numpy as np
import torch as xp
import zuko
from scipy.interpolate import CubicSpline
from showyourwork.paths import user as user_paths

import stream_ml.pytorch as sml
from stream_ml.core import WEIGHT_NAME
from stream_ml.core.utils import pairwise_distance
from stream_ml.pytorch.builtin import Parallax2DistMod
from stream_ml.pytorch.params import ModelParameter, ModelParameters
from stream_ml.pytorch.params.bounds import ClippedBounds, SigmoidBounds
from stream_ml.pytorch.params.scaler import StandardLnWidth, StandardLocation

paths = user_paths()


# =============================================================================
# Load Data

with asdf.open(
    paths.data / "mock" / "data.asdf", lazy_load=False, copy_arrays=True
) as af:
    stream_abs_mags = af["stream_abs_mags"]

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
    net=zuko.flows.MAF(features=2, context=1, transforms=4, hidden_features=[3] * 4),
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


def isochrone_spline(mags: np.ndarray) -> CubicSpline:
    """Make a spline of the isochrone."""
    gamma = np.concatenate(
        (np.array([0]), pairwise_distance(mags, axis=0, xp=np).cumsum())
    )
    gamma = gamma / gamma[-1]
    return CubicSpline(gamma, mags)


isochrone_spl = isochrone_spline(stream_abs_mags[:, :2].value)

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
    stream_mass_function=sml.builtin.StepwiseMassFunction(
        boundaries=(0, 0.8, 1), log_probs=(0, -3)
    ),
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
