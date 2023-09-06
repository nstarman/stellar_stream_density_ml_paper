"""Plot results."""

import sys
from pathlib import Path

import asdf
import numpy as np
import torch as xp
from astropy.table import QTable
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from torch import nn

# isort: split
import stream_ml.pytorch as sml
from stream_ml.pytorch.params import ModelParameter, ModelParameters, set_param
from stream_ml.pytorch.params.bounds import ClippedBounds, SigmoidBounds
from stream_ml.pytorch.params.scaler import StandardLnWidth, StandardLocation

sys.path.append(Path(__file__).parents[2].as_posix())
# isort: split

from scripts import paths
from scripts.helper import isochrone_spline

##############################################################################

with asdf.open(paths.data / "gd1" / "info.asdf", mode="r") as af:
    renamer = af["renamer"]
    scaler = sml.utils.StandardScaler(**af["scaler"]).astype(
        xp.Tensor, dtype=xp.float32
    )
    all_coord_bounds = {k: tuple(v) for k, v in af["coord_bounds"].items()}

astro_coords = ("phi2", "pmphi1", "pmphi2")
astro_coord_errs = ("phi2_err", "pmphi1_err", "pmphi2_err")
astro_coord_bounds = {k: v for k, v in all_coord_bounds.items() if k in astro_coords}

phot_coords = ("g", "r")
phot_coord_errs = ("g_err", "r_err")
phot_coord_bounds = {k: v for k, v in all_coord_bounds.items() if k in phot_coords}

coord_names = astro_coords + phot_coords
coord_bounds: dict[str, tuple[float, float]] = {
    k: v for k, v in all_coord_bounds.items() if k in coord_names
}


##############################################################################
# Background

# -----------------------------------------------------------------------------
# Astrometry

background_astrometric_model = sml.builtin.Exponential(
    net=sml.nn.lin_tanh(
        n_in=1, n_hidden=64, n_layers=4, n_out=len(coord_names), dropout=0.15
    ),
    data_scaler=scaler,
    coord_names=astro_coords,
    coord_err_names=astro_coord_errs,
    coord_bounds=astro_coord_bounds,
    params=ModelParameters(
        {
            "phi2": {
                "slope": ModelParameter(bounds=SigmoidBounds(-0.01, 0.01), scaler=None)
            },
            "pmphi1": {
                "slope": ModelParameter(bounds=SigmoidBounds(-0.5, 0.0), scaler=None)
            },
            "pmphi2": {
                "slope": ModelParameter(bounds=SigmoidBounds(-0.5, 0.0), scaler=None)
            },
        }
    ),
)

# -----------------------------------------------------------------------------
# Photometry


def _make_background_flow(num_layers: int = 4, num_features: int = 2) -> Flow:
    """Make the background photometry flow."""
    base_dist = ConditionalDiagonalNormal(
        shape=[num_features], context_encoder=nn.Linear(1, 4)
    )
    transforms = []
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=num_features))
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=num_features, hidden_features=3, context_features=1
            )
        )
    transform = CompositeTransform(transforms)
    return Flow(transform, base_dist)


flow_scaler = scaler[("phi1", *phot_coords)]

background_photometric_model = sml.builtin.compat.FlowModel(
    net=_make_background_flow(),
    jacobian_logdet=-xp.log(xp.prod(flow_scaler.scale[1:])),
    data_scaler=flow_scaler,
    coord_names=phot_coords,
    coord_bounds=phot_coord_bounds,
    params=ModelParameters(),
    with_grad=False,
)

# -----------------------------------------------------------------------------
# All

background_model = sml.IndependentModels(
    {
        "astrometric": background_astrometric_model,
        "photometric": background_photometric_model,
    }
)


##############################################################################
# STREAM

# -----------------------------------------------------------------------------
# Astrometry

gd1_cp = QTable.read(paths.data / "gd1" / "stream_control_points.ecsv")

# Selection of control points
stream_strometric_prior = sml.prior.ControlRegions(
    center=sml.Data.from_format(
        gd1_cp, fmt="astropy.table", names=("phi1", "phi2", "pm_phi1"), renamer=renamer
    ).astype(xp.Tensor, dtype=xp.float32),
    width=sml.Data.from_format(
        gd1_cp,
        fmt="astropy.table",
        names=("w_phi2", "w_pm_phi1"),
        renamer={"w_phi2": "phi2", "w_pm_phi1": "pmphi1"},
    ).astype(xp.Tensor, dtype=xp.float32),
    lamda=1_000,
)

stream_astrometric_model = sml.builtin.TruncatedNormal(
    net=sml.nn.lin_tanh(
        n_in=1,
        n_hidden=128,
        n_layers=5,
        n_out=2 * len(astro_coords),
        dropout=0.15,
    ),
    data_scaler=scaler,
    coord_names=astro_coords,
    coord_err_names=astro_coord_errs,
    coord_bounds=astro_coord_bounds,
    params=ModelParameters(
        {
            "phi2": {
                "mu": ModelParameter(
                    bounds=SigmoidBounds(*coord_bounds["phi2"]),
                    scaler=StandardLocation.from_data_scaler(scaler, "phi2", xp=xp),
                ),
                "ln-sigma": ModelParameter(
                    bounds=SigmoidBounds(-3.0, 0.0),
                    scaler=StandardLnWidth.from_data_scaler(scaler, "phi2", xp=xp),
                ),
            },
            "pmphi1": {
                "mu": ModelParameter(
                    bounds=SigmoidBounds(*coord_bounds["pmphi1"]),
                    scaler=StandardLocation.from_data_scaler(scaler, "pmphi1", xp=xp),
                ),
                "ln-sigma": ModelParameter(
                    bounds=SigmoidBounds(-3.0, -0.5),
                    scaler=StandardLnWidth.from_data_scaler(scaler, "pmphi1", xp=xp),
                ),
            },
            "pmphi2": {
                "mu": ModelParameter(
                    bounds=SigmoidBounds(*coord_bounds["pmphi2"]),
                    scaler=StandardLocation.from_data_scaler(scaler, "pmphi2", xp=xp),
                ),
                "ln-sigma": ModelParameter(
                    bounds=SigmoidBounds(-3.0, 0),
                    scaler=StandardLnWidth.from_data_scaler(scaler, "pmphi2", xp=xp),
                ),
            },
        }
    ),
    priors=(stream_strometric_prior,),
)


# -----------------------------------------------------------------------------
# Photometry

with asdf.open(paths.data / "gd1" / "isochrone.asdf", mode="r") as af:
    abs_mags = sml.Data(**af["isochrone_data"]).astype(xp.Tensor, dtype=xp.float32)

stream_isochrone_spl = isochrone_spline(abs_mags["g", "r"].array, xp=np)

# concentrating on the MS turnoff
gamma_edges = xp.concatenate(
    [
        xp.linspace(0, 0.43, 30),
        xp.linspace(0.43, 0.5, 15),
        xp.linspace(0.501, 1, 30),
    ]
)

stream_mass_function = sml.builtin.StepwiseMassFunction(
    boundaries=(0, 0.55, 1.01),
    log_probs=(0.0, 0.0),  # TODO: set a value
)

# Control points
mag_cp = QTable.read(paths.data / "gd1" / "magnitude_control_points.ecsv")
stream_photometric_prior = sml.prior.ControlRegions(
    center=sml.Data.from_format(
        mag_cp, fmt="astropy.table", names=("phi1", "distmod"), renamer=renamer
    ).astype(xp.Tensor, dtype=xp.float32),
    width=sml.Data.from_format(
        mag_cp,
        fmt="astropy.table",
        names=("w_distmod",),
        renamer={"w_distmod": "distmod"},
    ).astype(xp.Tensor, dtype=xp.float32),
    lamda=1_000,
)

stream_isochrone_model = sml.builtin.IsochroneMVNorm(
    net=sml.nn.lin_tanh(n_in=1, n_hidden=32, n_layers=4, n_out=2, dropout=0.15),
    data_scaler=flow_scaler,
    # coordinates
    coord_names=("distmod",),
    coord_bounds={"distmod": (13.0, 18.0)},
    # photometry
    phot_names=phot_coords,
    phot_apply_dm=(True, True),  # (g, r)
    phot_err_names=phot_coord_errs,
    phot_bounds=phot_coord_bounds,
    # isochrone
    gamma_edges=gamma_edges,
    isochrone_spl=stream_isochrone_spl,
    isochrone_err_spl=None,
    stream_mass_function=stream_mass_function,
    # params
    params=ModelParameters(
        {
            "distmod": {
                "mu": ModelParameter(bounds=SigmoidBounds(13.0, 18.0), scaler=None),
                "ln-sigma": ModelParameter(
                    bounds=SigmoidBounds(-7.6, -2.8), scaler=None
                ),
            },
        }
    ),
    priors=(stream_photometric_prior,),
)

# -----------------------------------------------------------------------------

stream_model = sml.IndependentModels(
    {"astrometric": stream_astrometric_model, "photometric": stream_isochrone_model}
)


##############################################################################
# SPUR

# =============================================================================
# Astrometry

spur_cp_tbl = QTable.read(paths.data / "gd1" / "spur_control_points.ecsv")
spur_control_points_prior = sml.prior.ControlRegions(
    center=sml.Data.from_format(
        spur_cp_tbl,
        fmt="astropy.table",
        names=("phi1", "phi2", "pm_phi1"),
        renamer=renamer,
    ).astype(xp.Tensor, dtype=xp.float32),
    width=sml.Data.from_format(
        spur_cp_tbl,
        fmt="astropy.table",
        names=("w_phi2", "w_pm_phi1"),
        renamer={"w_phi2": "phi2", "w_pm_phi1": "pmphi1"},
    ).astype(xp.Tensor, dtype=xp.float32),
    lamda=10_000,
)


spur_astrometric_model = sml.builtin.Normal(
    net=sml.nn.lin_tanh(
        n_in=1,
        n_hidden=64,
        n_layers=4,
        n_out=2 * len(astro_coords),
        dropout=0.15,
    ),
    data_scaler=scaler,
    coord_names=astro_coords,
    coord_err_names=astro_coord_errs,
    coord_bounds=astro_coord_bounds,
    params=ModelParameters(
        {
            "phi2": {
                "mu": ModelParameter(
                    bounds=SigmoidBounds(*coord_bounds["phi2"]),
                    scaler=StandardLocation.from_data_scaler(scaler, "phi2", xp=xp),
                ),
                "ln-sigma": ModelParameter(
                    bounds=SigmoidBounds(-3, 0),
                    scaler=StandardLnWidth.from_data_scaler(scaler, "phi2", xp=xp),
                ),
            },
            "pmphi1": {
                "mu": ModelParameter(
                    bounds=SigmoidBounds(*coord_bounds["pmphi1"]),
                    scaler=StandardLocation.from_data_scaler(scaler, "pmphi1", xp=xp),
                ),
                "ln-sigma": ModelParameter(
                    bounds=SigmoidBounds(-3, 0),
                    scaler=StandardLnWidth.from_data_scaler(scaler, "pmphi1", xp=xp),
                ),
            },
            "pmphi2": {
                "mu": ModelParameter(
                    bounds=SigmoidBounds(*coord_bounds["pmphi2"]),
                    scaler=StandardLocation.from_data_scaler(scaler, "pmphi2", xp=xp),
                ),
                "ln-sigma": ModelParameter(
                    bounds=SigmoidBounds(-3, 0),
                    scaler=StandardLnWidth.from_data_scaler(scaler, "pmphi2", xp=xp),
                ),
            },
        }
    ),
    priors=(spur_control_points_prior,),
)


spur_isochrone_model = sml.builtin.IsochroneMVNorm(
    net=None,
    data_scaler=flow_scaler,
    # coordinates
    coord_names=(),
    coord_bounds={},
    # photometry
    phot_names=phot_coords,
    phot_apply_dm=(True, True),  # g, r
    phot_err_names=phot_coord_errs,
    phot_bounds=phot_coord_bounds,
    # isochrone
    gamma_edges=gamma_edges,
    isochrone_spl=stream_isochrone_spl,
    isochrone_err_spl=None,
    stream_mass_function=stream_mass_function,
    # params
    params=ModelParameters(),
)


spur_model = sml.IndependentModels(
    {"astrometric": spur_astrometric_model, "photometric": spur_isochrone_model}
)


# =============================================================================
# Mixture


def spur_shares_stream_distmod(params: dict) -> dict:
    """Set the spur distance modulus to be the same as the stream."""
    # Set the distance modulus
    set_param(
        params,
        ("spur.photometric.distmod", "mu"),
        params["stream.photometric.distmod"]["mu"],
    )
    set_param(
        params,
        ("spur.photometric.distmod", "ln-sigma"),
        params["stream.photometric.distmod"]["ln-sigma"],
    )
    return params


model = sml.MixtureModel(
    {"stream": stream_model, "spur": spur_model, "background": background_model},
    net=sml.nn.lin_tanh(n_in=1, n_hidden=32, n_layers=4, n_out=2, dropout=0.15),
    data_scaler=scaler,
    params=ModelParameters(
        {
            "stream.weight": ModelParameter(
                bounds=SigmoidBounds(1e-3, 0.3), scaler=None
            ),
            "spur.weight": ModelParameter(bounds=SigmoidBounds(1e-3, 0.1), scaler=None),
            "background.weight": ModelParameter(
                bounds=ClippedBounds(0.7, 1.0), scaler=None
            ),
        }
    ),
    unpack_params_hooks=(spur_shares_stream_distmod,),  # stream => spur parallax
    priors=(
        sml.prior.HardThreshold(
            1,
            set_to=1e-3,
            upper=-90,
            param_name="stream.weight",
            coord_name="phi1",
            data_scaler=scaler,
        ),
        sml.prior.HardThreshold(
            1,
            set_to=1e-3,
            lower=10,
            param_name="stream.weight",
            coord_name="phi1",
            data_scaler=scaler,
        ),
        sml.prior.HardThreshold(
            1,
            set_to=1e-3,
            upper=-45,
            param_name="spur.weight",
            coord_name="phi1",
            data_scaler=scaler,
        ),
        sml.prior.HardThreshold(
            1,
            set_to=1e-3,
            lower=-15,
            param_name="spur.weight",
            coord_name="phi1",
            data_scaler=scaler,
        ),
    ),
)


# -----------------------------------------------------------------------------

pth = paths.data / "gd1" / "model.tmp"
if not pth.exists():
    with pth.open("w") as f:
        f.write("hack for snakemake")
