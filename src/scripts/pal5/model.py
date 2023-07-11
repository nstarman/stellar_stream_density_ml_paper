"""Plot results."""

import sys
from dataclasses import replace

import asdf
import torch as xp
import zuko
from astropy.table import QTable
from showyourwork.paths import user as user_paths

import stream_mapper.pytorch as sml
from stream_mapper.core import WEIGHT_NAME
from stream_mapper.pytorch.params import ModelParameter, ModelParameters
from stream_mapper.pytorch.params.bounds import SigmoidBounds
from stream_mapper.pytorch.params.scaler import StandardLnWidth, StandardLocation

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split


##############################################################################

pal5_cp = QTable.read(paths.data / "pal5" / "control_points_stream.ecsv")

with asdf.open(paths.data / "pal5" / "info.asdf", mode="r") as af:
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

background_phi2_model = sml.builtin.Exponential(
    net=sml.nn.sequential(
        data=1, hidden_features=16, layers=2, features=len(coord_names), dropout=0.0
    ),
    data_scaler=scaler,
    coord_names=("phi2",),
    coord_err_names=("phi2_err",),
    coord_bounds={"phi2": astro_coord_bounds["phi2"]},
    params=ModelParameters(
        {
            "phi2": {
                "slope": ModelParameter(bounds=SigmoidBounds(-0.1, -0.01), scaler=None)
            },
        }
    ),
    name="background_phi2_model",
)

pm_coords = ("pmphi1", "pmphi2")
pm_flow_scaler = scaler[("phi1", *pm_coords)]

background_pm_model = sml.builtin.compat.ZukoFlowModel(
    net=zuko.flows.MAF(2, 1, hidden_features=[8] * 3),
    jacobian_logdet=float(-xp.log(xp.prod(pm_flow_scaler.scale[1:]))),
    data_scaler=pm_flow_scaler,
    indep_coord_names=("phi1",),
    coord_names=pm_coords,
    coord_bounds={k: coord_bounds[k] for k in pm_coords},
    params=ModelParameters[xp.Tensor](),
    with_grad=False,
    name="background_pm_model",
)


background_astrometric_model = sml.IndependentModels(
    {
        "phi2": background_phi2_model,
        "pm": background_pm_model,
    },
    name="background_astrometric_model",
)


# # -----------------------------------------------------------------------------
# # Photometry

# phot_flow_scaler = scaler[("phi1", *phot_coords)]

# background_photometric_model = sml.builtin.compat.ZukoFlowModel(
#     net=zuko.flows.MAF(2, 1, hidden_features=[8, 8, 8]),
#     jacobian_logdet=float(-xp.log(xp.prod(phot_flow_scaler.scale[1:]))),
#     data_scaler=phot_flow_scaler,
#     coord_names=phot_coords,
#     coord_bounds=phot_coord_bounds,
#     params=ModelParameters[xp.Tensor](),
#     with_grad=False,
#     name="background_photometric_model",
# )

# -----------------------------------------------------------------------------
# All

background_model = sml.IndependentModels(
    {
        "astrometric": background_astrometric_model,
        # "photometric": background_photometric_model,
    },
    name="background_model",
)


##############################################################################
# STREAM

# -----------------------------------------------------------------------------
# Astrometry

# Selection of control points
stream_astrometric_prior = sml.prior.ControlRegions(
    center=sml.Data.from_format(
        pal5_cp, fmt="astropy.table", names=("phi1", "phi2"), renamer=renamer
    ).astype(xp.Tensor, dtype=xp.float32),
    width=sml.Data.from_format(
        pal5_cp,
        fmt="astropy.table",
        names=("w_phi2",),
        renamer={"w_phi2": "phi2"},
    ).astype(xp.Tensor, dtype=xp.float32),
    lamda=1_000,
)

# the model
stream_astrometric_model = sml.builtin.TruncatedNormal(
    net=sml.nn.sequential(
        data=1,
        hidden_features=64,
        layers=4,
        features=2 * len(astro_coords),
        dropout=0.0,
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
                    bounds=SigmoidBounds(-4.0, -0.4),
                    scaler=StandardLnWidth.from_data_scaler(scaler, "phi2", xp=xp),
                ),
            },
            "pmphi1": {
                "mu": ModelParameter(
                    bounds=SigmoidBounds(2.5, 4.75),  # *coord_bounds["pmphi1"]
                    scaler=StandardLocation.from_data_scaler(scaler, "pmphi1", xp=xp),
                ),
                "ln-sigma": ModelParameter(
                    bounds=SigmoidBounds(-3.0, -0.3),
                    scaler=StandardLnWidth.from_data_scaler(scaler, "pmphi1", xp=xp),
                ),
            },
            "pmphi2": {
                "mu": ModelParameter(
                    bounds=SigmoidBounds(0, 2.2),  # *coord_bounds["pmphi2"]
                    scaler=StandardLocation.from_data_scaler(scaler, "pmphi2", xp=xp),
                ),
                "ln-sigma": ModelParameter(
                    bounds=SigmoidBounds(-3.0, -0.9),
                    scaler=StandardLnWidth.from_data_scaler(scaler, "pmphi2", xp=xp),
                ),
            },
        }
    ),
    priors=(stream_astrometric_prior,),
    name="stream_astrometric_model",
)

# # -----------------------------------------------------------------------------
# # Photometry

# # Selection of control points
# stream_photometric_prior = sml.prior.ControlRegions(
#     center=sml.Data.from_format(
#         pal5_cp, fmt="astropy.table", names=("phi1", "distmod"), renamer=renamer
#     ).astype(xp.Tensor, dtype=xp.float32),
#     width=sml.Data.from_format(
#         pal5_cp,
#         fmt="astropy.table",
#         names=("w_distmod",),
#         renamer={"w_distmod": "distmod"},
#     ).astype(xp.Tensor, dtype=xp.float32),
#     lamda=1_000,
# )

# with asdf.open(paths.data / "pal5" / "isochrone.asdf", mode="r") as af:
#     abs_mags = sml.Data(**af["isochrone_data"]).astype(xp.Tensor, dtype=xp.float32)

# stream_isochrone_spl = isochrone_spline(abs_mags["g", "r"].array, xp=np)

# gamma_edges = xp.concatenate(
#     [
#         xp.linspace(00, 0.43, 30),
#         xp.linspace(0.43, 0.5, 15),
#         xp.linspace(0.501, 1, 30),
#     ]
# )

# stream_mass_function = sml.builtin.StepwiseMassFunction(
#     boundaries=(0, 0.35, 0.56, 1.01),
#     # log_probs=(0.0, 0.0, 0.0),  # TODO: set a value
#     log_probs=(-1, 0, -1),
# )

# stream_isochrone_model = sml.builtin.IsochroneMVNorm(
#     net=sml.nn.sequential(
#         data=1, hidden_features=32, layers=4, features=2, dropout=0.15
#     ),
#     data_scaler=phot_flow_scaler,
#     # # coordinates
#     coord_names=("distmod",),
#     coord_bounds={"distmod": (13.0, 18.0)},
#     # coord_names=(),
#     # coord_bounds={},
#     # photometry
#     phot_names=phot_coords,
#     phot_apply_dm=(True, True),  # (g, r)
#     phot_err_names=phot_coord_errs,
#     phot_bounds=phot_coord_bounds,
#     # isochrone
#     gamma_edges=gamma_edges,
#     isochrone_spl=stream_isochrone_spl,
#     isochrone_err_spl=None,
#     stream_mass_function=stream_mass_function,
#     # params
#     params=ModelParameters[xp.Tensor](
#         {
#             "distmod": {
#                 "mu": ModelParameter[xp.Tensor](
#                     bounds=SigmoidBounds(13.0, 18.0), scaler=None
#                 ),
#                 "ln-sigma": ModelParameter[xp.Tensor](
#                     bounds=SigmoidBounds(-7.6, -2.8), scaler=None
#                 ),
#             },
#         }
#     ),
#     priors=(stream_photometric_prior,),
#     name="stream_isochrone_model",
# )


# -----------------------------------------------------------------------------

stream_model = sml.IndependentModels(
    {
        "astrometric": stream_astrometric_model,
        # "photometric": stream_isochrone_model,
    },
    name="stream_model",
)


# =============================================================================
# Mixture


_stream_wgt_prior = sml.prior.HardThreshold(
    threshold=0,  # turn off no matter what
    param_name=f"stream.{WEIGHT_NAME}",
    coord_name="phi1",
    data_scaler=scaler,
    set_to=-100,
)


_mx = {"stream": stream_model, "background": background_model}
model = sml.MixtureModel(
    _mx,
    net=sml.nn.sequential(
        data=1, hidden_features=64, layers=5, features=len(_mx) - 1, dropout=0.15
    ),
    data_scaler=scaler,
    params=ModelParameters[xp.Tensor](
        {
            f"stream.{WEIGHT_NAME}": ModelParameter[xp.Tensor](
                bounds=SigmoidBounds(-10.0, -0.01, neg_inf=-1e4), scaler=None
            ),
            f"background.{WEIGHT_NAME}": ModelParameter[xp.Tensor](
                bounds=SigmoidBounds(-5.0, 0.0, neg_inf=-1e4), scaler=None
            ),
        }
    ),
    priors=(
        # turn off below -16
        replace(_stream_wgt_prior, upper=-12, data_scaler=scaler),
        # turn off above 10
        replace(_stream_wgt_prior, lower=10, data_scaler=scaler),
        # # turn off around progenitor
        # replace(_stream_wgt_prior, lower=-0.25, upper=0.25, data_scaler=scaler),
    ),
    name="model",
)
