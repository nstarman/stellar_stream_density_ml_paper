"""Plot results."""

import sys
from dataclasses import replace

import asdf
import numpy as np
import torch as xp
import zuko
from astropy.table import QTable
from showyourwork.paths import user as user_paths

import stream_ml.pytorch as sml
from stream_ml.pytorch.params import ModelParameter, ModelParameters
from stream_ml.pytorch.params.bounds import SigmoidBounds
from stream_ml.pytorch.params.scaler import StandardLnWidth, StandardLocation

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.helper import isochrone_spline

##############################################################################

with asdf.open(paths.data / "pal5" / "info.asdf", mode="r") as af:
    renamer = af["renamer"]
    scaler = sml.utils.StandardScaler(**af["scaler"]).astype(
        xp.Tensor, dtype=xp.float32
    )
    all_coord_bounds = {k: tuple(v) for k, v in af["coord_bounds"].items()}

astro_coords = ("phi2",)
astro_coord_errs = ("phi2_err",)
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
    net=sml.nn.sequential(
        data=1, hidden_features=16, layers=2, features=len(coord_names), dropout=0.0
    ),
    data_scaler=scaler,
    coord_names=astro_coords,
    coord_err_names=astro_coord_errs,
    coord_bounds=astro_coord_bounds,
    params=ModelParameters(
        {
            "phi2": {
                "slope": ModelParameter(bounds=SigmoidBounds(-0.1, -0.01), scaler=None)
            },
        }
    ),
)

# -----------------------------------------------------------------------------
# Photometry

flow_scaler = scaler[("phi1", *phot_coords)]

background_photometric_model = sml.builtin.compat.ZukoFlowModel(
    net=zuko.flows.MAF(2, 1, hidden_features=[8, 8, 8]),
    jacobian_logdet=-xp.log(xp.prod(flow_scaler.scale[1:])),
    data_scaler=flow_scaler,
    coord_names=phot_coords,
    coord_bounds=phot_coord_bounds,
    params=ModelParameters(),
    with_grad=False,
    name="background_photometric_model",
)

# -----------------------------------------------------------------------------
# All

background_model = sml.IndependentModels(
    {
        "astrometric": background_astrometric_model,
        # "photometric": background_photometric_model,
    }
)


##############################################################################
# STREAM

# -----------------------------------------------------------------------------
# Astrometry

pal5_cp = QTable.read(paths.data / "pal5" / "control_points_stream.ecsv")

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
        hidden_features=32,
        layers=3,
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
                    bounds=SigmoidBounds(-3.0, 0.0),
                    scaler=StandardLnWidth.from_data_scaler(scaler, "phi2", xp=xp),
                ),
            },
        }
    ),
    priors=(stream_astrometric_prior,),
)

# -----------------------------------------------------------------------------
# Photometry

# Selection of control points
stream_photometric_prior = sml.prior.ControlRegions(
    center=sml.Data.from_format(
        pal5_cp, fmt="astropy.table", names=("phi1", "distmod"), renamer=renamer
    ).astype(xp.Tensor, dtype=xp.float32),
    width=sml.Data.from_format(
        pal5_cp,
        fmt="astropy.table",
        names=("w_distmod",),
        renamer={"w_distmod": "distmod"},
    ).astype(xp.Tensor, dtype=xp.float32),
    lamda=1_000,
)

with asdf.open(paths.data / "pal5" / "isochrone.asdf", mode="r") as af:
    abs_mags = sml.Data(**af["isochrone_data"]).astype(xp.Tensor, dtype=xp.float32)

stream_isochrone_spl = isochrone_spline(abs_mags["g", "r"].array, xp=np)

gamma_edges = xp.concatenate(
    [
        xp.linspace(00, 0.43, 30),
        xp.linspace(0.43, 0.5, 15),
        xp.linspace(0.501, 1, 30),
    ]
)

stream_mass_function = sml.builtin.StepwiseMassFunction(
    boundaries=(0, 0.35, 0.56, 1.01),
    # log_probs=(0.0, 0.0, 0.0),  # TODO: set a value
    log_probs=(-1, 0, -1),
)

stream_isochrone_model = sml.builtin.IsochroneMVNorm(
    net=sml.nn.sequential(
        data=1, hidden_features=32, layers=4, features=2, dropout=0.15
    ),
    data_scaler=flow_scaler,
    # # coordinates
    coord_names=("distmod",),
    coord_bounds={"distmod": (13.0, 18.0)},
    # coord_names=(),
    # coord_bounds={},
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
    name="stream_isochrone_model",
)


# -----------------------------------------------------------------------------

stream_model = sml.IndependentModels(
    {
        "astrometric": stream_astrometric_model,
        # "photometric": stream_isochrone_model,
    }
)


# =============================================================================
# Mixture


_stream_wgt_prior = sml.prior.HardThreshold(
    threshold=1,  # turn off no matter what
    param_name="stream.weight",
    coord_name="phi1",
    data_scaler=scaler,
)


_mx = {"stream": stream_model, "background": background_model}
model = sml.MixtureModel(
    _mx,
    net=sml.nn.sequential(
        data=1, hidden_features=16, layers=3, features=len(_mx) - 1, dropout=0.0
    ),
    data_scaler=scaler,
    params=ModelParameters(
        {
            "stream.weight": ModelParameter(
                bounds=SigmoidBounds(1e-3, 0.301), scaler=None
            ),
            "background.weight": ModelParameter(
                bounds=SigmoidBounds(0.7, 1.0), scaler=None
            ),
        }
    ),
    priors=(
        # turn off below -16
        replace(_stream_wgt_prior, upper=-16, data_scaler=scaler),
        # turn off above 10
        replace(_stream_wgt_prior, lower=10, data_scaler=scaler),
        # turn off around progenitor
        replace(_stream_wgt_prior, lower=0.1, upper=0.1, data_scaler=scaler),
    ),
)


# -----------------------------------------------------------------------------

pth = paths.data / "pal5" / "model.tmp"
if not pth.exists():
    with pth.open("w") as f:
        f.write("hack for snakemake")
