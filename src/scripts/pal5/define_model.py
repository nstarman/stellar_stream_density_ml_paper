"""Plot results."""

import sys

import asdf
import torch as xp
from astropy.table import QTable
from showyourwork.paths import user as user_paths

import stream_ml.pytorch as sml

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.as_posix())
# isort: split

from scripts import paths
from stream_ml.pytorch.params import ModelParameter, ModelParameters
from stream_ml.pytorch.params.bounds import SigmoidBounds
from stream_ml.pytorch.params.scaler import StandardLnWidth, StandardLocation

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

phot_coords = ()
phot_coord_errs = ()
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
        n_in=1, n_hidden=32, n_layers=4, n_out=len(coord_names), dropout=0.15
    ),
    data_scaler=scaler,
    coord_names=astro_coords,
    coord_err_names=astro_coord_errs,
    coord_bounds=astro_coord_bounds,
    params=ModelParameters(
        {
            "phi2": {
                "slope": ModelParameter(bounds=SigmoidBounds(-0.15, 0.0), scaler=None)
            },
        }
    ),
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

pal5_cp = QTable.read(paths.data / "pal5" / "stream_control_points.ecsv")

# Selection of control points
stream_strometric_prior = sml.prior.ControlRegions(
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
        }
    ),
    priors=(stream_strometric_prior,),
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


_mx = {"stream": stream_model, "background": background_model}
model = sml.MixtureModel(
    _mx,
    net=sml.nn.lin_tanh(
        n_in=1, n_hidden=32, n_layers=4, n_out=len(_mx) - 1, dropout=0.15
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
    # priors=(
    #     sml.prior.HardThreshold(
    #         1,
    #         set_to=1e-3,
    #         upper=-90,
    #         param_name="stream.weight",
    #         coord_name="phi1",
    #         data_scaler=scaler,
    #     ),
    #     sml.prior.HardThreshold(
    #         1,
    #         set_to=1e-3,
    #         lower=10,
    #         param_name="stream.weight",
    #         coord_name="phi1",
    #         data_scaler=scaler,
    #     ),
    # ),
)


# -----------------------------------------------------------------------------

pth = paths.data / "pal5" / "model.tmp"
if not pth.exists():
    with pth.open("w") as f:
        f.write("hack for snakemake")
