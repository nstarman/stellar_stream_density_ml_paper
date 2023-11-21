"""Plot results."""

from __future__ import annotations

import sys
from dataclasses import KW_ONLY, dataclass, replace
from math import inf
from typing import TYPE_CHECKING, Any

import asdf
import astropy.units as u
import numpy as np
import torch as xp
import zuko
from astropy.table import QTable
from showyourwork.paths import user as user_paths

import stream_ml.pytorch as sml
from stream_ml.core.prior import Prior
from stream_ml.core.setup_package import WEIGHT_NAME
from stream_ml.core.utils import within_bounds
from stream_ml.pytorch import Data, Params
from stream_ml.pytorch.params import ModelParameter, ModelParameters, set_param
from stream_ml.pytorch.params.bounds import SigmoidBounds
from stream_ml.pytorch.params.scaler import StandardLnWidth, StandardLocation
from stream_ml.pytorch.typing import Array, NNModel
from stream_ml.pytorch.utils import StandardScaler

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.helper import isochrone_spline

if TYPE_CHECKING:
    from stream_ml.core import ModelAPI

##############################################################################
# Setup

# Load data
distance_cp = QTable.read(paths.data / "gd1" / "control_points_distance.ecsv")
gd1_cp_ = QTable.read(paths.data / "gd1" / "control_points_stream.ecsv")
spur_cp = QTable.read(paths.data / "gd1" / "control_points_spur.ecsv")


@dataclass(frozen=True, repr=False)
class UpWeight(Prior[Array]):
    """Force the weight to be larger than a threshold."""

    threshold: float

    _: KW_ONLY
    component: str = "spur"
    lower: float = -inf
    upper: float = inf
    lamda: float = 100_000

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        wgt_name = f"{self.component}.ln-weight"
        self._wgt_name: str
        object.__setattr__(self, "_wgt_name", wgt_name)

        super().__post_init__(*args, **kwargs)

    def logpdf(
        self,
        mpars: Params[Array],
        data: Data[Array],
        model: ModelAPI[Array, NNModel],  # noqa: ARG002
        current_lnpdf: Array | None = None,
        /,
    ) -> Array:
        """Force the weight to be larger than a threshold."""
        wgt_name = f"{self.component}.ln-weight"

        # Get where the weight is below the threshold
        where = (mpars[wgt_name] < self.threshold) & within_bounds(
            data["phi1"], self.lower, self.upper
        )

        # modify the lnpdf
        current_lnpdf[where] -= self.lamda * (
            (mpars[wgt_name][where] - self.threshold) ** 2
        )
        return current_lnpdf


##############################################################################
# Define Model


def make_model() -> sml.MixtureModel:
    """Define the model.

    Returns
    -------
    model : sml.MixtureModel
        The model.
    """
    scaler: StandardScaler
    with asdf.open(paths.data / "gd1" / "info.asdf", mode="r") as af:
        renamer = af["renamer"]
        scaler = StandardScaler(**af["scaler"]).astype(xp.Tensor, dtype=xp.float32)
        all_coord_bounds = {
            k: (v[0] - 1e-10, v[1] + 1e-10) for k, v in af["coord_bounds"].items()
        }

    # -----------------------------------------------------------------------------

    astro_coords = ("phi2", "plx", "pmphi1", "pmphi2")
    astro_coord_errs = ("phi2_err", "plx_err", "pmphi1_err", "pmphi2_err")
    astro_coord_bounds = {
        k: v for k, v in all_coord_bounds.items() if k in astro_coords
    }

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

    bkg1_coord_names = ("phi2",)
    background_astrometric_phi2_model = sml.builtin.Exponential(
        net=sml.nn.sequential(
            data=1, hidden_features=64, layers=4, features=2, dropout=0.15
        ),
        data_scaler=scaler,
        coord_names=bkg1_coord_names,
        coord_err_names=("phi2_err",),
        coord_bounds={
            k: v for k, v in astro_coord_bounds.items() if k in bkg1_coord_names
        },
        params=ModelParameters(
            {
                "phi2": {
                    "slope": ModelParameter(
                        bounds=SigmoidBounds(-0.03, 0.03), scaler=None
                    )
                },
            }
        ),
        name="background_astrometric_phi2_model",
    )

    bkg2_coord_names = ("plx", "pmphi1", "pmphi2")
    flow_plx_scaler = scaler[("phi1", *bkg2_coord_names)]
    background_astrometric_else_model = sml.builtin.compat.ZukoFlowModel(
        net=zuko.flows.NSF(3, 1, bins=20, hidden_features=[32] * 5),
        jacobian_logdet=float(-xp.log(xp.prod(flow_plx_scaler.scale[1:]))),
        data_scaler=flow_plx_scaler,
        indep_coord_names=("phi1",),
        coord_names=bkg2_coord_names,
        coord_bounds={k: coord_bounds[k] for k in bkg2_coord_names},
        params=ModelParameters[xp.Tensor](),
        with_grad=False,
        name="background_astrometric_else_model",
    )

    background_astrometric_model = sml.IndependentModels(
        {
            "phi2": background_astrometric_phi2_model,
            "else": background_astrometric_else_model,
        }
    )

    # -----------------------------------------------------------------------------
    # Photometry

    phot_flow_scaler = scaler[("phi1", *phot_coords)]

    background_photometric_model = sml.builtin.compat.ZukoFlowModel(
        net=zuko.flows.MAF(2, 1, hidden_features=[8, 8, 8]),
        jacobian_logdet=float(-xp.log(xp.prod(phot_flow_scaler.scale[1:]))),
        data_scaler=phot_flow_scaler,
        coord_names=phot_coords,
        coord_bounds=phot_coord_bounds,
        params=ModelParameters[xp.Tensor](),
        with_grad=False,
        name="background_photometric_model",
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

    gd1_cp = gd1_cp_[(-70 * u.deg <= gd1_cp_["phi1"]) & (gd1_cp_["phi1"] <= 10 * u.deg)]

    # Control points
    stream_astrometric_prior = sml.prior.ControlRegions(  # type: ignore[type-var]
        center=sml.Data.from_format(  # type: ignore[type-var]
            gd1_cp,
            fmt="astropy.table",
            names=("phi1", "phi2", "pm_phi1"),
            renamer=renamer,
        ).astype(xp.Tensor, dtype=xp.float32),
        width=sml.Data.from_format(  # type: ignore[type-var]
            gd1_cp,
            fmt="astropy.table",
            names=("w_phi2", "w_pm_phi1"),
            renamer={"w_phi2": "phi2", "w_pm_phi1": "pmphi1"},
        ).astype(xp.Tensor, dtype=xp.float32),
        lamda=1_000,
    )

    # TODO: put the parallax in the control points file
    stream_distance_prior = sml.prior.ControlRegions(  # type: ignore[type-var]
        center=sml.Data.from_format(  # type: ignore[type-var]
            distance_cp,
            fmt="astropy.table",
            names=("phi1", "parallax"),
            renamer=renamer,
        ).astype(xp.Tensor, dtype=xp.float32),
        width=sml.Data.from_format(  # type: ignore[type-var]
            distance_cp,
            fmt="astropy.table",
            names=("w_parallax",),
            renamer={"w_parallax": "plx"},
        ).astype(xp.Tensor, dtype=xp.float32),
        lamda=1_000,
    )

    stream_astrometric_model = sml.builtin.TruncatedNormal(
        net=sml.nn.sequential(
            data=1,
            hidden_features=128,
            layers=5,
            features=2 * len(astro_coords),
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
                "plx": {
                    "mu": ModelParameter(
                        # ensure parallax > 0, required for the distance modulus
                        bounds=SigmoidBounds(0.025, 0.2),  # force closer to answer
                        scaler=StandardLocation.from_data_scaler(scaler, "plx", xp=xp),
                    ),
                    "ln-sigma": ModelParameter(
                        bounds=SigmoidBounds(-7.0, -2.0),
                        scaler=StandardLnWidth.from_data_scaler(scaler, "plx", xp=xp),
                    ),
                },
                "pmphi1": {
                    "mu": ModelParameter(
                        bounds=SigmoidBounds(*coord_bounds["pmphi1"]),
                        scaler=StandardLocation.from_data_scaler(
                            scaler, "pmphi1", xp=xp
                        ),
                    ),
                    "ln-sigma": ModelParameter(
                        bounds=SigmoidBounds(-3.0, -0.5),
                        scaler=StandardLnWidth.from_data_scaler(
                            scaler, "pmphi1", xp=xp
                        ),
                    ),
                },
                "pmphi2": {
                    "mu": ModelParameter(
                        bounds=SigmoidBounds(*coord_bounds["pmphi2"]),
                        scaler=StandardLocation.from_data_scaler(
                            scaler, "pmphi2", xp=xp
                        ),
                    ),
                    "ln-sigma": ModelParameter(
                        bounds=SigmoidBounds(-3.0, 0),
                        scaler=StandardLnWidth.from_data_scaler(
                            scaler, "pmphi2", xp=xp
                        ),
                    ),
                },
            }
        ),
        priors=(
            stream_astrometric_prior,
            stream_distance_prior,
        ),
        name="stream_astrometric_model",
    )

    # -----------------------------------------------------------------------------
    # Photometry

    with asdf.open(paths.data / "gd1" / "isochrone.asdf", mode="r") as af:
        abs_mags = sml.Data(**af["isochrone_data"]).astype(xp.Tensor, dtype=xp.float32)

    stream_isochrone_spl = isochrone_spline(abs_mags["g", "r"].array, xp=np)

    # concentrating on the MS turnoff
    gamma_edges = xp.concatenate(
        [
            xp.linspace(00, 0.43, 30),
            xp.linspace(0.43, 0.5, 15),
            xp.linspace(0.501, 1, 30),
        ]
    )

    stream_mass_function = sml.builtin.StepwiseMassFunction(
        boundaries=(0, 0.35, 0.56, 1.01),
        log_probs=(-1, 0, -1),
    )

    stream_isochrone_model = sml.builtin.IsochroneMVNorm(
        data_scaler=phot_flow_scaler,
        coord_names=(),
        coord_bounds={},
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
        params=ModelParameters[xp.Tensor](),
        name="stream_isochrone_model",
    )

    # -----------------------------------------------------------------------------

    stream_model = sml.IndependentModels(
        {
            "astrometric": stream_astrometric_model,
            "photometric": stream_isochrone_model,
        },
        unpack_params_hooks=(
            sml.builtin.Parallax2DistMod(
                astrometric_coord="astrometric.plx",
                photometric_coord="photometric.distmod",
            ),
        ),
    )

    ##############################################################################
    # SPUR

    # =============================================================================
    # Astrometry

    spur_cp_prior = sml.prior.ControlRegions(
        center=sml.Data.from_format(
            spur_cp,
            fmt="astropy.table",
            names=("phi1", "phi2", "pm_phi1"),
            renamer=renamer,
        ).astype(xp.Tensor, dtype=xp.float32),
        width=sml.Data.from_format(
            spur_cp,
            fmt="astropy.table",
            names=("w_phi2", "w_pm_phi1"),
            renamer={"w_phi2": "phi2", "w_pm_phi1": "pmphi1"},
        ).astype(xp.Tensor, dtype=xp.float32),
        lamda=10_000,
    )

    astro_coords_mplx = tuple(c for c in astro_coords if c != "plx")
    astro_coord_errs_mplx = tuple(c for c in astro_coord_errs if c != "plx_err")
    astro_coord_bounds_mplx = {
        k: v for k, v in astro_coord_bounds.items() if k != "plx"
    }
    spur_astrometric_model = sml.builtin.Normal(
        net=sml.nn.sequential(
            data=1,
            hidden_features=64,
            layers=4,
            features=2 * (len(astro_coords) - 1),  # no parallax
            dropout=0.15,
        ),
        data_scaler=scaler,
        coord_names=astro_coords_mplx,
        coord_err_names=astro_coord_errs_mplx,
        coord_bounds=astro_coord_bounds_mplx,
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
                        scaler=StandardLocation.from_data_scaler(
                            scaler, "pmphi1", xp=xp
                        ),
                    ),
                    "ln-sigma": ModelParameter(
                        bounds=SigmoidBounds(-3, 0),
                        scaler=StandardLnWidth.from_data_scaler(
                            scaler, "pmphi1", xp=xp
                        ),
                    ),
                },
                "pmphi2": {
                    "mu": ModelParameter(
                        bounds=SigmoidBounds(*coord_bounds["pmphi2"]),
                        scaler=StandardLocation.from_data_scaler(
                            scaler, "pmphi2", xp=xp
                        ),
                    ),
                    "ln-sigma": ModelParameter(
                        bounds=SigmoidBounds(-3, 0),
                        scaler=StandardLnWidth.from_data_scaler(
                            scaler, "pmphi2", xp=xp
                        ),
                    ),
                },
            }
        ),
        priors=(spur_cp_prior,),
    )

    spur_model = sml.IndependentModels(
        {
            "astrometric": spur_astrometric_model,
            "photometric": stream_isochrone_model,
        }
    )

    ##############################################################################
    # MIXTURE

    def spur_shares_stream_distmod(params: dict) -> dict:
        """Set the spur distance modulus to be the same as the stream."""
        # Set the distance parallax
        set_param(
            params,
            ("spur.astrometric.plx", "mu"),
            params["stream.astrometric.plx"]["mu"],
        )
        set_param(
            params,
            ("spur.astrometric.plx", "ln-sigma"),
            params["stream.astrometric.plx"]["ln-sigma"],
        )

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

    stream_wgt_prior = sml.prior.HardThreshold(
        threshold=0,  # turn off no matter what
        param_name=f"stream.{WEIGHT_NAME}",
        coord_name="phi1",
        data_scaler=scaler,
    )
    spur_wgt_prior = replace(
        stream_wgt_prior, param_name=f"spur.{WEIGHT_NAME}", data_scaler=scaler
    )

    spur_force_wgt_prior = UpWeight(
        threshold=-4.0,
        lower=-45.0,
        upper=-15.0,
        lamda=0,
        component="spur",
        array_namespace="torch",
    )

    mm = {"stream": stream_model, "spur": spur_model, "background": background_model}
    model = sml.MixtureModel(
        mm,
        net=sml.nn.sequential(
            data=1, hidden_features=32, layers=4, features=len(mm) - 1, dropout=0.15
        ),
        data_scaler=scaler,
        params=ModelParameters[xp.Tensor](
            {
                f"stream.{WEIGHT_NAME}": ModelParameter[xp.Tensor](
                    bounds=SigmoidBounds(-10.0, -0.01), scaler=None
                ),
                f"spur.{WEIGHT_NAME}": ModelParameter[xp.Tensor](
                    bounds=SigmoidBounds(-10.0, -0.01), scaler=None
                ),
                f"background.{WEIGHT_NAME}": ModelParameter[xp.Tensor](
                    bounds=SigmoidBounds(-5.0, 0.0), scaler=None
                ),
            }
        ),
        unpack_params_hooks=(spur_shares_stream_distmod,),  # stream => spur parallax
        priors=(
            # stream: turn off below -90
            replace(stream_wgt_prior, upper=-90, data_scaler=scaler),
            # stream: turn off above 10
            replace(stream_wgt_prior, lower=10, data_scaler=scaler),
            # spur: turn off below -45
            replace(spur_wgt_prior, upper=-45, data_scaler=scaler),
            # spur: turn off above -15
            replace(spur_wgt_prior, lower=-15, data_scaler=scaler),
            # spur: optionally ramp it up
            spur_force_wgt_prior,
        ),
    )

    return model  # noqa: RET504
