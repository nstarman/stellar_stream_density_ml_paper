"""Emulate the background."""

import sys

import equinox as eqx
import flowjax.bijections as fjxb
import flowjax.distributions as fjxdist
import flowjax.flows
import flowjax.train
import jax
import jax.numpy as jnp
import jax.random as jr
from matmul import MatMul

# isort: split
jax.config.update("jax_enable_x64", True)

from showyourwork.paths import user as user_paths

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split


##############################################################################
# Parameters


def make_astro_data_preprocess(path: str | None = None) -> fjxb.AbstractBijection:
    """Make astro data preprocessor."""
    bijection = fjxb.Chain(
        [
            fjxb.Loc(jnp.array([0.0] * 5)),
            fjxb.Scale(jnp.array([1.0] * 5)),
            fjxb.Loc(jnp.zeros(5)),
            fjxb.Invert(fjxb.LeakyTanh(max_val=1, shape=(5,))),
            fjxb.TriangularAffine(loc=0, arr=jnp.eye(5)),
            fjxb.Scale(jnp.ones(5)),
        ],
    )

    if path is not None:
        bijection = eqx.tree_deserialise_leaves(path, bijection)

    return bijection


def make_astro_data_model(path: str | None = None) -> fjxdist.Transformed:
    """Make astro data model."""
    key, subkey = jr.split(jr.PRNGKey(0))

    untrained_flow = flowjax.flows.masked_autoregressive_flow(
        key=subkey,
        base_dist=fjxdist.Normal(jnp.zeros(5)),
        transformer=fjxb.Affine(),
        invert=False,
    )

    preprocess = make_astro_data_preprocess(None)

    flow = fjxdist.Transformed(untrained_flow, fjxb.Invert(preprocess))

    if path is not None:
        flow = eqx.tree_deserialise_leaves(path, flow)

    return flow


def make_astro_error_preprocess(path: str | None = None) -> fjxb.AbstractBijection:
    """Make astro error data preprocessor."""
    bijection = fjxb.Chain(
        [
            fjxb.Scale(jnp.ones(5)),
            fjxb.Loc(jnp.zeros(5)),
            fjxb.Invert(fjxb.LeakyTanh(max_val=1, shape=(5,))),
            MatMul(jnp.eye(5)),
            fjxb.Scale(jnp.ones(5)),
        ]
    )

    if path is not None:
        bijection = eqx.tree_deserialise_leaves(path, bijection)

    return bijection


def make_astro_error_model(path: str | None = None) -> fjxdist.Transformed:
    """Make astro error model."""
    key, subkey = jr.split(jr.PRNGKey(37))

    untrained_flow = flowjax.flows.masked_autoregressive_flow(
        key=subkey,
        base_dist=fjxdist.Normal(jnp.zeros(5), scale=1),
        transformer=fjxb.Affine(),
        invert=False,
    )

    preprocess_errors = make_astro_error_preprocess(None)

    flow = fjxdist.Transformed(untrained_flow, fjxb.Invert(preprocess_errors))

    if path is not None:
        flow = eqx.tree_deserialise_leaves(path, flow)

    return flow


def make_phot_data_preprocess(path: str | None = None) -> fjxb.AbstractBijection:
    """Make photometric data preprocessor."""
    bijection = fjxb.Chain(
        [
            fjxb.Loc(jnp.array([0.0] * 2)),
            fjxb.Scale(jnp.array([1.0] * 2)),
            fjxb.Loc(jnp.zeros(2)),
        ],
    )

    if path is not None:
        bijection = eqx.tree_deserialise_leaves(path, bijection)

    return bijection


def make_phot_data_model(path: str | None = None) -> fjxdist.Transformed:
    """Make photometric data model."""
    key, subkey = jr.split(jr.PRNGKey(0))

    untrained_flow = flowjax.flows.masked_autoregressive_flow(
        key=subkey,
        base_dist=fjxdist.Normal(jnp.zeros(2)),
        transformer=fjxb.Affine(),
        invert=False,
    )

    preprocess = make_phot_data_preprocess(None)

    flow = fjxdist.Transformed(untrained_flow, fjxb.Invert(preprocess))

    if path is not None:
        flow = eqx.tree_deserialise_leaves(path, flow)

    return flow


def make_phot_error_preprocess(path: str | None = None) -> fjxb.AbstractBijection:
    """Make photometric error data preprocessor."""
    bijection = fjxb.Chain(
        [
            fjxb.Scale(jnp.ones(2)),
            fjxb.Loc(jnp.zeros(2)),
            fjxb.Invert(fjxb.LeakyTanh(max_val=1, shape=(2,))),
        ]
    )

    if path is not None:
        bijection = eqx.tree_deserialise_leaves(path, bijection)

    return bijection


def make_phot_error_model(path: str | None = None) -> fjxdist.Transformed:
    """Make photometric error model."""
    key, subkey = jr.split(jr.PRNGKey(37))

    untrained_phot_error_flow = flowjax.flows.masked_autoregressive_flow(
        key=subkey,
        base_dist=fjxdist.Normal(jnp.zeros(2), scale=1),
        transformer=fjxb.Affine(),
        invert=False,
    )

    preprocess = make_phot_error_preprocess(None)

    flow = fjxdist.Transformed(untrained_phot_error_flow, fjxb.Invert(preprocess))

    if path is not None:
        flow = eqx.tree_deserialise_leaves(path, flow)

    return flow
