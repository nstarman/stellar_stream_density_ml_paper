"""Emulate the background."""

import itertools
import sys

import astropy.units as u
import equinox as eqx
import flowjax.bijections as fjxb
import flowjax.distributions as fjdist
import flowjax.flows
import flowjax.train
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
from emulation_model import (
    make_phot_data_model,
    make_phot_error_model,
    make_phot_error_preprocess,
)
from numpy.lib.recfunctions import structured_to_unstructured

# isort: split
jax.config.update("jax_enable_x64", True)

import contextlib

from showyourwork.paths import user as user_paths

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split


##############################################################################
# Parameters

try:
    snkmk = snakemake.params
except NameError:
    snkmk = {
        "diagnostic_plots": True,
        "load_from_static": True,
        "save_to_static": True,
    }

diagnostic_plots = paths.scripts / "mock2" / "_diagnostics" / "phot"
diagnostic_plots.mkdir(parents=True, exist_ok=True)


##############################################################################
# Errors

data_real = QTable.read(paths.data / "mock2" / "gd1_query_real.fits")
for col in data_real.colnames:
    data_real[col] = data_real[col].astype("float64")

names = ("G", "G-R")
dims = len(names)

# ===================================================================
# Load the error table

error_names = tuple(f"{name}_error" for name in names)

dX = structured_to_unstructured(np.array(data_real[error_names]))

# Data munging
selXerr = jnp.isfinite(dX[:, 0]) & jnp.isfinite(dX[:, 1]) & (dX[:, 1] < 1)
dX = dX[selXerr]

if snkmk["diagnostic_plots"]:
    # ---------------------------------
    # Plot the distributions

    fig, axs = plt.subplots(dims, dims, figsize=(3 * dims, 3 * dims))

    grid_vecs = (
        jnp.linspace(-1e-4, 0.04, 10),  # G
        jnp.linspace(-1e-2, 0.5, 10),  # G-R
    )

    for i, j in itertools.product(range(dims), repeat=2):
        ax = axs[i, j]
        if i < j:
            ax.axis("off")  # Upper triangle
            continue
        if i == j:
            ax.hist(dX[:, i], density=True, bins=30, alpha=0.5, label="Data", log=True)
            ax.set_title(names[i])
        else:
            ax.scatter(dX[:, j], dX[:, i], s=2)

            # Contour of the real data
            xy = jnp.vstack([dX[::50, j], dX[::50, i]])
            kernel = jax.scipy.stats.gaussian_kde(xy)
            kde = kernel(xy)
            xx, yy = jnp.mgrid[
                grid_vecs[j][0] : grid_vecs[j][-1] : 10j,
                grid_vecs[i][0] : grid_vecs[i][-1] : 10j,
            ]
            zz = kernel(jnp.vstack([xx.flatten(), yy.flatten()])).reshape(xx.shape)
            ax.contour(xx, yy, zz, levels=3, colors="r")

        # Set labels on left and bottom edges
        if i == dims - 1:
            ax.set_xlabel(names[j])
        if j == 0:
            ax.set_ylabel(names[i])

    fig.tight_layout()
    fig.savefig(diagnostic_plots / "phot_error_distribution.png")


# ===================================================================
# Preprocess the data

if snkmk["load_from_static"]:
    preprocess_errors = make_phot_error_preprocess(
        paths.static / "mock2" / "phot_errors_preprocess.eqx"
    )

else:
    eps = 1e-1  # for numerical stability

    preprocess_errors = fjxb.Chain(
        [
            fjxb.Scale(
                (2 - 2 * eps) / jnp.array([jnp.max(x) - jnp.min(x) for x in dX.T])
            ),
            fjxb.Loc(-jnp.ones(dims) + eps),
            fjxb.Invert(fjxb.LeakyTanh(max_val=10, shape=(dims,))),
        ]
    )

    if snkmk["save_to_static"]:
        eqx.tree_serialise_leaves(
            paths.static / "mock2" / "phot_errors_preprocess.eqx", preprocess_errors
        )


# Save the flow info
eqx.tree_serialise_leaves(
    paths.data / "mock2" / "phot_errors_preprocess.eqx", preprocess_errors
)

# Transform the data
dXp = jax.vmap(preprocess_errors.transform)(dX)


if snkmk["diagnostic_plots"]:
    # ---------------------------------
    # Plot the transformed distributions

    fig, axs = plt.subplots(dims, dims, figsize=(3 * dims, 3 * dims))

    grid_vecs = (jnp.linspace(-2, 2, 10), jnp.linspace(-1.5, 1, 10))

    for i, j in itertools.product(range(dims), repeat=2):
        ax = axs[i, j]
        if i < j:
            ax.axis("off")  # Upper triangle
            continue
        if i == j:
            ax.hist(dXp[:, i], density=True, alpha=0.5, label="Data")
            ax.set_title(f"e_{i}")
        else:
            # Data
            ax.scatter(dXp[:, j], dXp[:, i], s=2)

            # Contour of the real data
            xy = jnp.vstack([dXp[::50, j], dXp[::50, i]])
            kernel = jax.scipy.stats.gaussian_kde(xy)
            kde = kernel(xy)
            xx, yy = jnp.mgrid[
                grid_vecs[j][0] : grid_vecs[j][-1] : 10j,
                grid_vecs[i][0] : grid_vecs[i][-1] : 10j,
            ]
            zz = kernel(jnp.vstack([xx.flatten(), yy.flatten()])).reshape(xx.shape)
            ax.contour(xx, yy, zz, levels=3, colors="r")

        # Set labels on left and bottom edges
        if i == dims - 1:
            ax.set_xlabel(f"e_{j}")
        if j == 0:
            ax.set_ylabel(f"e_{i}")

    fig.tight_layout()
    fig.savefig(diagnostic_plots / "phot_error_preprocess_distribution.png")


# ===================================================================
# Train the data model


if snkmk["load_from_static"]:
    phot_error_flow = make_phot_error_model(
        paths.static / "mock2" / "phot_error_flow.eqx"
    )
    normalized_phot_error_flow = phot_error_flow.base_dist

else:
    key, subkey = jr.split(jr.PRNGKey(37))

    untrained_phot_error_flow = flowjax.flows.masked_autoregressive_flow(
        key=subkey,
        base_dist=fjdist.Normal(jnp.zeros(dXp.shape[1]), scale=1),
        transformer=fjxb.Affine(),
        invert=False,
    )

    key, subkey = jr.split(key)

    normalized_phot_error_flow, losses = flowjax.train.fit_to_data(
        key=subkey,
        dist=untrained_phot_error_flow,
        x=dXp,
        learning_rate=5e-4,
        max_patience=100,
        max_epochs=2_000,
        return_best=True,
        batch_size=int(0.075 * len(dXp)),  # 7.5% of the data
    )
    phot_error_flow = fjdist.Transformed(
        normalized_phot_error_flow, fjxb.Invert(preprocess_errors)
    )

    if snkmk["save_to_static"]:
        eqx.tree_serialise_leaves(
            paths.static / "mock2" / "phot_error_flow.eqx", phot_error_flow
        )

# Save the flow info
eqx.tree_serialise_leaves(paths.data / "mock2" / "phot_error_flow.eqx", phot_error_flow)


# ---------------------------------
# Plot the distributions

if snkmk["diagnostic_plots"]:
    grid_vecs = (jnp.linspace(-1.6, -0.5, 20), jnp.linspace(-1.6, -0.5, 20))
    meshgrids = jnp.meshgrid(*grid_vecs, indexing="ij")
    xyinput = jnp.column_stack(tuple(grid.reshape(-1, 1) for grid in meshgrids))
    outgrid = jnp.exp(normalized_phot_error_flow.log_prob(xyinput)).reshape(
        meshgrids[0].shape
    )

    c = normalized_phot_error_flow.log_prob(dXp)

    fig, axs = plt.subplots(dims, dims, figsize=(3 * dims, 3 * dims))

    for i, j in itertools.product(range(dims), repeat=2):
        ax = axs[i, j]
        if i < j:
            ax.axis("off")  # Upper triangle
            continue
        if i == j:
            ax.hist(dXp[:, i], bins=grid_vecs[i], density=True, alpha=0.5, label="Data")
            ax.set_title(f"e_{i}")
        else:
            # Contour of the flow
            sum_axes = tuple(k for k in range(dims) if k not in [i, j])
            sel = tuple(slice(None) if k in [i, j] else 0 for k in range(dims))
            ax.contourf(
                meshgrids[j][sel],
                meshgrids[i][sel],
                outgrid.sum(axis=sum_axes),
                levels=10,
            )

            # Contour of the real data
            xy = jnp.vstack([dXp[::20, j], dXp[::20, i]])
            kernel = jax.scipy.stats.gaussian_kde(xy)
            kde = kernel(xy)
            xx, yy = jnp.mgrid[
                grid_vecs[j][0] : grid_vecs[j][-1] : 10j,
                grid_vecs[i][0] : grid_vecs[i][-1] : 10j,
            ]
            zz = kernel(jnp.vstack([xx.flatten(), yy.flatten()])).reshape(xx.shape)
            ax.contour(xx, yy, zz, levels=4, colors="r")

            ax.set(
                xlim=grid_vecs[j][jnp.array([0, -1])],
                ylim=grid_vecs[i][jnp.array([0, -1])],
            )

        # Set labels on left and bottom edges
        if i == dims - 1:
            ax.set_xlabel(f"e_{j}")
        if j == 0:
            ax.set_ylabel(f"e_{i}")

    fig.tight_layout()
    fig.savefig(diagnostic_plots / "phot_error_flow_distribution.png")


# ---------------------------------
# Plot samples

if snkmk["diagnostic_plots"]:
    # Sample distribution
    key = jr.PRNGKey(84)
    data_flow = make_phot_data_model(paths.static / "mock2" / "phot_data_flow.eqx")
    phot_samples = data_flow.sample(key, (30_000 + 100,))
    phot_samples = phot_samples[phot_samples[:, 1] > -10]
    phot_samples = phot_samples[:30_000]

    samples = QTable(np.array(phot_samples), names=names, units=(u.mag, u.mag))
    samples_df = samples.to_pandas()

    # Sample errors
    key = jr.PRNGKey(84)
    error_samples = phot_error_flow.sample(key, (len(samples),))

    with contextlib.suppress(KeyError):
        samples.remove_columns(error_names)

    for i, (k, ke) in enumerate(zip(names, error_names, strict=True)):
        samples.add_column(
            error_samples[:, i] * samples[k].unit,
            index=list(samples.columns).index(k) + 1,
            name=ke,
        )

    # Re-make the DataFrame
    samples_df = samples.to_pandas()

    # Plot Distributions
    fig, axs = plt.subplots(1, dims, figsize=(dims * 4.5, 4))
    for i, k in enumerate(error_names):
        _, bins, _ = axs[i].hist(
            samples_df[k], density=True, bins=30, alpha=0.5, label="synth"
        )
        axs[i].hist(
            data_real[k].value,
            density=True,
            bins=bins,
            alpha=0.5,
            label="sim",
            histtype="step",
        )
        axs[i].set(xlabel=f"{k} [{data_real[k].unit}]")
        axs[i].legend()

    fig.tight_layout()
    fig.savefig(diagnostic_plots / "phot_error_sample_distribution.png")
