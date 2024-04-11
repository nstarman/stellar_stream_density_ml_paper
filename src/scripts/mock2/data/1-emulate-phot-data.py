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
from emulation_model import make_phot_data_model, make_phot_data_preprocess
from numpy.lib.recfunctions import structured_to_unstructured

# isort: split
jax.config.update("jax_enable_x64", True)


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
# Data

# ===================================================================
# Load the data

data_real = QTable.read(paths.data / "mock2" / "gd1_query_real.fits")
for col in data_real.colnames:
    data_real[col] = data_real[col].astype("float64")
data_real_df = data_real.to_pandas()

names = ("G", "G-R")
dims = len(names)

X = structured_to_unstructured(np.array(data_real[names]))
X = X[jnp.isfinite(X).all(-1)]

# ===================================================================
# Preprocess the data

if snkmk["load_from_static"]:
    preprocess_data = make_phot_data_preprocess(
        paths.static / "mock2" / "phot_data_preprocess.eqx"
    )

else:
    eps = 1e-2  # for numerical stability

    preprocess_data = fjxb.Chain(
        [
            fjxb.Loc(-jnp.array(tuple(map(jnp.min, X.T)))),
            fjxb.Scale(
                (2 - 2 * eps) / jnp.array([jnp.max(x) - jnp.min(x) for x in X.T])
            ),
            fjxb.Loc(-jnp.ones(dims) + eps),
        ]
    )

    # Save the flow info
    eqx.tree_serialise_leaves(
        paths.data / "mock2" / "phot_data_preprocess.eqx", preprocess_data
    )

    if snkmk["save_to_static"]:
        eqx.tree_serialise_leaves(
            paths.static / "mock2" / "phot_data_preprocess.eqx", preprocess_data
        )

# Transform the data
Xp = jax.vmap(preprocess_data.transform)(X)


if snkmk["diagnostic_plots"]:
    # ---------------------------------
    # Plot the distributions

    fig, axs = plt.subplots(dims, dims, figsize=(3 * dims, 3 * dims))

    grid_vecs = (
        jnp.linspace(-1, 1, 10),
        jnp.linspace(-1, 1, 10),
    )

    for i, j in itertools.product(range(dims), repeat=2):
        ax = axs[i, j]
        if i < j:
            ax.axis("off")  # Upper triangle
            continue
        if i == j:
            ax.hist(Xp[:, i], density=True, alpha=0.5, label="Data")
            ax.set_title(f"x_{i}")
        else:
            ax.scatter(Xp[:, j], Xp[:, i], s=2)

            # Contour of the real data
            xy = jnp.vstack([Xp[::50, j], Xp[::50, i]])
            kernel = jax.scipy.stats.gaussian_kde(xy)
            kde = kernel(xy)
            xx, yy = jnp.mgrid[
                grid_vecs[j][0] : grid_vecs[j][-1] : 10j,
                grid_vecs[i][0] : grid_vecs[i][-1] : 10j,
            ]
            zz = kernel(jnp.vstack([xx.flatten(), yy.flatten()])).reshape(xx.shape)
            ax.contour(xx, yy, zz, levels=3, colors="r")

            ax.set(
                xlim=(grid_vecs[j][0], grid_vecs[j][-1]),
                ylim=(grid_vecs[i][0], grid_vecs[i][-1]),
            )

        # Set labels on left and bottom edges
        if i == dims - 1:
            ax.set_xlabel(f"x_{j}")
        if j == 0:
            ax.set_ylabel(f"x_{i}")

    fig.tight_layout()
    fig.savefig(diagnostic_plots / "phot_data_distribution.png")


# ===================================================================
# Train the data model

if snkmk["load_from_static"]:
    flow = make_phot_data_model(paths.static / "mock2" / "phot_data_flow.eqx")
    normalized_flow = flow.base_dist

else:
    key, subkey = jr.split(jr.PRNGKey(0))

    untrained_flow = flowjax.flows.masked_autoregressive_flow(
        key=subkey,
        base_dist=fjdist.Normal(jnp.zeros(Xp.shape[1])),
        transformer=fjxb.Affine(),
        invert=False,
    )

    key, subkey = jr.split(key)

    normalized_flow, losses = flowjax.train.fit_to_data(
        key=subkey,
        dist=untrained_flow,
        x=Xp,
        learning_rate=1e-3,
        max_patience=100,
        max_epochs=2_000,
        return_best=True,
        batch_size=int(0.075 * len(Xp)),  # 7.5% of the data
    )
    flow = fjdist.Transformed(normalized_flow, fjxb.Invert(preprocess_data))

    # Save the flow info
    eqx.tree_serialise_leaves(paths.data / "mock2" / "phot_data_flow.eqx", flow)

    if snkmk["save_to_static"]:
        eqx.tree_serialise_leaves(paths.static / "mock2" / "phot_data_flow.eqx", flow)


# ---------------------------------
# Plot the distributions

if snkmk["diagnostic_plots"]:
    grid_vecs = (jnp.linspace(-1, 1, 12), jnp.linspace(-1, 1, 12))
    shape = tuple(len(vec) for vec in grid_vecs)
    meshgrids = jnp.meshgrid(*grid_vecs, indexing="ij")
    outgrid = jnp.exp(
        normalized_flow.log_prob(
            jnp.column_stack(
                tuple(
                    grid.reshape(-1, 1)
                    for grid in jnp.meshgrid(*grid_vecs, indexing="ij")
                )
            )
        )
    ).reshape(shape)

    c = normalized_flow.log_prob(Xp)

    fig, axs = plt.subplots(dims, dims, figsize=(3 * dims, 3 * dims))

    for i, j in itertools.product(range(dims), repeat=2):
        ax = axs[i, j]
        if i < j:
            ax.axis("off")  # Upper triangle
            continue
        if i == j:
            ax.hist(Xp[:, i], bins=grid_vecs[i], density=True, alpha=0.5, label="Data")
            ax.set_title(f"x_{i}")
        else:
            # Contours
            sum_axes = tuple(k for k in range(dims) if k not in [i, j])
            _x, _y = jnp.meshgrid(grid_vecs[j], grid_vecs[i], indexing="ij")
            ax.contourf(
                _x,
                _y,
                outgrid.sum(axis=sum_axes),
                # norm=LogNorm(),
                levels=30,
            )

            # Contour of the real data
            xy = jnp.vstack([Xp[::50, j], Xp[::50, i]])
            kernel = jax.scipy.stats.gaussian_kde(xy)
            kde = kernel(xy)
            xx, yy = jnp.mgrid[
                grid_vecs[j][0] : grid_vecs[j][-1] : 10j,
                grid_vecs[i][0] : grid_vecs[i][-1] : 10j,
            ]
            zz = kernel(jnp.vstack([xx.flatten(), yy.flatten()])).reshape(xx.shape)
            ax.contour(xx, yy, zz, levels=3, colors="k")

            ax.set(
                xlim=grid_vecs[j][jnp.array([0, -1])],
                ylim=grid_vecs[i][jnp.array([0, -1])],
            )

        # Set labels on left and bottom edges
        if i == dims - 1:
            ax.set_xlabel(f"x_{j}")
        if j == 0:
            ax.set_ylabel(f"x_{i}")

    fig.tight_layout()
    fig.savefig(diagnostic_plots / "phot_flow_distribution.png")


# ---------------------------------
# Plot samples

if snkmk["diagnostic_plots"]:
    # Sample
    key = jr.PRNGKey(84)
    phot_samples = flow.sample(key, (30_000 + 100,))
    phot_samples = phot_samples[phot_samples[:, 1] > -10]
    phot_samples = phot_samples[:30_000]

    samples = QTable(np.array(phot_samples), names=names, units=(u.mag, u.mag))
    samples_df = samples.to_pandas()

    # Plot Distributions
    fig, axs = plt.subplots(1, dims, figsize=(dims * 4.5, 4))
    for i, k in enumerate(names):
        _, bins, _ = axs[i].hist(
            samples_df[k], density=True, bins=40, alpha=0.5, label="synth"
        )
        axs[i].hist(
            data_real[k].value,
            density=True,
            bins=bins,
            alpha=0.5,
            label="data",
            histtype="step",
        )
        axs[i].set(xlabel=f"{k} [{samples[k].unit}]")
        axs[i].legend()

    fig.tight_layout()
    fig.savefig(diagnostic_plots / "phot_sample_distribution.png")
