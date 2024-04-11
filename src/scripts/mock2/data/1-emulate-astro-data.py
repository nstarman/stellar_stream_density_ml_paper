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
from astropy.coordinates import Distance
from astropy.table import QTable
from emulation_model import (
    make_astro_data_model,
    make_astro_data_preprocess,
)
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

diagnostic_plots = paths.scripts / "mock2" / "_diagnostics" / "astro"
diagnostic_plots.mkdir(parents=True, exist_ok=True)

##############################################################################
# Data

# ===================================================================
# Load the data

data_sim = QTable.read(paths.data / "mock2" / "gd1_query_sim.fits")
for col in data_sim.colnames:
    data_sim[col] = data_sim[col].astype("float64")

names = ("ra", "dec", "parallax", "pmra", "pmdec")
dims = len(names)

X = structured_to_unstructured(np.array(data_sim[names]))

# ===================================================================
# Preprocess the data

eps = 1e-2  # for numerical stability

if snkmk["load_from_static"]:
    preprocess_data = make_astro_data_preprocess(
        paths.static / "mock2" / "astro_data_preprocess.eqx"
    )

else:
    preprocess_data = fjxb.Chain(
        [
            fjxb.Loc(-jnp.array(tuple(map(jnp.min, X.T)))),
            fjxb.Scale(
                (2 - 2 * eps) / jnp.array([jnp.max(x) - jnp.min(x) for x in X.T])
            ),
            fjxb.Loc(-jnp.ones(dims) + eps),
            fjxb.Invert(fjxb.LeakyTanh(max_val=5, shape=(dims,))),
            fjxb.TriangularAffine(
                loc=0,  # NOTE: there are better inversion methods
                arr=jnp.linalg.cholesky(jnp.linalg.inv(jnp.cov(X.T))),
            ),
            fjxb.Scale(jnp.array([10, 2, 0.1, 20, 20])),
        ]
    )

    # Save the flow info
    eqx.tree_serialise_leaves(
        paths.data / "mock2" / "astro_data_preprocess.eqx", preprocess_data
    )

    if snkmk["save_to_static"]:
        eqx.tree_serialise_leaves(
            paths.static / "mock2" / "astro_data_preprocess.eqx", preprocess_data
        )

# Transformed data
Xp = jax.vmap(preprocess_data.transform)(X)


if snkmk["diagnostic_plots"]:
    # -------------------------------------------
    # Plot the transformed data

    fig, axs = plt.subplots(1, dims, figsize=(15, 4))
    for i in range(dims):
        axs[i].scatter(X[:, 2], Xp[:, i], s=3)
        axs[i].set_xlabel(names[2])
        axs[i].set_ylabel(f"scaled {names[i]}")
    fig.tight_layout()
    fig.savefig(diagnostic_plots / "astro_data_preprocess_distances.png")

    # -------------------------------------------
    # Plot the distributions

    fig, axs = plt.subplots(dims, dims, figsize=(3 * dims, 3 * dims))

    grid_vecs = (
        jnp.linspace(-2.5, 2.5, 10),
        jnp.linspace(-2, 2, 10),
        jnp.linspace(-3, 3, 10),
        jnp.linspace(-1.5, 3, 10),
        jnp.linspace(-1.5, 3, 10),
    )

    for i, j in itertools.product(range(dims), repeat=2):
        ax = axs[i, j]
        if i < j:
            ax.axis("off")  # Upper triangle
            continue
        if i == j:
            ax.hist(Xp[:, i], density=True, bins=30, alpha=0.5, label="Data")
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
    fig.savefig(diagnostic_plots / "astro_data_preprocess_distributions.png")


# ===================================================================
# Train the data model

if snkmk["load_from_static"]:
    flow = make_astro_data_model(paths.static / "mock2" / "astro_data_flow.eqx")
    normalized_flow = flow.base_dist

else:
    key, subkey = jr.split(jr.PRNGKey(0))

    untrained_flow = flowjax.flows.masked_autoregressive_flow(
        key=subkey,
        base_dist=fjdist.Normal(jnp.zeros(dims)),
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
    eqx.tree_serialise_leaves(paths.data / "mock2" / "astro_data_flow.eqx", flow)

    if snkmk["save_to_static"]:
        eqx.tree_serialise_leaves(paths.static / "mock2" / "astro_data_flow.eqx", flow)


if snkmk["diagnostic_plots"]:
    resolution = 12  # 20^5 is big. Don't want to go much higher.

    grid_vecs = (
        jnp.linspace(-3.5, 2.5, resolution),
        jnp.linspace(-2.5, 2.5, resolution),
        jnp.linspace(-5, 5, resolution),
        jnp.linspace(-1, 2.3, resolution),
        jnp.linspace(1, 2.2, resolution),
    )
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

    fig, axs = plt.subplots(dims, dims, figsize=(3 * dims, 3 * dims))

    c = normalized_flow.log_prob(Xp)

    for i, j in itertools.product(range(dims), repeat=2):
        ax = axs[i, j]
        if i < j:
            ax.axis("off")  # Upper triangle
            continue
        if i == j:
            ax.hist(Xp[:, i], density=True, bins=30, alpha=0.5, label="Data")
            ax.set_title(f"x_{i}")
        else:
            # Contours
            sum_axes = tuple(k for k in range(dims) if k not in [i, j])
            _x, _y = jnp.meshgrid(grid_vecs[j], grid_vecs[i], indexing="ij")
            ax.contourf(_x, _y, outgrid.sum(axis=sum_axes), levels=100)

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
                xlim=grid_vecs[j][jnp.array([0, -1])],
                ylim=grid_vecs[i][jnp.array([0, -1])],
            )

        # Set labels on left and bottom edges
        if i == dims - 1:
            ax.set_xlabel(f"x_{j}")
        if j == 0:
            ax.set_ylabel(f"x_{i}")

    fig.tight_layout()
    fig.savefig(diagnostic_plots / "astro_data_flow_distributions.png")


# -------------------------------------------
# Plot samples

if snkmk["diagnostic_plots"]:
    key = jr.PRNGKey(56)
    samples = flow.sample(key, (30_000,))

    samples = QTable(
        np.array(samples),
        names=names,
        units=(u.deg, u.deg, u.kpc, u.mas / u.yr, u.mas / u.yr),
    )
    samples["distance"] = Distance(parallax=samples["parallax"]).to("kpc")
    samples_df = samples.to_pandas()

    # Plot the distributions
    fig, axs = plt.subplots(dims, dims, figsize=(3 * dims, 3 * dims))

    lims = (
        (data_sim["ra"].value.min(), data_sim["ra"].value.max()),  # ra
        (32, 42),  # dec
        (data_sim["parallax"].value.min(), data_sim["parallax"].value.max()),
        (-20, 20),  # pmra
        (-30, 20),  # pmdec
    )
    c = jnp.exp(normalized_flow.log_prob(Xp))

    for i, j in itertools.product(range(dims), repeat=2):
        ax = axs[i, j]
        if i < j:
            ax.axis("off")  # Upper triangle
            continue
        if i == j:
            _, bins, _ = ax.hist(X[:, i], density=True, alpha=0.5, label="Data")
            ax.hist(
                samples_df[names[i]],
                bins=bins,
                density=True,
                alpha=0.5,
                label="Synthetic",
            )
            ax.set_title(names[i])
        else:
            sum_axes = tuple(k for k in range(dims) if k not in [i, j])
            sel = tuple(slice(None) if k in [i, j] else 0 for k in range(dims))
            ax.scatter(X[:, j], X[:, i], s=3, alpha=0.01, c=c, cmap="viridis_r")
            ax.set(xlim=lims[j], ylim=lims[i])

        # Set labels on left and bottom edges
        if i == dims - 1:
            ax.set_xlabel(f"{names[j]}")
        if j == 0:
            ax.set_ylabel(f"{names[i]}")

    fig.tight_layout()
    fig.savefig(diagnostic_plots / "astro_data_flow_samples.png")
