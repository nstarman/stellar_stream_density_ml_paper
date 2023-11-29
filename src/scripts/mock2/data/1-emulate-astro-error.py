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
import seaborn as sns
from astropy.coordinates import Distance
from astropy.table import QTable
from emulation_model import (
    make_astro_data_model,
    make_astro_error_model,
    make_astro_error_preprocess,
)
from matmul import MatMul
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

diagnostic_plots = paths.scripts / "mock2" / "_diagnostics" / "astro"
diagnostic_plots.mkdir(parents=True, exist_ok=True)

##############################################################################
# Errors

names = ("ra", "dec", "parallax", "pmra", "pmdec")
dims = len(names)

# ===================================================================
# Load the error data table

data_real = QTable.read(paths.data / "mock2" / "gd1_query_real.fits")
for col in data_real.colnames:
    data_real[col] = data_real[col].astype("float64")

data_real_df = data_real.to_pandas()

error_names = tuple(f"{name}_error" for name in names)

dX = structured_to_unstructured(np.array(data_real[error_names]))
selXerr = jnp.all(jnp.isfinite(dX), -1)
dX = dX[selXerr]


# ===================================================================
# Preprocess the error data

eps = 1e-1  # for numerical stability

if snkmk["load_from_static"]:
    preprocess_errors = make_astro_error_preprocess(
        paths.static / "mock2" / "astro_errors_preprocess.eqx"
    )

else:
    preprocess_errors = fjxb.Chain(
        [
            fjxb.Scale(
                (2 - 2 * eps) / jnp.array([jnp.max(x) - jnp.min(x) for x in dX.T])
            ),
            fjxb.Loc(-jnp.ones(dims) + eps),
            fjxb.Invert(fjxb.LeakyTanh(max_val=10, shape=(dims,))),
            MatMul(jnp.linalg.inv(jnp.cov(dX.T))),
            fjxb.Scale(jnp.ones(dims) / 100),
        ]
    )

    if snkmk["save_to_static"]:
        eqx.tree_serialise_leaves(
            paths.static / "mock2" / "astro_errors_preprocess.eqx", preprocess_errors
        )

# Save the flow info
eqx.tree_serialise_leaves(
    paths.data / "mock2" / "astro_errors_preprocess.eqx", preprocess_errors
)


# Transform the data
dXp = jax.vmap(preprocess_errors.transform)(dX)


if snkmk["diagnostic_plots"]:
    # -------------------------------------------
    # Plot the transformed data

    fig, axs = plt.subplots(1, dims, figsize=(15, 4))
    for i, k in enumerate(error_names):
        axs[i].scatter(data_real["distance"][selXerr].value, dX[:, i], s=3)
        axs[i].axhline(0, c="k", ls="--")
        axs[i].set(xlabel="Distance [kpc]", title=f"{k} [{data_real[k].unit}]")
    fig.savefig(diagnostic_plots / "astro_errors_preprocess_distances.png")

    # -------------------------------------------
    # Plot the distributions

    fig, axs = plt.subplots(dims, dims, figsize=(3 * dims, 3 * dims))

    grid_vecs = (
        jnp.linspace(-2, 2, 10),
        jnp.linspace(-1.5, 1, 10),
        jnp.linspace(-2.5, 2.5, 10),
        jnp.linspace(-1, 2, 10),
        jnp.linspace(-1.5, 1.5, 10),
    )

    for i, j in itertools.product(range(dims), repeat=2):
        ax = axs[i, j]
        if i < j:
            ax.axis("off")  # Upper triangle
            continue
        if i == j:
            ax.hist(dXp[:, i], density=True, alpha=0.5, label="Data", bins=30)
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

    fig.savefig(diagnostic_plots / "astro_errors_preprocess_distributions.png")


# ===================================================================
# Train the error model

if snkmk["load_from_static"]:
    error_flow = make_astro_error_model(paths.static / "mock2" / "astro_error_flow.eqx")
    normalized_error_flow = error_flow.base_dist

else:
    key, subkey = jr.split(jr.PRNGKey(37))

    untrained_error_flow = flowjax.flows.masked_autoregressive_flow(
        key=subkey,
        base_dist=fjdist.Normal(jnp.zeros(dXp.shape[1]), scale=1),
        transformer=fjxb.Affine(),
        invert=False,
    )

    key, subkey = jr.split(key)

    normalized_error_flow, losses = flowjax.train.fit_to_data(
        key=subkey,
        dist=untrained_error_flow,
        x=dXp,
        learning_rate=5e-4,
        max_patience=100,
        max_epochs=2_000,
        return_best=True,
        batch_size=int(0.075 * len(dXp)),  # 7.5% of the data
    )
    error_flow = fjdist.Transformed(
        normalized_error_flow, fjxb.Invert(preprocess_errors)
    )

    if snkmk["save_to_static"]:
        eqx.tree_serialise_leaves(
            paths.static / "mock2" / "astro_error_flow.eqx", error_flow
        )

# Save the flow info
eqx.tree_serialise_leaves(paths.data / "mock2" / "astro_error_flow.eqx", error_flow)


# -------------------------------------------
# Plot the distributions

if snkmk["diagnostic_plots"]:
    resolution = 20  # 20^5 is big. Don't want to go much higher.
    grid_vecs = (
        jnp.linspace(-1.5, 0, resolution),
        jnp.linspace(-1.2, -0.2, resolution),
        jnp.linspace(-0.2, 1, resolution),
        jnp.linspace(0.2, 0.8, resolution),
        jnp.linspace(-0.2, 0.5, resolution),
    )
    meshgrids = jnp.meshgrid(*grid_vecs, indexing="ij")
    xyinput = jnp.column_stack(tuple(grid.reshape(-1, 1) for grid in meshgrids))
    outgrid = jnp.exp(normalized_error_flow.log_prob(xyinput)).reshape(
        meshgrids[0].shape
    )
    c = normalized_error_flow.log_prob(dXp)

    fig, axs = plt.subplots(dims, dims, figsize=(3 * dims, 3 * dims))

    for i, j in itertools.product(range(dims), repeat=2):
        ax = axs[i, j]
        if i < j:
            ax.axis("off")  # Upper triangle
            continue
        if i == j:
            ax.hist(dXp[:, i], bins=30, density=True, alpha=0.5, label="Data")
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
    fig.savefig(diagnostic_plots / "astro_error_flow_distributions.png")


# -------------------------------------------
# Plot samples

if snkmk["diagnostic_plots"]:
    key = jr.PRNGKey(56)
    data_flow = make_astro_data_model(paths.static / "mock2" / "astro_data_flow.eqx")
    samples = data_flow.sample(key, (30_000,))

    samples = QTable(
        np.array(samples),
        names=names,
        units=(u.deg, u.deg, u.kpc, u.mas / u.yr, u.mas / u.yr),
    )
    samples["distance"] = Distance(parallax=samples["parallax"]).to("kpc")

    error_samples = error_flow.sample(key, (len(samples),))

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
    samples_df[:5]

    grid_vecs = (
        (data_real["ra_error"].value.min(), data_real["ra_error"].value.max()),  # ra
        (0, 5),  # dec
        (
            data_real["parallax_error"].value.min(),
            data_real["parallax_error"].value.max(),
        ),  # parallax
        (0, 5),  # pmra
        (0, 5),  # pmdec
    )

    fig, axs = plt.subplots(dims, dims, figsize=(3 * dims, 3 * dims))

    c = jnp.exp(normalized_error_flow.log_prob(dXp))

    for i, j in itertools.product(range(dims), repeat=2):
        ax = axs[i, j]
        if i < j:
            ax.axis("off")  # Upper triangle
            continue
        # Sum over axes not being plotted
        if i == j:
            _, bins, _ = ax.hist(dX[:, i], density=True, alpha=0.5, label="Data")
            ax.hist(
                samples_df[error_names[i]],
                bins=bins,
                density=True,
                alpha=0.5,
                label="Synthetic",
            )
            ax.set_title(error_names[i])
        else:
            # Real data
            ax.scatter(dX[:, j], dX[:, i], s=3, alpha=0.01, c=c, cmap="viridis_r")
            # Synthetic data
            ax.scatter(
                samples_df[error_names[j]],
                samples_df[error_names[i]],
                s=3,
                alpha=0.01,
                c="gray",
            )

            ax.set(xlim=grid_vecs[j], ylim=grid_vecs[i])

        # Set labels on left and bottom edges
        if i == dims - 1:
            ax.set_xlabel(f"{error_names[j]}")
        if j == 0:
            ax.set_ylabel(f"{error_names[i]}")

    fig.tight_layout()
    fig.savefig(diagnostic_plots / "astro_error_flow_samples.png")


# -------------------------------------------
# Plot All Parallax Dependencies

if snkmk["diagnostic_plots"]:
    fig = plt.figure(figsize=(5 * dims, 5))
    fig.suptitle("Distance Dependence")

    # Define GridSpec: 2 rows, 6 columns
    gs = plt.GridSpec(
        3,
        3 * dims,
        figure=fig,
        height_ratios=[1, 4, 4],
        width_ratios=[4, 1, 1.1] * dims,
        hspace=0,
        wspace=0,
    )

    # Main plots (middle row, cols 0-1, 3-4, 6-7)
    _axs_main = [fig.add_subplot(gs[2, 3 * i]) for i in range(dims)]
    _axs_error = [
        fig.add_subplot(gs[1, 3 * i], sharex=ax) for i, ax in enumerate(_axs_main)
    ]
    axs = np.array([_axs_error, _axs_main])
    axs_top = np.array(
        [fig.add_subplot(gs[0, 3 * i], sharex=axs[1, i]) for i in range(dims)]
    )
    axs_right = np.array(
        [
            [fig.add_subplot(gs[1, 1 + 3 * i], sharey=axs[0, i]) for i in range(dims)],
            [fig.add_subplot(gs[2, 1 + 3 * i], sharey=axs[1, i]) for i in range(dims)],
        ]
    )
    # Hide labels and ticks for marginal plots to avoid overlap
    for ax, ax_top, ax_right in zip(axs.T, axs_top.flat, axs_right.T, strict=True):
        plt.setp(ax[0].get_xticklabels(), visible=False)
        plt.setp(ax_top.get_xticklabels(), visible=False)
        plt.setp(ax_right[0].get_yticklabels(), visible=False)
        plt.setp(ax_right[1].get_yticklabels(), visible=False)

    # RA
    for i, component in enumerate(names):
        for label, data, color in zip(
            ("real", "resample"),
            (data_real_df, samples_df),
            ("red", "gray"),
            strict=True
        ):
            kw = {"alpha": 0.33, "color": color}
            sns.kdeplot(data=data, x="parallax", ax=axs_top[i], fill=True, **kw)

            sns.kdeplot(
                data=data, y=f"{component}_error", ax=axs_right[0, i], fill=True, **kw
            )
            sns.scatterplot(
                data=data, x="parallax", y=f"{component}_error", s=3, ax=axs[0, i], **kw
            )

            sns.scatterplot(
                data=data,
                x="parallax",
                y=f"{component}",
                s=3,
                ax=axs[1, i],
                label=label,
                **kw,
            )
            sns.kdeplot(
                data=data, y=f"{component}", ax=axs_right[1, i], fill=True, **kw
            )
        axs[1, i].legend()

    fig.tight_layout()
    fig.savefig(diagnostic_plots / "astro_distance_dependencies.png")
