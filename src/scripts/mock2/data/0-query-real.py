"""Query the fields."""

import itertools
import shutil
import sys

import astropy.units as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.coordinates import Angle, Distance, SkyCoord
from astropy.table import QTable
from astroquery.gaia import Gaia

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

# Ensure the directories exist
(paths.data / "mock2").mkdir(exist_ok=True)

diagnostic_plots = paths.scripts / "mock2" / "_diagnostics" / "query"
diagnostic_plots.mkdir(parents=True, exist_ok=True)


##############################################################################

names = ("ra", "dec", "parallax", "pmra", "pmdec", "G", "G-R")
dims = len(names)

if snkmk["load_from_static"]:
    # Load the data from the static directory
    shutil.copyfile(
        paths.static / "mock2" / "gd1_query_real.fits",
        paths.data / "mock2" / "gd1_query_real.fits",
    )

    result = QTable.read(paths.data / "mock2" / "gd1_query_real.fits")
    for col in result.colnames:
        result[col] = result[col].astype("float64")

else:
    # Approximate position of GD-1
    ra = 148.0 * u.deg  # RA in degrees
    dec = 36.7 * u.deg  # Dec in degrees
    coord = SkyCoord(ra=ra, dec=dec, frame="icrs")

    # Define the search radius
    ra_radius = Angle(20, u.deg)
    dec_radius = Angle(5, u.deg)

    # Define the distance range in parsecs
    max_plx = Distance(3 * u.kpc).parallax
    min_plx = Distance(15 * u.kpc).parallax

    # Construct the query
    ra_min = (coord.ra - ra_radius).to_value("deg")
    ra_max = (coord.ra + ra_radius).to_value("deg")
    dec_min = (coord.dec - dec_radius).to_value("deg")
    dec_max = (coord.dec + dec_radius).to_value("deg")

    query_real = f"""
    SELECT
        source_id, ra, ra_error, dec, dec_error,
        parallax, parallax_error, parallax_over_error,
        pmra, pmra_error, pmdec, pmdec_error,
        phot_g_mean_mag AS G, phot_g_mean_flux, phot_g_mean_flux_error,
        phot_rp_mean_mag AS R, phot_rp_mean_flux, phot_rp_mean_flux_error
    FROM gaiadr3.gaia_source as gaia
    WHERE
        ra BETWEEN {ra_min} AND {ra_max}
    AND dec BETWEEN {dec_min} AND {dec_max}
    AND parallax BETWEEN {min_plx.value} AND {max_plx.value}
    """

    # Run the query
    # result = do_query(query_real, paths.data / "mock2" / "gd1_query_real.fits")
    saveloc = paths.data / "mock2" / "gd1_query_real.fits"

    job = Gaia.launch_job_async(
        query_real,
        dump_to_file=True,
        output_file=saveloc.as_posix(),
        output_format="fits",
        autorun=True,
    )

    result = QTable(job.get_results())

    for col in result.colnames:
        # Unmask the masked columns
        try:
            v = result[col].unmasked
        except AttributeError:
            pass
        else:
            result[col] = v

        # Cast to float64
        if result[col].dtype != np.dtype("float64"):
            result[col] = result[col].astype("float64")

    # Add distance info
    result["distance"] = Distance(parallax=result["parallax"]).to(u.kpc)

    # Add the approximate magnitude errors
    # $m \propto -2.5 \log_{10} f$
    # Therefore, $\delta m \sim 2.5 / ln(10) * \frac{\delta f}{f}$
    result["G_error"] = (
        1.086
        * u.mag
        * np.abs(result["phot_g_mean_flux_error"] / result["phot_g_mean_flux"])
    )
    result["R_error"] = (
        1.086
        * u.mag
        * np.abs(result["phot_rp_mean_flux_error"] / result["phot_rp_mean_flux"])
    )

    # Add G-R color
    result["G-R"] = result["G"] - result["R"]
    result["G-R_error"] = np.hypot(result["G_error"], result["R_error"])

    # Save the table
    result.write(saveloc.as_posix(), overwrite=True)

    # Save the data to the static directory
    if snkmk["save_to_static"]:
        (paths.static / "mock2").mkdir(exist_ok=True)
        result.write(
            paths.static / "mock2" / "gd1_query_real.fits",
            overwrite=True,
        )


if snkmk["diagnostic_plots"]:
    result_df = result.to_pandas()

    # ---------------------------------------------------------------
    # Plot the distributions
    fig, axs = plt.subplots(dims, dims, figsize=(3 * dims, 3 * dims))

    grid_vecs = (
        jnp.linspace(125, 165, 10),  # ra
        jnp.linspace(32, 42, 10),  # dec
        jnp.linspace(0, 0.35, 10),  # parallax
        jnp.linspace(-50, 50, 10),  # pmra
        jnp.linspace(-50, 20, 10),  # pmdec
        jnp.linspace(10, 22, 10),  # G
        jnp.linspace(-1, 3, 10),  # G-R
    )

    for i, j in itertools.product(range(dims), repeat=2):
        ax = axs[i, j]
        if i < j:
            ax.axis("off")  # Upper triangle
            continue
        if i == j:
            ax.hist(result_df[names[i]], density=True, alpha=0.5, label="Data")
            ax.set_title(names[i])
        else:
            ax.scatter(result[names[j]], result[names[i]], s=2)

            # Contour of the real data
            x = result_df[names[j]]
            y = result_df[names[i]]
            sel = (
                (grid_vecs[j][0] < x)
                & (x < grid_vecs[j][-1])
                & (grid_vecs[i][0] < y)
                & (y < grid_vecs[i][-1])
            )
            xy = np.vstack([x[sel][::50], y[sel][::50]])
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
    fig.savefig(diagnostic_plots / "real_data_distributions.png")

    # ---------------------------------------------------------------
    # Plot distance dependence

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

    for i, y in enumerate(names):
        sns.scatterplot(data=result_df, x="distance", y=f"{y}_error", s=3, ax=axs[0, i])
        sns.kdeplot(data=result_df, y=f"{y}_error", ax=axs_right[0, i], fill=True)

        sns.scatterplot(data=result_df, x="distance", y=f"{y}", s=3, ax=axs[1, i])
        sns.kdeplot(data=result_df, x="distance", ax=axs_top[i], fill=True)
        sns.kdeplot(data=result_df, y=f"{y}", ax=axs_right[1, i], fill=True)

    fig.tight_layout()
    fig.savefig(diagnostic_plots / "real_distance_dependence.png")
