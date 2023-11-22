"""Setup."""

import shutil
import sys

import asdf
import galstreams
import matplotlib as mpl
import matplotlib.path as mpath
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from matplotlib import pyplot as plt
from showyourwork.paths import user as user_paths

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split


from scripts.pal5.frames import pal5_frame as frame

##############################################################################
# PARAMETERS

SAVE_LOC = paths.data / "pal5" / "masks.asdf"

try:
    snkmk = snakemake.params
except NameError:
    snkmk = {
        "load_from_static": True,
        "save_to_static": False,
        "diagnostic_plots": True,
    }


if snkmk["load_from_static"] and (paths.static / "pal5").exists():
    shutil.copyfile(paths.static / "pal5" / "masks.asdf", SAVE_LOC)

    sys.exit(0)


##############################################################################
# Read tables

# Gaia Data
table = QTable.read(paths.data / "pal5" / "gaia_ps1_xm.asdf")

c_pal5_icrs = SkyCoord(
    ra=table["ra"], dec=table["dec"], pm_ra_cosdec=table["pmra"], pm_dec=table["pmdec"]
)
c_pal5 = c_pal5_icrs.transform_to(frame)


##############################################################################
# Masks

masks_table = QTable()

# =============================================================================
# Off-stream selection
# Applying this mask to the data table will remove the off-stream region.

footprint = np.load(paths.data / "pal5" / "footprint.npz")["footprint"]

masks_table["off_stream"] = mpath.Path(footprint.T, readonly=True).contains_points(
    np.c_[table["phi1"].to_value("deg"), table["phi2"].to_value("deg")]
)

# =============================================================================
# Pal5 Cluster
# Applying this mask to the data table will remove the Pal5 cluster stars.

P5 = SkyCoord.from_name("palomar 5")
masks_table["Pal5"] = ~(P5.separation(c_pal5) < 0.5 * u.deg)


# =============================================================================
# M5
# Applying this mask to the data table will remove the M5 stars.

M5 = SkyCoord.from_name("messier 5")
masks_table["M5"] = ~(M5.separation(c_pal5_icrs) < 0.8 * u.deg)


# =============================================================================
# Other Stuff

masks_table["things"] = ~(
    (c_pal5.pm_phi1_cosphi2.value > -1)
    & (c_pal5.pm_phi1_cosphi2.value < 2)
    & (c_pal5.pm_phi2.value > -1)
    & (c_pal5.pm_phi2.value < 1)
)

masks_table["things2"] = ~(
    (c_pal5.phi1.value > 12.7)
    & (c_pal5.phi1.value < 13)
    & (c_pal5.phi2.value > -1.7)
    & (c_pal5.phi2.value < -1.3)
)


# =============================================================================
# Proper motion
# Applying this mask to the data table will remove the stars outside the
# proper motion box.

pm_edges = QTable.read(paths.data / "pal5" / "pm_edges.ecsv")
pm_edges.add_index("label", unique=True)

pm_med = pm_edges.loc["med_icrs"]
masks_table["pm_med_icrs"] = (
    (table["pmra"] > pm_med["pm_ra_min"])
    & (table["pmra"] < pm_med["pm_ra_max"])
    & (table["pmdec"] > pm_med["pm_dec_min"])
    & (table["pmdec"] < pm_med["pm_dec_max"])
)


# =============================================================================
# Photometric selection
# Applying this mask to the data table will remove the stars outside the
# photometric box.

with asdf.open(
    paths.data / "pal5" / "isochrone.asdf", lazy_load=False, copy_arrays=True
) as af:
    iso_15 = af["isochrone_15"]

masks_table["phot_15"] = mpath.Path(iso_15, readonly=True).contains_points(
    np.c_[table["g0"], table["r0"]]
)


# =============================================================================
# Phi1 Subset

masks_table["phi1_subset"] = np.abs(table["phi1"]) <= 15 * u.deg


# =============================================================================
# Save

# Try removing the file first. ASDF can be weird about overwriting.
SAVE_LOC.unlink(missing_ok=True)
# Save
masks_table.write(SAVE_LOC)

if snkmk["save_to_static"]:
    shutil.copyfile(SAVE_LOC, paths.static / "pal5" / "masks.asdf")


###############################################################################
# Diagnostic plot

if not snkmk["diagnostic_plots"]:
    sys.exit(0)

fig = plt.figure(figsize=(15, 7))
gs = mpl.gridspec.GridSpec(2, 2, width_ratios=[1, 3])
ax00 = fig.add_subplot(
    gs[0, 0], xlabel=r"$\mu_{\alpha}^*$ [mas/yr]", ylabel=r"$\mu_\delta$ [mas/yr]"
)
ax10 = fig.add_subplot(gs[1, 0], xlabel=r"$g-r$ [mas/yr]", ylabel=r"$g$ [mas/yr]")
ax01 = fig.add_subplot(gs[0, 1], xlabel=r"$\phi_1$ [deg]", ylabel=r"$\phi_2$ [deg]")
ax11 = fig.add_subplot(
    gs[1, 1], xlabel=r"$\phi_1$ [deg]", ylabel=r"$\mu_{\phi_1}^*$ [mas/yr]"
)

# Initial mask getting rid of other clusters
_mask = masks_table["M5"] & masks_table["things"] & masks_table["things2"]
# Full mask, including pm & photo selection
_mask_full = _mask & masks_table["pm_med_icrs"] & masks_table["phot_15"]

allstreams = galstreams.MWStreams(implement_Off=True)
pal5PW19 = allstreams["Pal5-PW19"].track.transform_to(frame)
pal5I21 = allstreams["Pal5-I21"].track.transform_to(frame)

# -----------------------------------------------
# PM

ax00.hist2d(
    table["pmra"][_mask].value,
    table["pmdec"][_mask].value,
    bins=(np.linspace(-10, 10, 96), np.linspace(-10, 10, 96)),
    cmap="Greys",
    norm=mpl.colors.LogNorm(),
)

ax00.plot(
    pal5PW19.icrs.pm_ra_cosdec.value,
    pal5PW19.icrs.pm_dec.value,
    c="tab:blue",
    label="Pal 5 - PW19",
)
ax00.plot(
    pal5I21.icrs.pm_ra_cosdec.value,
    pal5I21.icrs.pm_dec.value,
    c="tab:blue",
    label="Pal 5 - I21",
)

row = pm_edges.loc["med_icrs"]
rec = mpl.patches.Rectangle(
    (row["pm_ra_min"].value, row["pm_dec_min"].value),
    row["pm_ra_max"].value - row["pm_ra_min"].value,
    row["pm_dec_max"].value - row["pm_dec_min"].value,
    color="tab:red",
    label="Selection",
)
rec.set_facecolor((*rec.get_facecolor()[:-1], 0.05))
ax00.add_patch(rec)

# -----------------------------------------------
# CMD

ax10.hist2d(
    table["g0"][_mask].value - table["r0"][_mask].value,
    table["g0"][_mask].value,
    bins=(np.linspace(-0.5, 1.5, 96), np.linspace(12, 23, 96)),
    cmap="Greys",
    norm=mpl.colors.LogNorm(),
)
ax10.plot(
    iso_15[:, 0] - iso_15[:, 1],
    iso_15[:, 0],
    c="r",
    lw=1,
    label="Isochrone buffer",
)
ax10.set_ylim(22, 12)

# -----------------------------------------------
# Applying to phi1, phi2

ax01.plot(
    c_pal5.phi1[_mask_full & masks_table["Pal5"]],
    c_pal5.phi2[_mask_full & masks_table["Pal5"]],
    c="black",
    marker=",",
    linestyle="none",
    alpha=1,
)
ax01.plot(
    c_pal5.phi1[_mask_full & ~masks_table["Pal5"]],
    c_pal5.phi2[_mask_full & ~masks_table["Pal5"]],
    c="green",
    marker=",",
    linestyle="none",
    alpha=1,
)

ax01.plot(
    pal5PW19.phi1.degree,
    pal5PW19.phi2.degree,
    c="tab:blue",
    label="Pal 5",
)

# -----------------------------------------------
# Applying to phi1, pm_phi1

ax11.plot(
    c_pal5.phi1[_mask_full],
    table["pm_phi1"][_mask_full],
    c="black",
    marker=",",
    linestyle="none",
    alpha=1,
)

fig.tight_layout()
fig.savefig(paths.scripts / "pal5" / "_diagnostics" / "masks.png", dpi=300)
