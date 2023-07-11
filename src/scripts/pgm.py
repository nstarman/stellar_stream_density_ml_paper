"""Plot Stream PGM."""

import sys

import daft
import matplotlib.pyplot as plt
from showyourwork.paths import user as user_paths

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.parent.as_posix())
# isort: split


# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")

# Colors.
w_color = {"ec": "tab:blue"}
m_color = {"ec": "#f89406"}

# =============================================================================
# Stream Model

# Instantiate the PGM.
pgm = daft.PGM()

# Astrometric Nodes
pgm.add_node(
    "stream_sigma_w,obs", r"$\Sigma_n^{(w)}$", 2, 2, fixed=True, plot_params=w_color
)
pgm.add_node(
    "stream_w,obs", r"$w_n^{\rm obs}$", 3, 2, observed=True, plot_params=w_color
)

pgm.add_node(
    "stream_mu_w,model", r"$\mu^{(w)}}$", 2, 3, observed=False, plot_params=w_color
)
pgm.add_node(
    "stream_sigma_w,model", r"$\Sigma^{(w)}$", 3, 3, observed=False, plot_params=w_color
)

#   Add in the edges.
pgm.add_edge("stream_sigma_w,obs", "stream_w,obs")
pgm.add_edge("stream_mu_w,model", "stream_w,obs")
pgm.add_edge("stream_sigma_w,model", "stream_w,obs")

# Photometric Nodes
pgm.add_node(
    "stream_sigma_m,obs", r"$\Sigma_n^{(m)}$", 5, 2, fixed=True, plot_params=m_color
)
pgm.add_node(
    "stream_m,obs", r"$m_n^{\rm obs}$", 4, 2, observed=True, plot_params=m_color
)

pgm.add_node(
    "stream_mu_m,model", r"$\mu^{(m)}$", 4, 3, observed=False, plot_params=m_color
)
pgm.add_node(
    "stream_sigma_m,model", r"$\Sigma^{(m)}$", 5, 3, observed=False, plot_params=m_color
)

#   Add in the edges.
pgm.add_edge("stream_sigma_m,obs", "stream_m,obs")
pgm.add_edge("stream_mu_m,model", "stream_m,obs")
pgm.add_edge("stream_sigma_m,model", "stream_m,obs")

# Full Data Node
pgm.add_node("stream_x,obs", r"$x_n^{obs}$", 3, 1, alternate=True)
pgm.add_edge("stream_w,obs", "stream_x,obs", directed=False)
pgm.add_edge("stream_m,obs", "stream_x,obs", directed=False)

# Mixture probability
pgm.add_node("stream_mixture_coefficient", r"$f_q$", 1, 3)
pgm.add_node("stream_mixture_index", r"$q_n$", 1, 1)
pgm.add_edge("stream_mixture_coefficient", "stream_mixture_index")
pgm.add_edge("stream_mixture_index", "stream_x,obs")

# Phi1 Node
_ntwk_kw = {"alpha": 0.5, "linestyle": "--", "zorder": -100}
pgm.add_node("stream_phi1", r"${\phi_1}_n$", 3, 4.1, observed=True, plot_params=w_color)
pgm.add_edge("stream_phi1", "stream_mixture_coefficient", plot_params=_ntwk_kw)
pgm.add_edge("stream_phi1", "stream_mu_w,model", plot_params=_ntwk_kw)
pgm.add_edge("stream_phi1", "stream_sigma_w,model", plot_params=_ntwk_kw)
pgm.add_edge("stream_phi1", "stream_mu_m,model", plot_params=_ntwk_kw)
pgm.add_edge("stream_phi1", "stream_sigma_m,model", plot_params=_ntwk_kw)


# And a plate.
pgm.add_plate(
    [0.5, 0.5, 5, 4],
    label=r"",
    shift=-0.1,
    rect_params={"linestyle": "--", "alpha": 0.5},
)
pgm.add_plate([0.5, 0.5, 5, 2], label=r"$n = 1, \cdots, N$", shift=-0.1)
pgm.add_plate(
    [1.5, 2.75, 4, 1], label=r"$q = 1, \cdots, Q$", shift=-0.1, position="top right"
)

# =============================================================================
# Background Model

base_shift = 7

# Astrometric Nodes
pgm.add_node(
    "sigma_w,obs",
    r"$\Sigma_n^{(w)}$",
    base_shift + 2,
    2,
    fixed=True,
    plot_params=w_color,
)
pgm.add_node(
    "w,obs", r"$w_n^{\rm obs}$", base_shift + 3, 2, observed=True, plot_params=w_color
)

pgm.add_node(
    "theta_w,model",
    r"$\theta^{(w)}}$",
    base_shift + 3,
    3,
    observed=False,
    plot_params=w_color,
)

#   Add in the edges.
pgm.add_edge("sigma_w,obs", "w,obs")
pgm.add_edge("theta_w,model", "w,obs")

# Photometric Nodes
pgm.add_node(
    "sigma_m,obs",
    r"$\Sigma_n^{(m)}$",
    base_shift + 5,
    2,
    fixed=True,
    plot_params=m_color,
)
pgm.add_node(
    "m,obs", r"$m_n^{\rm obs}$", base_shift + 4, 2, observed=True, plot_params=m_color
)

pgm.add_node(
    "theta_m,model",
    r"$\theta^{(m)}$",
    base_shift + 4,
    3,
    fixed=True,
    plot_params=m_color,
)

#   Add in the edges.
pgm.add_edge("sigma_m,obs", "m,obs")
pgm.add_edge("theta_m,model", "m,obs")

# Full Data Node
pgm.add_node("x,obs", r"$x_n^{obs}$", base_shift + 3, 1, alternate=True)
pgm.add_edge("w,obs", "x,obs", directed=False)
pgm.add_edge("m,obs", "x,obs", directed=False)

# Mixture probability
pgm.add_node("mixture_coefficient", r"$f_q$", base_shift + 1, 3)
pgm.add_node("mixture_index", r"$q_n$", base_shift + 1, 1)
pgm.add_edge("mixture_coefficient", "mixture_index")
pgm.add_edge("mixture_index", "x,obs")

# Phi1 Node
_ntwk_kw = {"alpha": 0.5, "linestyle": "--", "zorder": -100}
pgm.add_node(
    "phi1", r"${\phi_1}_n$", base_shift + 3, 4.1, observed=True, plot_params=w_color
)
pgm.add_edge("phi1", "mixture_coefficient", plot_params=_ntwk_kw)
pgm.add_edge("phi1", "theta_w,model", plot_params=_ntwk_kw)
pgm.add_edge("phi1", "theta_m,model", plot_params=_ntwk_kw)


# And a plate.
pgm.add_plate(
    [base_shift + 0.5, 0.5, 5, 4],
    label=r"",
    shift=-0.1,
    rect_params={"linestyle": "--", "alpha": 0.5},
)
pgm.add_plate([base_shift + 0.5, 0.5, 5, 2], label=r"$n = 1, \cdots, N$", shift=-0.1)
pgm.add_plate(
    [base_shift + 2.5, 2.75, 2, 1],
    label=r"$q = 1, \cdots, Q$",
    shift=-0.1,
    position="top right",
)


ax2 = pgm.render()


# =============================================================================

ax2.figure.savefig(paths.figures / "pgm.pdf")
