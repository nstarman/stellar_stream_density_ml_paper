"""Train photometry background flow."""

import sys

import daft
import matplotlib.pyplot as plt
from showyourwork.paths import user as user_paths

paths = user_paths()

# Add the parent directory to the path
sys.path.append(paths.scripts.as_posix())
# isort: split


# Matplotlib style
plt.style.use(paths.scripts / "paper.mplstyle")


# Instantiate the PGM.
# Instantiate the PGM.
pgm = daft.PGM()

# Colors.
w_color = {"ec": "tab:blue"}
m_color = {"ec": "#f89406"}

# Astrometric Nodes
pgm.add_node(
    "sigma_w,obs", r"$\Sigma_n^{(w)}$", 2, 2, observed=True, plot_params=w_color
)
pgm.add_node("w,obs", r"$w_n^{\rm obs}$", 3, 2, observed=True, plot_params=w_color)

pgm.add_node(
    "theta_w,model", r"$\theta^{(w)}}$", 3, 3, observed=False, plot_params=w_color
)

#   Add in the edges.
pgm.add_edge("sigma_w,obs", "w,obs")
pgm.add_edge("theta_w,model", "w,obs")

# Photometric Nodes
pgm.add_node(
    "sigma_m,obs", r"$\Sigma_n^{(m)}$", 5, 2, observed=True, plot_params=m_color
)
pgm.add_node("m,obs", r"$m_n^{\rm obs}$", 4, 2, observed=True, plot_params=m_color)

pgm.add_node("theta_m,model", r"$\theta^{(m)}$", 4, 3, fixed=True, plot_params=m_color)

#   Add in the edges.
pgm.add_edge("sigma_m,obs", "m,obs")
pgm.add_edge("theta_m,model", "m,obs")

# Full Data Node
pgm.add_node("x,obs", r"$x_n^{obs}$", 3, 1, observed=True)
pgm.add_edge("w,obs", "x,obs")
pgm.add_edge("m,obs", "x,obs")

# Mixture probability
pgm.add_node("mixture_coefficient", r"$\alpha_q$", 1, 3)
pgm.add_node("mixture_index", r"$q_n$", 1, 1)
pgm.add_edge("mixture_coefficient", "mixture_index")
pgm.add_edge("mixture_index", "x,obs")

# Phi1 Node
_ntwk_kw = {"alpha": 0.5, "linestyle": "--", "zorder": -100}
pgm.add_node("phi1", r"${\phi_1}_n$", 3, 4.1, observed=True, plot_params=w_color)
pgm.add_edge("phi1", "mixture_coefficient", plot_params=_ntwk_kw)
pgm.add_edge("phi1", "theta_w,model", plot_params=_ntwk_kw)
pgm.add_edge("phi1", "theta_m,model", plot_params=_ntwk_kw)


# And a plate.
pgm.add_plate(
    [0.5, 0.5, 5, 4],
    label=r"",
    shift=-0.1,
    rect_params={"linestyle": "--", "alpha": 0.5},
)
pgm.add_plate([0.5, 0.5, 5, 2], label=r"$n = 1, \cdots, N$", shift=-0.1)
pgm.add_plate(
    [2.5, 2.75, 2, 1], label=r"$q = 1, \cdots, Q$", shift=-0.1, position="top right"
)

# pgm.add_edge("phi1", "network")

# Render and save.
axes = pgm.render()

axes.figure.savefig(paths.figures / "gd1" / "pgm_background.pdf")
