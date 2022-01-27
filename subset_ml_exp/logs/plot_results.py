import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn")

gp_u_rmse_vals = [
    161.71086598,
    28.85840613,
    61.62322635,
    89.01224779,
    196.37292656,
    137.8028809,
    138.24521619,
    74.17284679,
    25.34343525,
    21.49195802,
    47.98723338,
]

gp_r_rmse_vals = [
    161.71086598,
    28.87517229,
    61.64195233,
    89.16821283,
    196.67367176,
    137.78413805,
    138.54287059,
    74.18713076,
    25.49369658,
    21.78349681,
    48.04409536,
]

ax = plt.subplot(111)

ax.plot(gp_u_rmse_vals, label="GP + Uncertainity")
ax.plot(gp_r_rmse_vals, label="GP + Random")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax.grid(color="w")
ax.set_xlabel("Time-stamps (1 time-stamp = 24 hrs)")
ax.set_ylabel("RMSE (lower is better)")
ax.set_xticks(np.arange(0, 11))
ax.set_title(
    "Beijing Dataset, RMSE after installing each sensor, LHS test center allocation"
)
plt.savefig(f"./beijing_rmse.png")
