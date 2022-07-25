from desdeo_tools.utilities import fast_non_dominated_sort_indices
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

data = np.loadtxt("./results_3d.dat", delimiter=" ")
fronts = fast_non_dominated_sort_indices(data[:, :3])
non_dom = data[fronts[0]]

"""
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")
ax.scatter(non_dom[:, 0], non_dom[:, 1], non_dom[:, 2])

plt.show()
"""

np.savetxt("3d_non_dom.dat", non_dom, delimiter=" ", header="# disturbance distance min workload vars")
