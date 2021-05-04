""" Executions Auxiliary static methods """

import numpy as np
from itertools import product, combinations


def draw_cube(axes) -> None:
    """ Draw a cube passing axes as a parameter """
    r = [-1, 1]
    for s, l in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - l)) == r[1] - r[0]:
            axes.plot3D(*zip(s, l), color="w", zorder=-1)
