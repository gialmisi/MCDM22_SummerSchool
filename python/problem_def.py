import numpy as np
from desdeo_tools.scalarization.ASF import StomASF

asf = StomASF(np.array([0.01, 100.0, -0.10]))

# w_i
index_values = np.array(
    [
        0.1609,
        0.1164,
        0.1026,
        0.1516,
        0.0939,
        0.1320,
        0.0687,
        0.0930,
        0.2116,
        0.2529,
        0.0868,
        0.0828,
        0.0975,
        0.8177,
        0.4115,
        0.3795,
        0.0710,
        0.0427,
        0.1043,
        0.0997,
        0.1698,
        0.2531,
    ]
)

# d_i^j
distance_matrix = np.array(
    [
        [16.16, 24.08, 24.32, 21.12],
        [19, 26.47, 27.24, 17.33],
        [25.29, 32.49, 33.42, 12.25],
        [0, 7.93, 8.31, 36.12],
        [3.07, 6.44, 7.56, 37.37],
        [1.22, 7.51, 8.19, 36.29],
        [2.8, 10.31, 10.95, 33.5],
        [2.87, 5.07, 5.67, 38.8],
        [3.8, 8.01, 7.41, 38.16],
        [12.35, 4.52, 4.35, 48.27],
        [11.11, 3.48, 2.97, 47.14],
        [21.99, 22.02, 24.07, 39.86],
        [8.82, 3.3, 5.36, 43.31],
        [7.93, 0, 2.07, 43.75],
        [9.34, 2.25, 1.11, 45.43],
        [8.31, 2.07, 0, 44.43],
        [7.31, 2.44, 1.11, 43.43],
        [7.55, 0.75, 1.53, 43.52],
        [11.13, 18.41, 19.26, 25.4],
        [17.49, 23.44, 24.76, 23.21],
        [11.03, 18.93, 19.28, 25.43],
        [36.12, 43.75, 44.43, 0],
    ]
)

work_load_low = 0.8
work_load_high = 1.2

# a_tilde
original_assignments = np.zeros((22, 4))
original_assignments[[3, 4, 5, 6, 7, 14], 0] = 1
original_assignments[[9, 10, 11, 12, 13], 1] = 1
original_assignments[[8, 15, 16, 17], 2] = 1
original_assignments[[0, 1, 2, 18, 19, 20, 21], 3] = 1


def disruption(a: np.ndarray, a_tilde: np.ndarray = original_assignments, w: np.ndarray = index_values) -> float:
    sum_1 = np.sum(np.heaviside(a - a_tilde, 0) * w[:, None], axis=0)
    sum_2 = np.sum(sum_1)
    return sum_2


def disruption_all(a: np.ndarray, a_tilde: np.ndarray = original_assignments, w: np.ndarray = index_values) -> float:
    a = to_bivariable(a)
    sum_1 = np.sum(np.heaviside(a - a_tilde, 0) * w[:, None], axis=0)
    return sum_1


def distance(a: np.ndarray, d: np.ndarray = distance_matrix) -> float:
    sum_1 = np.sum(a * d, axis=0)
    sum_2 = np.sum(sum_1)
    return sum_2


def work_load_individual(a: np.ndarray, w: np.ndarray = index_values) -> np.ndarray:
    sum_1 = np.sum(a * w[:, None], axis=0)
    return sum_1


def work_load(a: np.ndarray) -> float:
    return np.sum(work_load_individual(a))


def to_bivariable(a: np.ndarray) -> np.ndarray:
    bi_a = np.zeros((22, 4))
    indices = zip(range(len(a)), a.astype(int))
    rows, cols = zip(*indices)
    bi_a[rows, cols] = 1

    return bi_a


def scalarized_objective_fun(a_flat: np.ndarray, asf=asf, ref_point=np.array([3.5, 120])) -> float:
    fs = objective_f(a_flat)

    return asf(np.atleast_2d(fs), ref_point)


def objective_f(a_flat: np.ndarray) -> np.ndarray:
    a = to_bivariable(a_flat)
    f_1 = disruption(a)
    f_2 = distance(a)
    ws = work_load_individual(a)
    f_3 = np.min(ws)

    return np.array([f_1, f_2, -f_3])


def work_loads(a_flat: np.ndarray) -> np.ndarray:
    a = to_bivariable(a_flat)
    ws = work_load_individual(a)
    return ws


a_toy = np.zeros((22, 4))

a_toy[[9, 10, 11, 12, 13], 0] = 1
a_toy[[3, 4, 5, 6, 7, 14], 1] = 1
a_toy[[8, 15, 16, 18], 2] = 1
a_toy[[0, 1, 2, 17, 19, 20, 21], 3] = 1

# print(disruption(a_toy))
# print(disruption(original_assignments))

# print(distance(original_assignments))

# print(simple_objective_fun(np.zeros((22, 4))))

# bi = to_bivariable(np.ones(22) * 2)
# print(simple_objective_fun(flat))
