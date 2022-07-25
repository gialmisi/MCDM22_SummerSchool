import numpy as np

from problem_def import scalarized_objective_fun, work_loads, disruption_all, objective_f
from utils import to_floats, to_ints
from scipy.optimize import minimize, NonlinearConstraint, differential_evolution

objective = scalarized_objective_fun
N = 22

to_beat = np.zeros(22)
to_beat[[3, 4, 5, 6, 7, 14]] = 0
to_beat[[9, 10, 11, 12, 13]] = 1
to_beat[[8, 15, 16, 17]] = 2
to_beat[[0, 1, 2, 18, 19, 20, 21]] = 3

lower_bounds = np.array([0] * N)
upper_bounds = np.array([3] * N)

bounds_int = np.stack((lower_bounds, upper_bounds)).T
bounds_int[3, :] = [0, 0.49]
bounds_int[13, :] = [0.5, 1.49]
bounds_int[15, :] = [1.5, 2.49]
bounds_int[21, :] = [2.50, 3.0]

# SR central bricks
# SR1 (0, 3)
# SR2 (1, 13)
# SR3 (2, 15)
# SR4 (3, 21)

bounds = to_floats(bounds_int)


con_ws = NonlinearConstraint(lambda x: work_loads(to_ints(x)), 0.8, 1.2)

found = 0
while True:
    ref_point = np.random.uniform(low=np.array([0.1, 100, -0.60]), high=np.array([3, 200, -1.60]), size=3)
    print(f"Finding new with ref point: {ref_point}...")

    res = differential_evolution(
        lambda x: objective(to_ints(x), ref_point=ref_point),
        bounds,
        maxiter=2000,
        init="sobol",
        strategy="best1bin",
        disp=True,
        # constraints=con_ws,
        polish=True,
    )

    x = res.x
    obj = objective_f(to_ints(x))
    ws = work_loads(to_ints(x))

    print(f"f1: {obj[0]}", f"f2: {obj[1]}")
    # print(f"Cons: {ws <= 1.2} and {ws >= 0.8}")
    print(f"Work loads: {work_loads(to_ints(x))}")
    print(f"x: {to_ints(x)}")

    if res.success:
        with open("./results_3d.dat", "a") as fh:
            obj = " ".join(map(str, obj))
            var = " ".join(map(str, to_ints(x)))
            line = f"{obj} {var}\n"
            fh.write(line)

    print(f"Found solutions: {found}")
    found += 1
