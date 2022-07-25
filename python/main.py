import rbfopt

max_eval = 100
nlp_path = "ipopt"
minlp_path = "lp_solve"
settings = rbfopt.RbfoptSettings(
    max_evaluations=max_eval, global_search_method="solver", nlp_solver_path=nlp_path, minlp_solver_path=minlp_path
)

import numpy as np

from problem_def import simple_objective_fun

objective = simple_objective_fun
N = 22

to_beat = np.zeros(22)
to_beat[[3, 4, 5, 6, 7, 14]] = 0
to_beat[[9, 10, 11, 12, 13]] = 1
to_beat[[8, 15, 16, 17]] = 2
to_beat[[0, 1, 2, 18, 19, 20, 21]] = 3


lower_bounds = np.array([0] * N)
upper_bounds = np.array([3] * N)

# SR central bricks
# SR1 (0, 3)
# lower_bounds[3] = 0
# upper_bounds[3] = 0

# SR2 (1, 13)
# lower_bounds[13] = 1
# upper_bounds[13] = 1

# SR3 (2, 15)
# lower_bounds[15] = 2
# upper_bounds[15] = 2

# SR4 (3, 21)
# lower_bounds[21] = 3
# upper_bounds[21] = 3

var_types = np.array(["I"] * N)

bb = rbfopt.RbfoptUserBlackBox(N, lower_bounds, upper_bounds, var_types, lambda x: objective(x)[0])
alg = rbfopt.RbfoptAlgorithm(settings, bb)
val, x, itercount, evalcount, fast_evalcount = alg.optimize()

print(f"val: {val}")
print(f"x:{x}")
print(objective(x))
print(f"To beat {objective(to_beat)}")
