import numpy as np
from detection.detectors import discdet_ginn_2d, starting_origins_v02, starting_edgelens_v02
from detection.algorithms import DiscontinuityDetection
from time import time

import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
print("Switched to:", matplotlib.get_backend())


def test_discontinuous(X):
    y = np.cos(np.arctan2(X[:, 0] + 0.31, X[:, 1] + 0.43).flatten())

    return y


detect = DiscontinuityDetection(detector=discdet_ginn_2d, target_function=test_discontinuous,
                                domain_hypercube_origin=np.zeros(2), domain_hypercube_edgelen=4.)

starting_sg_origins = starting_origins_v02.copy()
starting_sg_edgelens = np.array(starting_edgelens_v02.copy())
# ADAPT THE STARTING SGs FROM DOMAIN [-1,1]^2 TO DOMAIN [-2,2]^2 OF THE TARGET FUNCTION
starting_sg_origins = 4 * (starting_sg_origins / 2)
starting_sg_edgelens = 4 * (starting_sg_edgelens / 2)

X, Y = np.meshgrid(np.linspace(-2, 2, 150), np.linspace(-2, 2, 150))
XX = np.hstack([np.expand_dims(X.flatten(), axis=1), np.expand_dims(Y.flatten(), axis=1)])
Z = test_discontinuous(XX).reshape(X.shape)

detect.initialize_detection(starting_sg_origins=starting_sg_origins, starting_sg_edgelens=starting_sg_edgelens,
                            lambda_min=2**(-4), detection_threshold=0.5
                            )

dt_steps = []
dt_global = time()
while detect.sg_to_visit['edgelens'].size > 0:
    dt_step = time()
    detect.one_step_detection(verbose=True)
    dt_steps.append(time() - dt_step)

    plt.figure()
    plt.contourf(X, Y, Z, cmap='gray', levels=100)
    plt.scatter(detect.sg_visited['origins'][:, 0], detect.sg_visited['origins'][:, 1], marker='x', c='cyan',
                label='SG-origins (visited)')
    plt.scatter(detect.sg_to_visit['origins'][:, 0], detect.sg_to_visit['origins'][:, 1], marker='o', c='yellow',
                label='next SG-origins to be visited')
    plt.scatter(detect.troubled_pts[:, 0], detect.troubled_pts[:, 1], marker='d', c='magenta',
                label='final troubled points')
    plt.legend()

dt_global = time() - dt_global
dt_global_noplots = np.sum(dt_steps)

funceval_count, unique_funceval_count = detect._aposteriori_count_func_evaluations()

print()
print('@@@@@@@@@@@@@@@ DETECTION "COSTS" @@@@@@@@@@@@@@@')
print(f'*** TOTAL TIME (PLOTS INCLUDED): {np.round(dt_global, decimals=4)} seconds')
print(f'*** TOTAL TIME (NO PLOTS): {np.round(dt_global_noplots, decimals=4)} seconds')
print(f'*** AVG TIME PER STEP (OVER {len(dt_steps)} STEPS): {np.round(np.mean(dt_steps), decimals=4)} seconds')
print(f'*** TOTAL SGs VISTED: {len(detect.sg_visited["edgelens"])}')
print(f'*** TOTAL UNIQUE FUNCTION EVALUATIONS: {unique_funceval_count} (~{np.round(np.sqrt(unique_funceval_count), decimals=2)}^2)')
print(f'*** TOTAL FUNCTION EVALUATIONS: {funceval_count} (~{np.round(np.sqrt(funceval_count), decimals=2)}^2)')


plt.show()
