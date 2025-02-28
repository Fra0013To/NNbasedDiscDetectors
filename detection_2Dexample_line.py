import numpy as np
from detection.detectors import discdet_ginn_2d, starting_origins_v02, starting_edgelens_v02
from detection.algorithms import DiscontinuityDetection

import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
print("Switched to:", matplotlib.get_backend())


def test_discontinuous(X):
    cont_reg = (2 * X[:, 0] >= X[:, 1])
    Xnorms = np.linalg.norm(X, axis=1)
    y1 = 2 * np.sin(1.25 * np.pi * Xnorms) + 4
    y = 2 * np.sin(0.75 * np.pi * Xnorms)

    y[cont_reg] = y1[cont_reg]

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
                            lambda_min=1e-3, detection_threshold=0.5
                            )

while detect.sg_to_visit['edgelens'].size > 0:
    detect.one_step_detection(verbose=True)

    plt.figure()
    plt.contourf(X, Y, Z, cmap='gray')
    plt.scatter(detect.sg_visited['origins'][:, 0], detect.sg_visited['origins'][:, 1], marker='x', c='cyan',
                label='origins (visited)')
    plt.scatter(detect.sg_to_visit['origins'][:, 0], detect.sg_to_visit['origins'][:, 1], marker='o', c='yellow',
                label='next origins to be visited')
    plt.scatter(detect.troubled_pts[:, 0], detect.troubled_pts[:, 1], marker='d', c='magenta',
                label='final troubled points')
    plt.legend()

plt.show()
