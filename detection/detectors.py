import numpy as np
import tensorflow as tf
import pickle as pkl
import itertools


class DiscontinuityDetector:
    def __init__(self, data_folder, model_name):
        self.model_artifact = tf.keras.models.load_model(f'{data_folder}/{model_name}')
        with open(f'{data_folder}/SGG.pkl', 'rb') as file:
            self.sgg = pkl.load(file)

    def _place_sg(self, new_origins, new_edgelens):
        """
        Method that evaluate the points of SGs similar to the one stored in self.sgg
        :param new_origins: M-by-n array of the origins of the M n-dimensional SGs
        :param new_edgelens: 1D-array of M elements, representing the edge-length of the M SGs
        :return: M-by-N-by-n array of the function evaluations for each one of the N pts of each one of the M SGs
        """
        # TRANSFORM new_origins INTO A 2D-ARRAY, IN CASE IT IS A 1D-ARRAY (E.G., THE CASE OF ONLY 1 SG).
        new_origins = np.array(new_origins, ndmin=2)
        M, n = new_origins.shape
        # TRANSFORM new_edgelens INTO A COLUMN-VECTOR
        new_edgelens = np.array(new_edgelens, ndmin=2).T
        # EXPAND new_edgelens TO THE THIRD AXIS AND REPEAT FOR EACH DIMENSION  (NEW SHAPE (M, 1, n))
        new_edgelens = np.expand_dims(new_edgelens, axis=-1)
        new_edgelens_copy = new_edgelens.copy()
        new_edgelens = np.concatenate([new_edgelens_copy for i in range(n)], axis=-1)

        # PREPARE THE M SGs (SHAPE (M, N, n))
        # OLD VERSION:
        # SGs = np.vstack([np.expand_dims(self.sgg['grid_pts'].copy(), axis=0) for m in range(M)])
        # NEW VERSION (MORE EFFICIENT):
        SGs = np.repeat(np.expand_dims(self.sgg['grid_pts'].copy(), axis=0), M, axis=0)
        SGs = (SGs / 2) * new_edgelens

        SGs = SGs + np.expand_dims(new_origins, axis=1)  # ADD RESHAPED new_origins (NEW SHAPE (M, 1, n))

        return SGs

    @staticmethod
    def _eval_func_at_placedsg(SGs, eval_function):
        """
        Method that evaluate a target function at the points of SGs similar to the one stored in self.sgg
        :param SGs: output of method self._place_sg
        :param eval_function: function to be used for the evaluation. We assume to have a function that takes as input
            a K-by-n array of K n-dimensional pts, and that returns an array of K function evaluation at those pts.
        :return: M-by-N array of the function evaluations for each one of the N pts of each one of the M SGs
        """

        M, N, n = SGs.shape
        Y = eval_function(SGs.reshape((M * N, n))).reshape((M, N))

        return Y

    def _place_sg_and_eval_func(self, new_origins, new_edgelens, eval_function):
        """
        Method that evaluate a target function at the points of SGs similar to the one stored in self.sgg
        :param new_origins: M-by-n array of the origins of the M n-dimensional SGs
        :param new_edgelens: 1D-array of M elements, representing the edge-length of the M SGs
        :param eval_function: function to be used for the evaluation. We assume to have a function that takes as input
            a K-by-n array of K n-dimensional pts, and that returns an array of K function evaluation at those pts.
        :return: M-by-N array of the function evaluations for each one of the N pts of each one of the M SGs
        """

        SGs = self._place_sg(new_origins, new_edgelens)

        # OLD VERSION:
        # M, N, n = SGs.shape
        # Y = eval_function(SGs.reshape((M * N, n))).reshape((M, N))
        # NEW VERSION:
        Y = self._eval_func_at_placedsg(SGs, eval_function)

        return Y

    @staticmethod
    def __input_preprocesser(X):
        # TRANSFORM X INTO A 2D-ARRAY, IN CASE IT IS A 1D-ARRAY (I.E., THE CASE OF ONLY 1 SG EVALUATION).
        X = np.array(X, ndmin=2)
        X_rowmaxs = np.abs(X).max(axis=1)

        # TO AVOID ZERO-DIVISION ERROR AND ASSUMING THE DIVISION BY ANY SMALL NON-ZERO ELEMENT WORKS
        mask_where_zeroflat = (X_rowmaxs == 0.)
        X_rowmaxs[mask_where_zeroflat] = 1.

        X_rowmaxs = X_rowmaxs.reshape(X.shape[0], 1)

        X_preprocessed = X / X_rowmaxs

        # BECAUSE DURING THE TRAINING FLAT REGIONS ARE TRANSFORMED INTO +/-1 VALUES FOR ALL THE NODES,
        # NOT INTO 0 VALUES FOR ALL THE NODES
        X_preprocessed[mask_where_zeroflat, :] = 1.

        return X_preprocessed

    def _soft_detect(self, function_evaluations):
        """
        Method that return the NN model outputs for the function evaluations received as input.
        The output values are in [0, 1] (0 = non-troubled pt, 1 = troubled pt)
        :param function_evaluations: M-by-N array, where each row is the evaluation of the target function at the N
            pts of a sparse grid similar to the one stored in self.sgg
        :return: M-by-N array of values in [0, 1] (0 = non-troubled pt, 1 = troubled pt)
        """
        Y = function_evaluations
        pred = self.model_artifact.serve(self.__input_preprocesser(Y))

        return pred

    def detect(self, function_evaluations, thresh=0.5):
        """
        Method that return the NN model outputs for the function evaluations received as input.
        The output values are in {0, 1} (0 = non-troubled pt, 1 = troubled pt), according to the chosen threshold value.
        :param function_evaluations: M-by-N array, where each row is the evaluation of the target function at the N
            pts of a sparse grid similar to the one stored in self.sgg
        :param thresh: float value in (0, 1)
        :return: M-by-N array of values in {0, 1} (0 = non-troubled pt, 1 = troubled pt), according to the chosen
            threshold value
        """
        Y = function_evaluations
        pred = np.heaviside(self._soft_detect(Y) - thresh, 1.)

        return pred

    def _place_sg_and_soft_detect(self, new_origins, new_edgelens, eval_function):
        Y = self._place_sg_and_eval_func(new_origins, new_edgelens, eval_function)
        pred = self._soft_detect(Y)

        return pred

    def place_sg_and_detect(self, new_origins, new_edgelens, eval_function, thresh=0.5):
        Y = self._place_sg_and_eval_func(new_origins, new_edgelens, eval_function)
        pred = self.detect(Y, thresh)

        return pred


discdet_ginn_2d = DiscontinuityDetector(data_folder='data/2D', model_name='GINN_2D')
discdet_mlp_2d = DiscontinuityDetector(data_folder='data/2D', model_name='MLP_2D')
discdet_ginn_4d = DiscontinuityDetector(data_folder='data/4D', model_name='GINN_4D')

starting_origins_v00 = np.zeros((1, 2))
starting_edgelens_v00 = [2.]

starting_origins_v01 = starting_origins_v00.copy()
starting_edgelens_v01 = starting_edgelens_v00.copy()
for p in itertools.product([-0.5, 0.5], [-0.5, 0.5]):
    starting_origins_v01 = np.vstack([starting_origins_v01, np.array(p, ndmin=2)])
    starting_edgelens_v01.append(1.)

starting_origins_v02 = starting_origins_v01.copy()
starting_edgelens_v02 = starting_edgelens_v01.copy()
for p in itertools.product([-0.75, -0.25, 0.25, 0.75], [-0.75, -0.25, 0.25, 0.75]):
    starting_origins_v02 = np.vstack([starting_origins_v02, np.array(p, ndmin=2)])
    starting_edgelens_v02.append(0.5)

starting_origins_v03 = starting_origins_v02.copy()
starting_edgelens_v03 = starting_edgelens_v02.copy()
for p in itertools.product([-0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875],
                           [-0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875]
                           ):
    starting_origins_v03 = np.vstack([starting_origins_v03, np.array(p, ndmin=2)])
    starting_edgelens_v03.append(0.25)

starting_origins_v04 = starting_origins_v03.copy()
starting_edgelens_v04 = starting_edgelens_v03.copy()
for p in itertools.product([-0.9375, -0.8125, -0.6875, -0.5625, -0.4375, -0.3125, -0.1875, -0.0625,
                            0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375],
                           [-0.9375, -0.8125, -0.6875, -0.5625, -0.4375, -0.3125, -0.1875, -0.0625,
                            0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375],
                           ):
    starting_origins_v04 = np.vstack([starting_origins_v04, np.array(p, ndmin=2)])
    starting_edgelens_v04.append(0.125)


