import numpy as np
import tensorflow as tf


class DiscontinuityDetection:
    def __init__(self, detector, target_function,
                 domain_hypercube_origin, domain_hypercube_edgelen,
                 ):
        self.detector = detector
        self.target_function = target_function
        self.domain_hypercube_origin = domain_hypercube_origin
        self.domain_hypercube_edgelen = domain_hypercube_edgelen
        self.__shortestedge2edgelen = np.round(2 / np.array(self.detector.sgg['nodes_maxedge_len']))
        self.__max_discretelevel_sparsegrid = np.round(np.log2(2 /
                                                               np.array(self.detector.sgg['nodes_maxedge_len']).min()
                                                               )
                                                       ) + 1

        self.sg_to_visit = {'origins': None, 'edgelens': None}
        self.sg_visited = {'origins': None, 'edgelens': None}
        self.troubled_pts = None
        self.lambda_min = None
        self.detection_threshold = None
        self.max_discretelevel_for_clipping = None
        self.__clipper_vec = None
        self.__clipper_step = None

    def __pts_clipper(self, X):
        M, n = X.shape
        clip_num = self.__clipper_vec.size
        for dim in range(n):
            clip_reldim = self.__clipper_vec + self.domain_hypercube_origin[dim]
            clip_start = clip_reldim[0]
            indices = np.round((X[:, dim] - clip_start) / self.__clipper_step).astype(int)
            indices = np.clip(indices, 0, clip_num - 1)
            X[:, dim] = clip_reldim[indices]

        return X

    def __edgelens_clipper(self, L):
        clip_num = self.__clipper_vec.size
        clip_reldim = self.__clipper_vec + 1.
        clip_start = clip_reldim[0]
        indices = np.round((L - clip_start) / self.__clipper_step).astype(int)
        indices = np.clip(indices, 0, clip_num - 1)

        return clip_reldim[indices]

    def initialize_detection(self, starting_sg_origins, starting_sg_edgelens,
                             lambda_min=(2 ** (-5)), detection_threshold=0.5
                             ):
        """
        Method that set the starting SGs for the detection algorithm, similar to the one stored in self.detector.sgg
        :param starting_sg_origins: M-by-n array of the origins of the M n-dimensional SGs
        :param starting_sg_edgelens: 1D-array of M elements, representing the edge-length of the M SGs
        :param lambda_min: the edge-length threshold for stopping the detection algorithm
        :param detection_threshold: float value in (0, 1) for the detector's threshold
        """
        self.lambda_min = lambda_min
        self.max_discretelevel_for_clipping = (np.ceil(np.log2(self.domain_hypercube_edgelen / self.lambda_min) + 1) +
                                               self.__max_discretelevel_sparsegrid
                                               )
        clipper_vec = np.linspace(0., 0.5 * self.domain_hypercube_edgelen,
                                  int(2 ** (self.max_discretelevel_for_clipping - 2) + 1)
                                  )
        self.__clipper_vec = np.concatenate([-clipper_vec[-1:0:-1], clipper_vec])
        self.__clipper_step = self.domain_hypercube_edgelen / (2 ** (self.max_discretelevel_for_clipping - 1))

        self.sg_to_visit = {'origins': self.__pts_clipper(starting_sg_origins),
                            'edgelens': self.__edgelens_clipper(starting_sg_edgelens)
                            }
        self.detection_threshold = detection_threshold

    def _aposteriori_count_func_evaluations(self, verbose=False):
        if verbose:
            print('')
            print(f'@@@@@@@@@@@@@@@ NUMBER OF VISITED SGs: {self.sg_visited["origins"].shape[0]} @@@@@@@@@@@@@@@')

        SGs = self.detector._place_sg(self.sg_visited['origins'], self.sg_visited['edgelens'])
        M, N, n = SGs.shape

        SGs_uniques = SGs.reshape((M * N, n))
        SGs_uniques = np.unique(SGs_uniques, axis=0)

        funceval_count = M * N
        unique_funceval_count = SGs_uniques.shape[0]

        return funceval_count, unique_funceval_count

    def one_step_detection(self, verbose=False):
        if verbose:
            print('')
            print(f'@@@@@@@@@@@@@@@ NEW STEP: {self.sg_to_visit["origins"].shape[0]} SGs TO BE VISITED @@@@@@@@@@@@@@@')

        SGs = self.detector._place_sg(self.sg_to_visit['origins'], self.sg_to_visit['edgelens'])

        M, N, n = SGs.shape
        SGs = self.__pts_clipper(SGs.reshape((M * N, n))).reshape((M, N, n))

        Y = self.detector._eval_func_at_placedsg(SGs, self.target_function)
        preds = self.detector.detect(Y, thresh=self.detection_threshold)

        if self.sg_visited['origins'] is None:
            self.sg_visited = self.sg_to_visit.copy()
        else:
            self.sg_visited['origins'] = np.vstack([self.sg_visited['origins'], self.sg_to_visit['origins'].copy()])
            self.sg_visited['edgelens'] = np.concatenate([self.sg_visited['edgelens'], self.sg_to_visit['edgelens'].copy()])

        maxe_lens = np.vstack([self.__edgelens_clipper(0.5 *
                                                       np.array(self.detector.sgg['nodes_maxedge_len'], ndmin=2) *
                                                       e_len)
                               for e_len in self.sg_to_visit['edgelens']
                               ]
                              )

        to_visit_inds = np.argwhere(preds * maxe_lens >= self.lambda_min)
        if verbose:
            print(f'*** DETECTED {to_visit_inds.shape[0]} "CURRENT-LEVEL" TROUBLED POINTS (I.E., NEW "TO VISIT" SGs)')

        new_origins = SGs[to_visit_inds[:, 0], to_visit_inds[:, 1], :]
        new_edgelens = maxe_lens[to_visit_inds[:, 0], to_visit_inds[:, 1]]

        # REMOVE DUPLICATES AND ALREADY VISITED FROM THE (origin, edgelen) TO BE VISITED IN THE NEXT STEP
        true_to_visit_inds = []
        already_visited_count = 0
        all_new = np.unique(np.hstack([new_origins, np.expand_dims(new_edgelens, axis=1)]), axis=0)
        all_visited = np.hstack([self.sg_visited['origins'], np.expand_dims(self.sg_visited['edgelens'], axis=1)])
        if verbose:
            print(f'*** CHECKING FOR ALREADY-VISITED SGs (REMOVED FROM THE "TO VISIT" LIST)')

        for ii in range(all_new.shape[0]):
            if np.linalg.norm(all_new[ii:ii + 1, :] - all_visited, axis=1).min() > 0:  # I.E.: all_new[ii, :] not in all_visited
                true_to_visit_inds.append(ii)
            else:
                already_visited_count += 1

        if verbose:
            print(f'########### FOUND {already_visited_count} ALREADY-VISITED SGs #############')

        self.sg_to_visit['origins'] = all_new[true_to_visit_inds, :2]
        self.sg_to_visit['edgelens'] = all_new[true_to_visit_inds, 2]

        # DETECT "FINAL" TROUBLE POINTS
        preds_inf = preds.copy()
        preds_inf[preds_inf == 0] = np.inf
        are_troubled = np.argwhere(preds_inf * maxe_lens < self.lambda_min)
        new_troubled = SGs[are_troubled[:, 0], are_troubled[:, 1], :]
        if verbose:
            print(f'*** DETECTED {new_troubled.shape[0]} "FINAL" TROUBLED POINTS (AT THIS STEP)')

        if self.troubled_pts is None:
            self.troubled_pts = new_troubled
        else:
            if verbose:
                print(f'*** CHECKING FOR ALREADY-DETECTED "FINAL" TROUBLED POINTS (I.E., REMOVE DUPLICATES)')

            self.troubled_pts = np.unique(np.vstack([self.troubled_pts, new_troubled]), axis=0)












