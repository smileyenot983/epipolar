from fundamental import * 
import random

"""
RANSAC overview
ransac has cycle, where:
1. sample N points
2. kernel.fit() - estimates model given sampled points
3. scorer.score() - compute number of inliers
4. if number of inliers > current -> rewrite current
"""



"""
Ransac class which does following:
1. samples N points from data
2. calculates some given parameter(for example fundamental or essential matrix)
3. checks proportion of inliers
4. if enough iterations done or enough inliers obtained -> return best results
"""
class Ransac:
    def __init__(self, algo) -> None:
        # algorithm to run
        self.estimator = algo
        self.n_iterations = 1024
        self.min_samples = algo.min_samples_

        # error = x'^T * F * x
        self.threshold = 1

    def run(self, kpts1, kpts2):
        assert(kpts1.shape[0] >= self.min_samples)
        assert(kpts2.shape[0] >= self.min_samples)
        assert(kpts1.shape[0] == kpts2.shape[0])

        F_best = None
        ratio_best = 0.0
        for i in range(self.n_iterations):
            # sample kpts
            curr_idxs = random.sample(range(0, kpts1.shape[0]), self.min_samples)

            # estimate given parameter
            F = self.estimator.estimate(kpts1[curr_idxs], kpts2[curr_idxs])
            # calculate number of inliers
            inlier_ratio = self.estimator.calc_inlier_ratio(kpts1, kpts2, F, 1.0)
            if(inlier_ratio > ratio_best):
                ratio_best = inlier_ratio
                F_best = F
            # print(f"inlier_ratio: {inlier_ratio}")
            # err = self.estimator.calc_err(kpts1, kpts2, F)

        return ratio_best, F_best