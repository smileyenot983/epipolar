from fundamental import * 

class Ransac:
    def __init__(self, algo) -> None:
        # algorithm to run
        self.estimator = algo
        self.n_iterations = 1024

    def run(self, kpts1, kpts2):
        return self.estimator(kpts1, kpts2)