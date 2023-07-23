from fundamental import Fundamental

import numpy as np

class FivePoint(Fundamental):
    def __init__(self) -> None:
        super().__init__(5)

    def estimate(self, kpts1, kpts2):
        A = np.zeros((5,9))
        for i in range(A.shape[0]):
            A[i][0] = kpts1[i][0] * kpts2[i][0]
            A[i][1] = kpts1[i][1] * kpts2[i][0]
            A[i][2] = kpts2[i][0]
            A[i][3] = kpts1[i][0] * kpts2[i][1]
            A[i][4] = kpts1[i][1] * kpts2[i][1]
            A[i][5] = kpts2[i][1]
            A[i][6] = kpts1[i][0]
            A[i][7] = kpts1[i][1]
            A[i][8] = 1.0
        
        # nullspace: consists of 4 vectors:
        u,s,vt = np.linalg.svd(A, full_matrices=True)

        null_vecs = vt.T[:,5:]
        print(f"null_vecs.shape: {null_vecs.shape}")
        print(null_vecs)

