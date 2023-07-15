import numpy as np
import cv2

from scipy.optimize import least_squares

# calculates error = x1^T*F*x2
def calc_err_total(kpts1, kpts2, F):
    total_err = 0.0
    for i in range(kpts1.shape[0]):
        kp1 = np.array([kpts1[i,0], kpts1[i,1], 1.0])
        kp2 = np.array([kpts2[i,0], kpts2[i,1], 1.0])
        err = np.linalg.multi_dot([kp1.T,F,kp2])
        total_err += err

    return total_err


class Fundamental:
    def __init__(self, min_samples):
        self.min_samples_ = min_samples

    def calc_err_total(self, kpts1, kpts2, F):
        total_err = 0.0
        for i in range(kpts1.shape[0]):
            kp1 = np.array([kpts1[i,0], kpts1[i,1], 1.0])
            kp2 = np.array([kpts2[i,0], kpts2[i,1], 1.0])
            err = np.linalg.multi_dot([kp1.T,F,kp2])
            total_err += err

        return total_err
    
    def calc_err(self, kp1, kp2, F):
        kp1h = np.array([kp1[0], kp1[0], 1.0])
        kp2h = np.array([kp2[0], kp2[0], 1.0])

        return np.linalg.multi_dot([kp1h.T, F, kp2h])
    

    # returns inlier/n_total_samples ratio
    def calc_inlier_ratio(self, kpts1, kpts2, F, threshold):
        n_inliers = 0

        err_mean = 0
        for i in range(kpts1.shape[0]):
            err = self.calc_err(kpts1[i,:], kpts2[i,:], F)
            err_mean += err
            if(abs(err) < threshold):
                n_inliers+=1

        return n_inliers/kpts1.shape[0]

class EightPoint(Fundamental):
    def __init__(self):
        super().__init__(8)

    def estimate(self, kpts1, kpts2):
        assert(kpts1.shape == kpts2.shape )
        assert(kpts1.shape[0] == self.min_samples_)

        A = np.zeros((kpts1.shape[0],9))

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
        
        # find nullspace to get solution for Af = 0
        # u - [row_space, left null_space]
        # v - [column_space, null_space]    
        u,s,vt = np.linalg.svd(A, full_matrices=True)
        
        # only 1 vector in nullspace
        null_vec = vt.T[:,8]
        null_mat = null_vec.reshape((3,3))

        # enforcing constraint on third singular value being equal to 0
        u2,s2,vt2 = np.linalg.svd(null_mat, full_matrices=True)
        s2[2] = 0.0

        s2 = np.diag(s2)
        fundamental = np.linalg.multi_dot([u2,s2,vt2])
        # up-to-scale, dividing by last element to make it equal to 1
        fundamental /= fundamental[2,2]

        return fundamental

class SevenPoint(Fundamental):
    def __init__(self):
        super().__init__(7)

    def estimate(self, kpts1, kpts2):
        A = np.zeros((7,9))

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

            u,s,vt = np.linalg.svd(A, full_matrices=True)

            # 2 vecs in nullspace
            null_vec1 = vt.T[:,8]
            null_vec2 = vt.T[:,7]

            null_mat1 = null_vec1.reshape((3,3))
            null_mat2 = null_vec2.reshape((3,3))

            # creating variables, to make code more readable
            a = null_mat1[0,0]
            b = null_mat1[0,1]
            c = null_mat1[0,2]
            d = null_mat1[1,0]
            e = null_mat1[1,1]
            f = null_mat1[1,2]
            g = null_mat1[2,0]
            h = null_mat1[2,1]
            i = null_mat1[2,2]

            j = null_mat2[0,0]
            k = null_mat2[0,1]
            l = null_mat2[0,2]
            m = null_mat2[1,0]
            n = null_mat2[1,1]
            o = null_mat2[1,2]
            p = null_mat2[2,0]
            q = null_mat2[2,1]
            r = null_mat2[2,2]

            # here write the result of  constrant = det(f_1 + \lambda*f_2)=0 -> as a result there is a cubic polynomial
            coeffs =[
                a*e*i + b*f*g + c*d*h - a*f*h - b*d*i - c*e*g,
                a*e*r + a*i*n + b*f*p + b*g*o + c*d*q + c*h*m + d*h*l + e*i*j + f*g*k - 
                a*f*q - a*h*o - b*d*r - b*i*m - c*e*p - c*g*n - d*i*k - e*g*l - f*h*j,
                a*n*r + b*o*p + c*m*q + d*l*q + e*j*r + f*k*p + g*k*o + h*l*m + i*j*n - 
                a*o*q - b*m*r - c*n*p - d*k*r - e*l*p - f*j*q - g*l*n - h*j*o - i*k*m,
                j*n*r + k*o*p + l*m*q - j*o*q - k*m*r - l*n*p
            ]

            # finding roots of polynomial  A*x^3 + B*x^2 + C*x + D
            # cubic polynomial -> 1 or 3 solutions
            roots = np.roots([coeffs[3],coeffs[2],coeffs[1],coeffs[0]])

            # in order to choose best root we calc error and choose solution which minimizes this error
            fundamental_best = None
            err_best = None
            for root in roots:
                fundamental_i = null_mat1 + root * null_mat2
                err_i = self.calc_err_total(kpts1,kpts2,fundamental_i)

                if(err_best is None or err_i < err_best):
                    err_best = err_i
                    fundamental_best = fundamental_i

            return fundamental_best


class LevMarq(Fundamental):
    def __status__(self):
        print("status")

    def __del__(self):
        print("destructor")

    
    

# estimating fundamental matrix with 2 ways:
# 7 point + solve quadratic polynom
# 8 points

# at least 8 matched keypoint positions
# x^TFx' = 0 => Af = 0
# TODO: 1. add feature coords normalization
def eight_point(kpts1, kpts2):
    assert(kpts1.shape[0]==kpts2.shape[0]==8)
    A = np.zeros((kpts1.shape[0],9))

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
    
    # find nullspace to get solution for Af = 0
    # u - [row_space, left null_space]
    # v - [column_space, null_space]    
    u,s,vt = np.linalg.svd(A, full_matrices=True)
    
    # only 1 vector in nullspace
    null_vec = vt.T[:,8]
    null_mat = null_vec.reshape((3,3))

    # enforcing constraint on third singular value being equal to 0
    u2,s2,vt2 = np.linalg.svd(null_mat, full_matrices=True)
    s2[2] = 0.0

    s2 = np.diag(s2)
    fundamental = np.linalg.multi_dot([u2,s2,vt2])
    # up-to-scale, dividing by last element to make it equal to 1
    fundamental /= fundamental[2,2]

    return fundamental

def seven_point(kpts1, kpts2):
    assert(kpts1.shape[0]==kpts2.shape[0]==7)
    # assert(len(kpts1)==len(kpts2)==7)
    A = np.zeros((7,9))

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

    u,s,vt = np.linalg.svd(A, full_matrices=True)

    # 2 vecs in nullspace
    null_vec1 = vt.T[:,8]
    null_vec2 = vt.T[:,7]

    null_mat1 = null_vec1.reshape((3,3))
    null_mat2 = null_vec2.reshape((3,3))

    # creating variables, to make code more readable
    a = null_mat1[0,0]
    b = null_mat1[0,1]
    c = null_mat1[0,2]
    d = null_mat1[1,0]
    e = null_mat1[1,1]
    f = null_mat1[1,2]
    g = null_mat1[2,0]
    h = null_mat1[2,1]
    i = null_mat1[2,2]

    j = null_mat2[0,0]
    k = null_mat2[0,1]
    l = null_mat2[0,2]
    m = null_mat2[1,0]
    n = null_mat2[1,1]
    o = null_mat2[1,2]
    p = null_mat2[2,0]
    q = null_mat2[2,1]
    r = null_mat2[2,2]

    # here write the result of  constrant = det(f_1 + \lambda*f_2)=0 -> as a result there is a cubic polynomial
    coeffs =[
        a*e*i + b*f*g + c*d*h - a*f*h - b*d*i - c*e*g,
        a*e*r + a*i*n + b*f*p + b*g*o + c*d*q + c*h*m + d*h*l + e*i*j + f*g*k - 
        a*f*q - a*h*o - b*d*r - b*i*m - c*e*p - c*g*n - d*i*k - e*g*l - f*h*j,
        a*n*r + b*o*p + c*m*q + d*l*q + e*j*r + f*k*p + g*k*o + h*l*m + i*j*n - 
        a*o*q - b*m*r - c*n*p - d*k*r - e*l*p - f*j*q - g*l*n - h*j*o - i*k*m,
        j*n*r + k*o*p + l*m*q - j*o*q - k*m*r - l*n*p
    ]

    # finding roots of polynomial  A*x^3 + B*x^2 + C*x + D
    # cubic polynomial -> 1 or 3 solutions
    roots = np.roots([coeffs[3],coeffs[2],coeffs[1],coeffs[0]])

    # in order to choose best root we calc error and choose solution which minimizes this error
    fundamental_best = None
    err_best = None
    for root in roots:
        fundamental_i = null_mat1 + root * null_mat2
        err_i = calc_err_total(kpts1,kpts2,fundamental_i)

        if(err_best == None or err_i < err_best):
            err_best = err_i
            fundamental_best = fundamental_i

    return fundamental_best

            
def levmarq(kpts1, kpts2, F0 = None):
    assert(kpts1.shape[0] == kpts2.shape[0] )
    assert(kpts1.shape[0]>=7)
    # here should be cost function of type:
    # dist(x, l_e) + dist(x', l_e')
    # where x, x' - 2d feature coords
    # l_e, l_e' - epipolar lines(l_e' = F * x;   l_e = F^T *x')
    # here F - parameter of cost function, which should be optimized 
    #         cost: 3.6198104017824025e-05 
    def cost(F):

        total_loss = 0
        for i in range(kpts1.shape[0]):
            kpts1_h = np.array([kpts1[i,0],kpts1[i,1],1.0])
            kpts2_h = np.array([kpts2[i,0],kpts2[i,1],1.0])
        
            product1 = kpts1_h[0]*F[0]*kpts2_h[0] + kpts1_h[1]*F[3]*kpts2_h[0] + kpts1_h[2]*F[6]*kpts2_h[0]
            product2 = kpts1_h[0]*F[1]*kpts2_h[1] + kpts1_h[1]*F[4]*kpts2_h[1] + kpts1_h[2]*F[7]*kpts2_h[1]
            product3 = kpts1_h[0]*F[2]*kpts2_h[2] + kpts1_h[1]*F[5]*kpts2_h[2] + kpts1_h[2]*F[8]*kpts2_h[2]

            total_loss += product1+product2+product3
    
        return total_loss

    # get initial estimate(without initial estimate lev marquardt may fail)
    if(F0 is None):
        F0 = seven_point(kpts1[:7], kpts2[:7])
    F0 = np.ravel(F0[:3,:3])

    res = least_squares(lambda x: cost(x), F0)

    F_levmarq = np.array(res.x).reshape(3,3)
    # print(f"F_levmarq : {F_levmarq}")

    return F_levmarq


