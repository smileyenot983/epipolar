import numpy as np
import cv2

from scipy.optimize import least_squares

# estimating fundamental matrix with 2 ways:
# 7 point + solve quadratic polynom
# 8 points

# at least 8 matched keypoint positions
# x^TFx' = 0 => Af = 0
def eight_point(kpts1, kpts2):
    A = np.zeros((len(kpts1),9))

    for i in range(A.shape[0]):
        A[i][0] = kpts1[i].pt[0] * kpts2[i].pt[0]
        A[i][1] = kpts1[i].pt[1] * kpts2[i].pt[0]
        A[i][2] = kpts2[i].pt[0]
        A[i][3] = kpts1[i].pt[0] * kpts2[i].pt[1]
        A[i][4] = kpts1[i].pt[1] * kpts2[i].pt[1]
        A[i][5] = kpts2[i].pt[1]
        A[i][6] = kpts1[i].pt[0]
        A[i][7] = kpts1[i].pt[1]
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

    print(f"fundamental(my) : {fundamental}")

    kpts1_np = []
    kpts2_np = []
    for i in range(len(kpts1)):
        kpts1_np.append((int(kpts1[i].pt[0]),int(kpts1[i].pt[1])))
        kpts2_np.append((int(kpts2[i].pt[0]),int(kpts2[i].pt[1])))
    kpts1_np = np.array(kpts1_np)
    kpts2_np = np.array(kpts2_np)

    print(f"kpts1_np.shape: {kpts1_np.shape}")

    fundamental_cv2, mask_cv2 = cv2.findFundamentalMat(kpts1_np,kpts2_np, cv2.FM_7POINT)    
    print(f"fundamental_cv2: {fundamental_cv2}")


    for i in range(8):
        # checking the solution with x^T * F * x' = 0 condition
        x1 = np.array([kpts1_np[i,0], kpts1_np[i,1], 1.0])
        x2 = np.array([kpts2_np[i,0], kpts2_np[i,1], 1.0])

        product_my = np.linalg.multi_dot([x1.T, fundamental, x2])
        product_cv = np.linalg.multi_dot([x1.T, fundamental_cv2, x2])
        print(f"product_my: {product_my}")
        print(f"product_cv: {product_cv}")


def seven_point(kpts1, kpts2):

    assert(len(kpts1)==len(kpts2)==7)
    A = np.zeros((7,9))

    for i in range(A.shape[0]):
        A[i][0] = kpts1[i].pt[0] * kpts2[i].pt[0]
        A[i][1] = kpts1[i].pt[1] * kpts2[i].pt[0]
        A[i][2] = kpts2[i].pt[0]
        A[i][3] = kpts1[i].pt[0] * kpts2[i].pt[1]
        A[i][4] = kpts1[i].pt[1] * kpts2[i].pt[1]
        A[i][5] = kpts2[i].pt[1]
        A[i][6] = kpts1[i].pt[0]
        A[i][7] = kpts1[i].pt[1]
        A[i][8] = 1.0

    u,s,vt = np.linalg.svd(A, full_matrices=True)

    print(f"s: {s}")
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

    # here write the result of det(f_1 + \lambda*f_2)=0 -> as a result there is a cubic polynomial
    coeffs =[
        a*e*i + b*f*g + c*d*h - a*f*h - b*d*i - c*e*g,
        a*e*r + a*i*n + b*f*p + b*g*o + c*d*q + c*h*m + d*h*l + e*i*j + f*g*k - 
        a*f*q - a*h*o - b*d*r - b*i*m - c*e*p - c*g*n - d*i*k - e*g*l - f*h*j,
        a*n*r + b*o*p + c*m*q + d*l*q + e*j*r + f*k*p + g*k*o + h*l*m + i*j*n - 
        a*o*q - b*m*r - c*n*p - d*k*r - e*l*p - f*j*q - g*l*n - h*j*o - i*k*m,
        j*n*r + k*o*p + l*m*q - j*o*q - k*m*r - l*n*p
    ]

    roots = np.roots([coeffs[3],coeffs[2],coeffs[1],coeffs[0]])
    print(f"roots: {roots}")

    kpts1_np = []
    kpts2_np = []
    for i in range(len(kpts1)):
        kpts1_np.append((int(kpts1[i].pt[0]),int(kpts1[i].pt[1])))
        kpts2_np.append((int(kpts2[i].pt[0]),int(kpts2[i].pt[1])))
    kpts1_np = np.array(kpts1_np)
    kpts2_np = np.array(kpts2_np)


    fundamental_cv2, mask_cv2 = cv2.findFundamentalMat(kpts1_np,kpts2_np, cv2.FM_7POINT)  

    print(f"fundamental_cv2.shape: {fundamental_cv2.shape}")
    for i in range(int(fundamental_cv2.shape[0]/3)):
        fundamental_cv2_i = fundamental_cv2[3*i:3*(i+1),:]
        print(f"fundamental_cv2 i: {i} f: {fundamental_cv2_i}")

    # check determinant=0 and x^T*F*x'=0        product_my = np.linalg.multi_dot([x1.T, fundamental, x2])
    for l in roots:
        fundamental = null_mat1 + l*null_mat2
        det = np.linalg.det(fundamental)
        print(f"l: {l} det: {det} fundamental: {fundamental}")
        for i in range(7):
            # checking the solution with x^T * F * x' = 0 condition
            x1 = np.array([kpts1_np[i,0], kpts1_np[i,1], 1.0])
            x2 = np.array([kpts2_np[i,0], kpts2_np[i,1], 1.0])

            product_my = np.linalg.multi_dot([x1.T, fundamental, x2])
            print(f"product_my: {product_my}")
            
def levmarq(kpts1, kpts2):

    def func(x):
        return x[0]**2 + x[1]**2
    
    
    
    x0 = np.array([10.0,10.0])
    res = least_squares(lambda x: func(x), x0)

    print(res)


