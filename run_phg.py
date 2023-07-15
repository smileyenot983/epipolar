import numpy as np
import cv2
# import torch
import matplotlib.pyplot as plt

from feature_extractor import FeatureExtractor
from feature_matcher import FeatureMatcher

# from fundamental import eight_point
# from fundamental import seven_point
# from fundamental import levmarq
# from fundamental import calc_err_total

from ransac import Ransac

from fundamental import * 

def read_img(path, grayscale = True):
    return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2GRAY)

def draw_kpts(im, kpts):
    im_kpts = np.copy(im)

    for kp in kpts:
        cv2.circle(im_kpts,(int(kp.pt[0]),int(kp.pt[1])),1,(255,0,0),2)

    return im_kpts

if __name__ == "__main__":
    print("start photogrammetry")

    path1 = "./data/img1.png"
    path2 = "./data/img2.png"

    im1 = read_img(path1)
    im2 = read_img(path2)

    extractor = FeatureExtractor()

    kpts1, descs1 = extractor.extract(im1)
    kpts2, descs2 = extractor.extract(im2)

    im1_kpts = draw_kpts(im1, kpts1)

    matcher = FeatureMatcher()

    matches = matcher.match_features(descs1, descs2)

    print(f"len(matches): {len(matches)}")

    kpts1_np = []
    kpts2_np = []
    for i in range(len(matches)):

        kpts1_np.append(kpts1[matches[i].queryIdx].pt)
        kpts2_np.append(kpts2[matches[i].trainIdx].pt)
    kpts1_np = np.array(kpts1_np)
    kpts2_np = np.array(kpts2_np)

    # f8 = eight_point(kpts1_np[:8,:], kpts2_np[:8,:])
    # err = calc_err_total(kpts1_np,kpts2_np, f8)
    # err_mean = err / kpts1_np.shape[0]
    # print(f"8 | err: {err}, err_mean: {err_mean}")
    # f7 = seven_point(kpts1_np[:7,:], kpts2_np[:7,:])
    # err = calc_err_total(kpts1_np,kpts2_np, f7)
    # err_mean = err / kpts1_np.shape[0]
    # print(f"7 | err: {err}, err_mean: {err_mean}")
    # # f_levmarq = levmarq(kpts1_np[:20,:], kpts2_np[:20,:])

    # fCV, mask = cv2.findFundamentalMat(kpts1_np, kpts2_np,cv2.LMEDS)
    # err = calc_err_total(kpts1_np,kpts2_np, fCV)
    # err_mean = err / kpts1_np.shape[0]
    # print(f"CV | err: {err}, err_mean: {err_mean}")

    # calc error on all matched keypoints
    # err8 = calc_err_total(kpts1_np, kpts2_np, f8)
    # err7 = calc_err_total(kpts1_np, kpts2_np, f7)
    # err_levmarq = calc_err_total(kpts1_np, kpts2_np, f_levmarq)
    # print(f"err8: {err8}")
    # print(f"err7: {err7}")
    # print(f"err_levmarq: {err_levmarq}")

    # print(f"f8 : {f8}")
    # print(f"f7 : {f7}")
    # print(f"f_levmarq : {f_levmarq}")

    eightPoint = EightPoint()
    ransac = Ransac(eightPoint)
    ratio, F = ransac.run(kpts1_np, kpts2_np)
    err_total = calc_err_total(kpts1_np, kpts2_np, F)
    err_mean = err_total/kpts1_np.shape[0]
    print(f"EightPoint| inlier_ratio: {ratio} | err_total: {err_total} | err_mean: {err_mean}")

    sevenPoint = SevenPoint()
    ransac = Ransac(sevenPoint)
    ratio, F = ransac.run(kpts1_np, kpts2_np)
    err_total = calc_err_total(kpts1_np, kpts2_np, F)
    err_mean = err_total/kpts1_np.shape[0]
    print(f"SevenPoint| inlier_ratio: {ratio} | err_total: {err_total} | err_mean: {err_mean}")

    # print(f"F_ransac: {F_ransac}")

    # levmarq = LevMarq(7)
    # levmarq.__status__()

    # eightPoint = EightPoint()
    # print(f"eightPoint.min_samples_: {eightPoint.min_samples_}")

    # ransac_estimator = Ransac(eightPoint)

    # ransac_estimator.run(kpts1_np, kpts2_np)


    


    

