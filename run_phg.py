import numpy as np
import cv2
# import torch
import matplotlib.pyplot as plt

from feature_extractor import FeatureExtractor
from feature_matcher import FeatureMatcher

from fundamental import eight_point
from fundamental import seven_point
from fundamental import levmarq

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

    kpts1_slice = []
    kpts2_slice = []
    for i in range(7):

        kpts1_slice.append(kpts1[matches[i].queryIdx])
        kpts2_slice.append(kpts2[matches[i].trainIdx])

    # eight_point(kpts1_slice, kpts2_slice)
    # seven_point(kpts1_slice, kpts2_slice)
    levmarq(kpts1_slice, kpts2_slice)

    

