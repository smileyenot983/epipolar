import cv2

class FeatureMatcher:
    def __init__(self, matcher_type = "flann"):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        if matcher_type == "flann":
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif matcher_type == "bf":
            self.matcher = cv2.BFMatcher()

    def match_features(self, desc1, desc2, ratio_test = True):
        
        if ratio_test:
            k = 2
        else:
            k=1
        matches = self.matcher.knnMatch(desc1, desc2, k)

        good_matches = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                
                good_matches.append(m)

        return good_matches