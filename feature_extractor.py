import cv2


# for now only SIFT features are supported, probably more will be added
class FeatureExtractor:
    def __init__(self) -> None:
        self.extractor = cv2.SIFT_create()

    
    def extract(self, img):
        kpts, descs = self.extractor.detectAndCompute(img, None)

        return kpts, descs
        