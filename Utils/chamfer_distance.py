import numpy as np
import cv2


def chamfer_distance(prediction, target):
    """Trabaja con arrays de shape (batch_size, height, width)"""
    diag = np.sqrt(prediction.shape[1]**2 + prediction.shape[2]**2)
    distance = []
    for i in range(target.shape[0]):
        img1 = prediction[i].astype(np.ubyte)
        img2 = target[i].astype(np.ubyte)
        # Check images have something
        if img1.all() or not img1.any() or img2.all() or not img2.any():
            distance.append(diag)
            print("Esto pasa: imagen vacia")
            continue

        edges1 = 255*img1
        edges2 = 255*img2
        """edges1 = cv2.Canny(img1, 1,2)
        edges2 = cv2.Canny(img2, 1,2)
        if edges1.all() or not edges1.any() or edges2.all() or not edges2.any():
            distance.append(diag)
            continue"""
        dst1 = cv2.distanceTransform(~edges1, distanceType=cv2.DIST_L2, maskSize=3)
        dst2 = cv2.distanceTransform(~edges2, distanceType=cv2.DIST_L2, maskSize=3)


        dst = (dst1*edges2).sum()/edges2.sum() + (dst2*edges1).sum()/edges1.sum()
        dst /= 2.0
        distance.append(dst)

    distance = np.array(distance)
    return distance