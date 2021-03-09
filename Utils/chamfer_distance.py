import numpy as np
import cv2
import torch


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


def generate_loss_images(original_images, weight=0.1, distance='l2'):
    loss_images = original_images.detach().clone()
    im_size = loss_images.shape[-1]
    images = loss_images.view(-1, im_size, im_size)

    for idx, im in enumerate(images):
        im = im.numpy().astype(np.ubyte)
        im_scaled = 255 * im

        dst_map = cv2.distanceTransform(~im_scaled, distanceType=cv2.DIST_L2, maskSize=3)
        if distance == 'quadratic':
            dst_map = dst_map*dst_map
        elif distance == 'exp':
            dst_map = np.exp(dst_map)
        dst_map = weight * dst_map * np.sum(im) / np.sum(dst_map)

        loss_images[idx, 0] -= torch.from_numpy(dst_map)

    return loss_images

def generate_distance_images(original_images, distance='l2'):
    distance_images = torch.empty_like(original_images)
    im_size = distance_images.shape[-1]
    images = original_images.view(-1, im_size, im_size)

    for idx, im in enumerate(images):
        im = im.numpy().astype(np.ubyte)
        im_scaled = 255 * im

        dst_map = cv2.distanceTransform(~im_scaled, distanceType=cv2.DIST_L2, maskSize=3)
        if distance == 'quadratic':
            dst_map = dst_map*dst_map
        elif distance == 'exp':
            dst_map = np.exp(dst_map)

        distance_images[idx, 0] = torch.from_numpy(dst_map)

    return distance_images



if __name__ == '__main__':
    import torch
    import os

    basedir = "/home/asuso/PycharmProjects/trans_bezier"
    images = torch.load(os.path.join(basedir, "Datasets/OneBezierDatasets/Training/images/fixedCP" + str(5)))
    images = torch.load(os.path.join(basedir, "Datasets/MNIST/thinned_umbral1"))

    offset_x = 0
    offset_y = 0

    idx = 157

    im = images[157]

    print(im[0])