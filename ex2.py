import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    return labeled_img



images_path = "/home/asuso/PycharmProjects/trans_bezier/Datasets/MNIST/thinned_relocated_2digitsPerImage"
images = torch.load(images_path)

idx = 14569
for n in range(1):
    img = images[idx+n, 0].numpy().astype(np.uint8)

    num_labels, labels_im = cv2.connectedComponents(img)

    print("Imagen numero", n)
    for i in range(num_labels):
        print(i, np.sum(labels_im == i))
    print("\n\n")

    """segmented_img = imshow_components(labels_im)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_img)
    plt.title("Segmented ({})".format(num_labels))
    plt.show()"""