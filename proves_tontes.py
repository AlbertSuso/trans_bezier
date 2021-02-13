import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from Utils.chamfer_distance import generate_loss_images


mnist = pickle.load(open('/home/asuso/mnist-handwriting-dataset/data/mnist-thinned.pkl', 'rb'))
data = mnist['data'].astype(np.float32) / 255.
data = torch.from_numpy(data).permute(0, 3, 1, 2)
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.Normalize((0.1307,), (0.3081,)),
    lambda x: x > 1,
    lambda x: x.float(),
])
dataset = transform(data)

loss_dataset = generate_loss_images(dataset, weight=0.1)

print(torch.min(loss_dataset))
print(torch.max(loss_dataset))


for idx in range(155, 500, 10):
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(data[idx, 0], cmap='gray')
    plt.title("Thinned 28x28")
    plt.subplot(1, 3, 2)
    plt.imshow(dataset[idx, 0], cmap='gray')
    plt.title("Thinned 64x64")
    plt.subplot(1, 3, 3)
    plt.imshow(loss_dataset[idx, 0], cmap='gray')
    plt.title("Loss 64x64")
    plt.show()



