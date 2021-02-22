import pickle
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# dataset_basedir = "/data2fast/users/asuso"
dataset_basedir = "/home/asuso/PycharmProjects/trans_bezier"

mnist = pickle.load(open(dataset_basedir+"/Datasets/MNIST/mnist-thinned.pkl", 'rb'))
data = mnist['data'].astype(np.float32) / 255.
data = torch.from_numpy(data).permute(0, 3, 1, 2)

dataset = torch.zeros(60000, 1, 64, 64)
position = torch.randint(0, 64-28+1, (60000, 2), dtype=torch.long)

for i in range(60000):
    dataset[i, 0, position[i, 0]:position[i, 0]+28, position[i, 1]:position[i, 1]+28] = data[i, 0]

"""transform = transforms.Compose([
    transforms.Resize(64),
    transforms.Normalize((0.1307,), (0.3081,)),
    lambda x: x > 1,
    lambda x: x.float(),
])
dataset = transform(data)"""

torch.save(dataset, dataset_basedir+"/Datasets/MNIST/thinned_relocated")
#torch.save(dataset, dataset_basedir+"/Datasets/MNIST/thinned_resized_umbral1")

idx = 157
plt.imshow(dataset[idx, 0], cmap='gray')
plt.show()
