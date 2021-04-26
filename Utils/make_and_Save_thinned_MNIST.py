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

num_digits_per_image = 2

dataset = torch.zeros(60000, 1, 64, 64)
position = torch.randint(0, 64-28+1, (60000, 2, num_digits_per_image), dtype=torch.long)
digit_pairs = torch.randint(0, 60000, (60000, 2))


for i in range(60000):
    for j in range(num_digits_per_image):
        dataset[i, 0, position[i, 0, j]:position[i, 0, j]+28, position[i, 1, j]:position[i, 1, j]+28] = data[digit_pairs[i, j], 0]

"""transform = transforms.Compose([
    transforms.Resize(64),
    transforms.Normalize((0.1307,), (0.3081,)),
    lambda x: x > 1,
    lambda x: x.float(),
])
dataset = transform(data)"""

torch.save(dataset, dataset_basedir+"/Datasets/MNIST/thinned_relocated_"+str(num_digits_per_image)+"digitsPerImage")
#torch.save(dataset, dataset_basedir+"/Datasets/MNIST/thinned_resized_umbral1")

print(dataset.shape)

idx = 157
for idx in range(157, 170):
    plt.imshow(dataset[idx, 0], cmap='gray')
    plt.show()

    assert torch.sum(dataset[idx, 0] == 0)+torch.sum(dataset[idx, 0] == 1) == dataset[idx, 0].shape[0]*dataset[idx, 0].shape[1]
    assert torch.sum(dataset[idx, 0] == 0) > torch.sum(dataset[idx, 0] == 1)
    assert torch.sum(dataset[idx, 0] == 1) > 0
