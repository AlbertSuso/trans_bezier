import pickle
import torch
import torchvision.transforms as transforms
import numpy as np

# dataset_basedir = "/data2fast/users/asuso"
dataset_basedir = "/home/asuso/PycharmProjects/trans_bezier"

mnist = pickle.load(open(dataset_basedir+"/Datasets/MNIST/mnist-thinned.pkl", 'rb'))
data = mnist['data'].astype(np.float32) / 255.
data = torch.from_numpy(data).permute(0, 3, 1, 2)
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.Normalize((0.1307,), (0.3081,)),
    lambda x: x > 1,
    lambda x: x.float(),
])
dataset = transform(data)
torch.save(dataset, dataset_basedir+"/Datasets/MNIST/thinned_umbral1")
