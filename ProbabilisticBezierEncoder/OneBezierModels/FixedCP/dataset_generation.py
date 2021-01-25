import os
import time
import torch

from scipy.special import binom
from torch.utils.data import Dataset


class OneBezierDataset(Dataset):
    def __init__(self, basedir="/home/albert/PycharmProjects/transformer_bezier/Datasets", training=True):
        super().__init__(self)
        if training:
            basedir = os.path.join(basedir, "Training")
        else:
            basedir = os.path.join(basedir, "Test")
        self._images = torch.load(os.path.join(basedir, "images/1bezier"))
        self._sequences = torch.load(os.path.join(basedir, "sequences/1bezier"))

    def __getitem__(self, idx):
        return self._images[idx], self._sequences[idx]

def bezier(CP, num_cps, t, device='cuda'):
    """
    CP.shape = (max_num_cp, batch_size, 2)
    num_cp.shape = (batch_size)
    t.shape=(1, resolution)

    Returns tensor of shape=(batch_size, resolution, 2) containing the "resolution" points belonging to the "batch_size" bezier curves evaluated in "t".
    """
    # Calculation of all the binomial coefficients needed for the computation of bezier curves
    binomial_coefs = torch.zeros_like(CP[:, :, 0])
    for i, num_cp in enumerate(num_cps):
        binomial_coefs[:num_cp, i] = binom((num_cp-1).cpu(), [k for k in range(num_cp)])

    output = torch.zeros((num_cps.shape[0], t.shape[1], 2), dtype=torch.float32, device=device)
    t_inv = 1-t

    for i, P in enumerate(CP):
        output = output + (binomial_coefs[i].unsqueeze(1) * t**i * t_inv**((num_cps-1-i).unsqueeze(1))).unsqueeze(-1) * P.unsqueeze(1)
    return torch.round(output).long()

def generate1bezier(im_size=64, batch_size=64, num_control_points=3, resolution=150, device='cuda'):
    images = torch.zeros((batch_size, 1, im_size, im_size), dtype=torch.float32, device=device)
    tgt_seq = torch.empty((num_control_points+1, batch_size), dtype=torch.long, device=device)

    # El ultimo token de todas las secuencias será el que indica que debemos parar
    tgt_seq[-1] = im_size*im_size

    # Generamos aleatoriamente los control points de las curvas del dataset
    control_points = (im_size-0.5)*torch.rand((num_control_points, batch_size, 2), device=device)

    #Generamos la tgt_seq correspondiente a los puntos de control
    rounded_cp = torch.round(control_points)
    tgt_seq[:-1, :] = im_size*rounded_cp[:, :, 0] + rounded_cp[:, :, 1]

    output = bezier(control_points, num_control_points*torch.ones(batch_size, dtype=torch.long, device=device), torch.linspace(0, 1, resolution, device=device).unsqueeze(0), device=device)

    #output.shape=(batch_size, resolution, 2)
    for i in range(batch_size):
        images[i, 0, output[i, :, 0], output[i, :, 1]] = 1
    return images, tgt_seq


if __name__ == '__main__':
    # basedir = "/data2fast/users/asuso"
    basedir = "/home/albert/PycharmProjects/trans_bezier"

    t0 = time.time()
    for num_cp in [3, 4, 5, 6, 7, 8]:
        im, seq = generate1bezier(batch_size=64, num_control_points=num_cp, device='cuda')
        im = im.to('cpu')
        seq = seq.to('cpu')
        torch.save(im, basedir+"/Datasets/OneBezierDatasets/Training/images/fixedCP"+str(num_cp))
        torch.save(seq, basedir+"/Datasets/OneBezierDatasets/Training/sequences/fixedCP"+str(num_cp))

        im, seq = generate1bezier(batch_size=10000, num_control_points=num_cp, device='cuda')
        im = im.to('cpu')
        seq = seq.to('cpu')
        torch.save(im, basedir+"/Datasets/OneBezierDatasets/Test/images/fixedCP"+str(num_cp))
        torch.save(seq, basedir+"/Datasets/OneBezierDatasets/Test/sequences/fixedCP"+str(num_cp))

    print("En generar tots els datasets hem trigat", time.time()-t0)
