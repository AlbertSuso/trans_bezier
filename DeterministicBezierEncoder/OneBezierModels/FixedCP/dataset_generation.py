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


def bezier(CP, t, device='cuda'):
    """
    :param CP: iterable of 2-size tensors of dim=1 (points in R^2)
    :param t: float in (0, 1)
    :return: 2-size tensor of dim=1 (point in R^2) corresponding to de evaluation  in t of the bezier curve
             with control points PC.
    """
    n = len(CP) - 1
    output = torch.tensor([0, 0], dtype=torch.float32, device=device)

    t_inv = (1-t)
    for i, P in enumerate(CP):
        output = output + binom(n, i) * t**i * t_inv**(n-i) * P

    return output

def generate1bezier(im_size=64, batch_size=64, num_control_points=3, resolution=150, device='cuda'):
    images = torch.zeros((batch_size, 1, im_size, im_size), dtype=torch.float32, device=device)
    tgt_seq = torch.empty((num_control_points+1, batch_size), dtype=torch.long, device=device)

    # El ultimo token de todas las secuencias será el que indica que debemos parar
    tgt_seq[-1] = im_size*im_size

    # Generamos aleatoriamente los control points de las curvas del dataset
    control_points = (im_size-0.5)*torch.rand((batch_size, num_control_points, 2), device=device)

    for i, CP in enumerate(control_points):

        # Generamos la tgt_seq correspondiente a los puntos de control CP
        rounded_CP = torch.round(CP)
        for k, P in enumerate(rounded_CP):
            tgt_seq[k, i] = P[0]*im_size + P[1]

        # Generamos la imagen correspondiente a los puntos de control CP
        for j, t in enumerate(torch.linspace(0, 1, resolution)):
            output = bezier(CP, t, device=device)
            output = torch.round(output).long()
            images[i, 0, output[0], output[1]] = 1

    return images, tgt_seq


if __name__ == '__main__':
    # basedir = "/data2fast/users/asuso"
    basedir = "/home/albert/PycharmProjects/trans_bezier"

    t0 = time.time()
    for num_cp in [4, 5, 6, 7, 8]:
        im, seq = generate1bezier(batch_size=500, num_control_points=num_cp, device='cuda')
        im = im.to('cpu')
        seq = seq.to('cpu')
        torch.save(im, basedir+"/Datasets/OneBezierDatasets/Training/images/fixedCP"+str(num_cp))
        torch.save(seq, basedir+"/Datasets/OneBezierDatasets/Training/sequences/fixedCP"+str(num_cp))

        im, seq = generate1bezier(batch_size=100, num_control_points=num_cp, device='cuda')
        im = im.to('cpu')
        seq = seq.to('cpu')
        torch.save(im, basedir+"/Datasets/OneBezierDatasets/Test/images/fixedCP"+str(num_cp))
        torch.save(seq, basedir+"/Datasets/OneBezierDatasets/Test/sequences/fixedCP"+str(num_cp))

    print("En generar tots els datasets hem trigat", time.time()-t0)
