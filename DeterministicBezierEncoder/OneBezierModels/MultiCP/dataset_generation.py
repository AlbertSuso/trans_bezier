import os
import time
import torch

from scipy.special import binom
from torch.utils.data import Dataset


class OneBezierDataset(Dataset):
    def __init__(self, basedir="/home/asuso/PycharmProjects/transformer_bezier/Datasets", training=True):
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
        pot = (num_cps-1-i).unsqueeze(1)
        pot[pot < 0] = 0
        output = output + (binomial_coefs[i].unsqueeze(1) * t**i * t_inv**pot).unsqueeze(-1) * P.unsqueeze(1)
    return torch.round(output).long()

def generate1bezier(im_size=64, batch_size=64, max_control_points=3, resolution=150, device='cuda'):
    images = torch.zeros((batch_size, 1, im_size, im_size), dtype=torch.float32, device=device)
    tgt_seq = torch.ones((max_control_points+1, batch_size), dtype=torch.long, device=device)
    tgt_padding_mask = torch.ones((batch_size, max_control_points+1), dtype=torch.long, device=device).bool()

    # El ultimo token de todas las secuencias será el que indica que debemos parar
    tgt_seq *= im_size*im_size

    # Generamos aleatoriamente los control points de las curvas del dataset
    control_points = (im_size-0.5)*torch.rand((max_control_points, batch_size, 2), device=device)
    # Escogemos aleatoriamente cuantos puntos de control tendra cada curva
    num_cps = torch.randint(2, max_control_points + 1, (batch_size,), dtype=torch.long, device=device)

    # Generamos la tgt_seq y la tgt_padding_mask
    rounded_cp = torch.round(control_points)
    for i, num_cp in enumerate(num_cps):
        tgt_seq[:num_cp, i] = im_size * rounded_cp[:num_cp, i, 0] + rounded_cp[:num_cp, i, 1]
        tgt_padding_mask[i, :num_cp+1] = False

    output = bezier(control_points, num_cps, torch.linspace(0, 1, resolution, device=device).unsqueeze(0), device=device)

    for i in range(batch_size):
        images[i, 0, output[i, :, 0], output[i, :, 1]] = 1

    return images, tgt_seq, tgt_padding_mask



if __name__ == '__main__':
    # basedir = "/data2fast/users/asuso"
    basedir = "/home/asuso/PycharmProjects/trans_bezier"

    t0 = time.time()
    for num_cp in [4]:
        im, seq, tgt_padding_mask = generate1bezier(batch_size=50000, max_control_points=num_cp, device='cuda')
        im = im.to('cpu')
        seq = seq.to('cpu')
        tgt_padding_mask = tgt_padding_mask.to('cpu')
        torch.save(im, basedir+"/Datasets/OneBezierDatasets/Training/images/multiCP"+str(num_cp))
        torch.save(seq, basedir+"/Datasets/OneBezierDatasets/Training/sequences/multiCP"+str(num_cp))
        torch.save(tgt_padding_mask, basedir+"/Datasets/OneBezierDatasets/Training/padding_masks/multiCP"+str(num_cp))

        im, seq, tgt_padding_mask  = generate1bezier(batch_size=10000, max_control_points=num_cp, device='cuda')
        im = im.to('cpu')
        seq = seq.to('cpu')
        tgt_padding_mask = tgt_padding_mask.to('cpu')
        torch.save(im, basedir+"/Datasets/OneBezierDatasets/Test/images/multiCP"+str(num_cp))
        torch.save(seq, basedir+"/Datasets/OneBezierDatasets/Test/sequences/multiCP"+str(num_cp))
        torch.save(tgt_padding_mask, basedir+"/Datasets/OneBezierDatasets/Test/padding_masks/multiCP"+str(num_cp))

    print("En generar tots els datasets hem trigat", time.time()-t0)
