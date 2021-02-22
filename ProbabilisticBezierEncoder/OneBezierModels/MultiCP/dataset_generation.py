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
    return output

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
    output = torch.round(output).long()

    for i in range(batch_size):
        images[i, 0, output[i, :, 0], output[i, :, 1]] = 1

    return images, tgt_seq, tgt_padding_mask



if __name__ == '__main__':
    import matplotlib.pyplot as plt


    # basedir = "/data2fast/users/asuso"
    basedir = "/home/asuso/PycharmProjects/trans_bezier"

    t0 = time.time()
    for num_cp in [5]:
        im_t, seq_t, tgt_padding_mask_t = generate1bezier(batch_size=100000, max_control_points=num_cp, device='cuda')
        im_t = im_t.to('cpu')
        seq_t = seq_t.to('cpu')
        tgt_padding_mask_t = tgt_padding_mask_t.to('cpu')
        torch.save(im_t, basedir+"/Datasets/OneBezierDatasets/Training/images/multiCP"+str(num_cp)+"_larger")
        torch.save(seq_t, basedir+"/Datasets/OneBezierDatasets/Training/sequences/multiCP"+str(num_cp)+"_larger")
        torch.save(tgt_padding_mask_t, basedir+"/Datasets/OneBezierDatasets/Training/padding_masks/multiCP"+str(num_cp)+"_larger")

        im_v, seq_v, tgt_padding_mask_v = generate1bezier(batch_size=20000, max_control_points=num_cp, device='cuda')
        im_v = im_v.to('cpu')
        seq_v = seq_v.to('cpu')
        tgt_padding_mask_v = tgt_padding_mask_v.to('cpu')
        torch.save(im_v, basedir+"/Datasets/OneBezierDatasets/Test/images/multiCP"+str(num_cp)+"_larger")
        torch.save(seq_v, basedir+"/Datasets/OneBezierDatasets/Test/sequences/multiCP"+str(num_cp)+"_larger")
        torch.save(tgt_padding_mask_v, basedir+"/Datasets/OneBezierDatasets/Test/padding_masks/multiCP"+str(num_cp)+"_larger")

    print("En generar tots els datasets hem trigat", time.time()-t0)

    print("El porcentaje medio de pixeles pintados es de", torch.sum(im_t) / (50000 * 64 * 64))

    """idx = 40000
    for i in range(0, 200, 20):
        im = im_t[idx + i].unsqueeze(0).cuda()
        padding_mask = tgt_padding_mask_t[idx + i]
        num_cp = padding_mask.shape[0] - torch.sum(padding_mask) - 1
        tgt_seq = seq_t[:num_cp, idx + i].cuda()

        tgt_im = torch.empty((3, 64, 64))
        tgt_im[:] = im[0]

        tgt_control_points = torch.empty((tgt_seq.shape[0], 2))
        for i, cp in enumerate(tgt_seq):
            tgt_control_points[i, 0] = cp // 64
            tgt_control_points[i, 1] = cp % 64

        for cp_tgt in tgt_control_points:
            tgt_im[:, int(cp_tgt[0]), int(cp_tgt[1])] = 0
            tgt_im[0, int(cp_tgt[0]), int(cp_tgt[1])] = 1

        plt.imshow(tgt_im.transpose(0, 1).transpose(1, 2))
        plt.title("Target\n{}".format(tgt_control_points))
        plt.show()"""
