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

def generate1bezier(im_size=64, batch_size=64, max_beziers = 2, num_control_points=3, resolution=150, device='cuda'):
    images = torch.zeros((batch_size, 1, im_size, im_size), dtype=torch.float32, device=device)
    tgt_seq = torch.ones((max_beziers*(num_control_points+1), batch_size), dtype=torch.long, device=device)
    tgt_padding_mask = torch.ones((batch_size, max_beziers*(num_control_points+1))).bool()

    # Rellenamos las tgt_seq con EOS tokens
    tgt_seq *= (im_size*im_size+1)

    # Generamos aleatoriamente los control points de las curvas del dataset
    control_points = (im_size-0.5)*torch.rand((max_beziers*(num_control_points+1), batch_size, 2), device=device)

    # Generamos aleatoriamente cuantas curvas de bezier tendra cada imagen
    num_bezier_curves = torch.randint(1, max_beziers+1, (batch_size,), device=device)

    #Generamos la tgt_seq y la tgt_padding_mask
    rounded_cp = torch.round(control_points)
    for i, num_bezier in enumerate(num_bezier_curves):
        # Hacemos la padding_mask de esta imagen
        tgt_padding_mask[i, :num_bezier * (num_control_points + 1) + 1] = False

        #Hacemos la tgt_seq de esta imagen
        tgt_seq[:num_bezier*(num_control_points+1), i] = im_size * rounded_cp[:num_bezier*(num_control_points+1), i, 0] + rounded_cp[:num_bezier*(num_control_points+1), i, 1]
        # Añadimos los BOS tokens a la tgt_seq
        for j in range(num_control_points, (num_bezier-1)*(num_control_points+1), num_control_points+1):
            tgt_seq[j, i] = im_size*im_size

    for i in range(max_beziers):
        # Seleccionamos las imagenes que han de tener curva i-esima
        obj_images = num_bezier_curves > i
        # Seleccionamos los control points correspondientes a la curva i-esima de las imagenes objetivo
        cps = control_points[(num_control_points+1)*i:(num_control_points+1)*i+num_control_points, obj_images]

        # Generamos la secuencia de pixels de las curvas de bezier asociadas
        output = bezier(cps, num_control_points*torch.ones(torch.sum(obj_images), dtype=torch.long, device=device), torch.linspace(0, 1, resolution, device=device).unsqueeze(0), device=device)

        # Pintamos las curvas en las imagenes correspondientes
        idx = 0
        for j in range(batch_size):
            if obj_images[j]:
                images[j, 0, output[idx, :, 0], output[idx, :, 1]] = 1
                idx += 1


    return images, tgt_seq, tgt_padding_mask


if __name__ == '__main__':
    # basedir = "/data2fast/users/asuso"
    basedir = "/home/albert/PycharmProjects/trans_bezier"

    t0 = time.time()
    max_beziers = 2
    for num_cp in [3]:
        im, seq, tgt_padding_mask = generate1bezier(im_size=64, batch_size=50000, max_beziers=max_beziers, num_control_points=num_cp, device='cuda')
        im = im.to('cpu')
        seq = seq.to('cpu')
        tgt_padding_mask = tgt_padding_mask.to('cpu')
        torch.save(im, basedir + "/Datasets/MultiBezierDatasets/Training/images/fixedCP"+str(num_cp)+"_maxBeziers"+str(max_beziers))
        torch.save(seq, basedir + "/Datasets/MultiBezierDatasets/Training/sequences/fixedCP"+str(num_cp)+"_maxBeziers"+str(max_beziers))
        torch.save(tgt_padding_mask,
                   basedir + "/Datasets/MultiBezierDatasets/Training/padding_masks/fixedCP"+str(num_cp)+"_maxBeziers"+str(max_beziers))

        im, seq, tgt_padding_mask =generate1bezier(im_size=64, batch_size=10000, max_beziers=max_beziers, num_control_points=num_cp, device='cuda')
        im = im.to('cpu')
        seq = seq.to('cpu')
        tgt_padding_mask = tgt_padding_mask.to('cpu')
        torch.save(im, basedir + "/Datasets/MultiBezierDatasets/Test/images/fixedCP"+str(num_cp)+"_maxBeziers"+str(max_beziers))
        torch.save(seq, basedir + "/Datasets/MultiBezierDatasets/Test/sequences/fixedCP"+str(num_cp)+"_maxBeziers"+str(max_beziers))
        torch.save(tgt_padding_mask, basedir + "/Datasets/MultiBezierDatasets/Test/padding_masks/fixedCP"+str(num_cp)+"_maxBeziers"+str(max_beziers))

    print("En generar tots els datasets hem trigat", time.time()-t0)
