import torch
import torch.nn as nn
import torch.nn.functional as F

from ProbabilisticBezierEncoder.OneBezierModels.FixedCP.dataset_generation import bezier


def chamfer_loss(control_points, num_cps, im, grid):
    """
    control_points.shape=(max_num_cp, batch_size, 2)
    num_cps.shape=(batch_size,)
    im.shape=(batch_size, 1, 64, 64)
    distance_im.shape=(batch_size, 1, 64, 64)
    grid.shape=(1, 1, 64, 64, 2) (donde grid[:, :, x, y, :] = [x, y])
    """
    batch_size = im.shape[0]

    # Obtenemos las secuencias de puntos de las curvas de bezier predichas
    # pred_seq.shape=(batch_size, resolution, 2)
    pred_seq = bezier(control_points, num_cps,
                      torch.linspace(0, 1, 150, device=num_cps.device).unsqueeze(0), device=num_cps.device)


    """# Renderizamos las curvas de bezier predichas
    pred_images = torch.zeros_like(im)
    for i in range(batch_size):
        pred_images[i, 0, pred_seq[i, :, 0], pred_seq[i, :, 1]] = 1"""

    # Creamos los mapas de distancias de las curvas predichas
    pred_seq = pred_seq.unsqueeze(-2).unsqueeze(-2)
    distance_map = torch.sqrt(torch.sum((grid-pred_seq)**2, dim=-1))
    distance_map, _ = torch.min(distance_map, dim=1)

    return distance_map


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt

    basedir = "/home/asuso/PycharmProjects/trans_bezier/Datasets/OneBezierDatasets/Training"
    num_cp = 3
    images = torch.load(os.path.join(basedir, "images/fixedCP"+str(num_cp)))
    sequences = torch.load(os.path.join(basedir, "sequences/fixedCP"+str(num_cp)))

    batch_size = 1
    idx = 190
    im = images[idx:idx+batch_size].cuda()
    seq = sequences[:-1, idx:idx+batch_size]

    control_points = torch.empty((num_cp, batch_size, 2), device=im.device)
    for i in range(num_cp):
        control_points[i, :, 0] = seq[i] // 64
        control_points[i, :, 1] = seq[i] % 64

    grid = torch.empty((1, 1, im.shape[2], im.shape[3], 2), device=im.device, dtype=torch.float32)
    for i in range(im.shape[2]):
        grid[0, 0, i, :, 0] = i
        grid[0, 0, :, i, 1] = i

    distance_map = chamfer_loss(control_points, num_cp * torch.ones(batch_size, dtype=torch.long, device='cuda'), im, grid)
    recovered_image = torch.zeros_like(distance_map)
    recovered_image[distance_map == 0] = 1

    print("La distancia minima es", torch.min(distance_map))
    print("La distancia maxima es", torch.max(distance_map))
    print("La distancia media es", torch.mean(distance_map))
    print("La distancia mediana es", torch.median(distance_map))
    print("La proporción de pixeles por debajo de 1 es", torch.sum(distance_map < 1)/(64*64))

    plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(im[0, 0].cpu(), cmap='gray')
    plt.title("Original image")
    plt.subplot(1, 4, 2)
    plt.imshow(distance_map[0].cpu(), cmap='gray')
    plt.title("Distance map")
    plt.subplot(1, 4, 3)
    plt.imshow((distance_map[0]*distance_map[0]).cpu(), cmap='gray')
    plt.title("Quadratic map")
    plt.subplot(1, 4, 4)
    plt.imshow(torch.exp(distance_map[0]).cpu(), cmap='gray')
    plt.title("Exponential map")
    plt.show()