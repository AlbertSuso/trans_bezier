import torch

from ProbabilisticBezierEncoder.OneBezierModels.FixedCP.dataset_generation import bezier


def pmap_loss(control_points, num_cps, actual_covariances, im, loss_im, probabilistic_map_generator, mode='p'):
    batch_size = im.shape[0]

    probability_map = probabilistic_map_generator(control_points, num_cps, actual_covariances, mode=mode)
    return -torch.sum(probability_map * loss_im[:, 0] / torch.sum(im[:, 0], dim=(1, 2)).view(-1, 1, 1))/batch_size


def dmap_loss(control_points, num_cps, im, grid, distance='l2'):
    pred_seq = bezier(control_points, num_cps,
                      torch.linspace(0, 1, 150, device=num_cps.device).unsqueeze(0), device=num_cps.device)

    pred_seq = pred_seq.unsqueeze(-2).unsqueeze(-2)
    distance_map = torch.sum((grid - pred_seq) ** 2, dim=-1)
    distance_map, _ = torch.min(distance_map, dim=1)

    if distance == 'l2':
        distance_map = torch.sqrt(distance_map)

    # normalizamos el mapa de distancias
    # distance_map = distance_map/torch.max(distance_map)
    return torch.sum(distance_map * im[:, 0]/torch.sum(im[:, 0], dim=(1, 2))) / im.shape[0]

def chamfer_loss(control_points, num_cps, im, distance_im, covariance, probabilistic_map_generator, grid):
    """
    control_points.shape=(max_num_cp, batch_size, 2)
    num_cps.shape=(batch_size,)
    im.shape=(batch_size, 1, 64, 64)
    distance_im.shape=(batch_size, 1, 64, 64)
    grid.shape=(1, 1, 64, 64, 2) (donde grid[:, :, x, y, :] = [x, y])
    """
    batch_size = im.shape[0]

    pmap = probabilistic_map_generator(control_points, num_cps, covariance, mode='p')

    pred_seq = bezier(control_points, num_cps,
                      torch.linspace(0, 1, 150, device=num_cps.device).unsqueeze(0), device=num_cps.device)
    pred_seq = pred_seq.unsqueeze(-2).unsqueeze(-2)
    distance_map = torch.sum((grid - pred_seq) ** 2, dim=-1)
    dmap, _ = torch.min(distance_map, dim=1)
    dmap = torch.sqrt(dmap)

    return torch.sum(im[:, 0]*dmap/torch.sum(im[:, 0], dim=(1, 2)).view(-1, 1, 1)+pmap*distance_im[:, 0]/torch.sum(pmap, dim=(1, 2)).view(-1, 1, 1))/batch_size

    """LO QUE HAY DESPUES DE ESTO SE TIENE QUE ELIMINAR
    im_seq = torch.round(pred_seq).long()
    predicted_im = torch.zeros_like(im)
    for i in range(batch_size):
        predicted_im[i, 0, im_seq[i, :, 0, 0, 0], im_seq[i, :, 0, 0, 1]] = 1
    true_chamfer = torch.sum(im[:, 0]*dmap/torch.sum(im[:, 0], dim=(1, 2)).view(-1, 1, 1) + predicted_im[:, 0]*distance_im[:, 0]/torch.sum(predicted_im[:, 0], dim=(1, 2)).view(-1, 1, 1)) / batch_size

    return true_chamfer, torch.sum(im[:, 0]*dmap/torch.sum(im[:, 0], dim=(1, 2)).view(-1, 1, 1) + pmap*distance_im[:, 0]/torch.sum(pmap, dim=(1, 2)).view(-1, 1, 1)) / batch_size"""


if __name__ == '__main__':
    import torch
    import os

    from Utils.chamfer_distance import generate_distance_images
    from Utils.probabilistic_map import ProbabilisticMap

    basedir = "/home/asuso/PycharmProjects/trans_bezier"

    num_cp = 5
    batch_size = 1

    images = torch.load(os.path.join(basedir, "Datasets/OneBezierDatasets/Training/images/fixedCP" + str(num_cp)))
    sequences = torch.load(os.path.join(basedir, "Datasets/OneBezierDatasets/Training/sequences/fixedCP" + str(num_cp)))
    distance_images = generate_distance_images(images, distance='l2')

    # Inicializamos el generador de mapas probabilisticos y la matriz de covariancias para el pmap
    probabilistic_map_generator = ProbabilisticMap((64, 64, 50))
    cp_covariance = torch.tensor([[[1, 0], [0, 1]] for i in range(num_cp)], dtype=torch.float32)
    cp_covariances = cp_covariance.unsqueeze(1)
    # Iniciamos el grid para el dmap
    grid = torch.empty((1, 1, images.shape[2], images.shape[3], 2), dtype=torch.float32)
    for i in range(64):
        grid[0, 0, i, :, 0] = i
        grid[0, 0, :, i, 1] = i


    relative_error = 0
    absolute_error = 0
    idx = 157
    for i in range(1000):
        im = images[idx].unsqueeze(0)
        distance_im = distance_images[idx].unsqueeze(0)
        tgt_seq = sequences[:-1, idx].unsqueeze(1)
        cps = torch.empty((num_cp, 1, 2))
        for i, cp in enumerate(tgt_seq):
            cps[i, 0, 0] = cp // 64
            cps[i, 0, 1] = cp % 64

        true_chamfer, loss_chamfer = chamfer_loss(cps, num_cp+torch.zeros(batch_size, dtype=torch.long, device=cps.device), im, distance_im, cp_covariances, probabilistic_map_generator, grid)

        relative_error += torch.abs(true_chamfer-loss_chamfer)/true_chamfer
        absolute_error += torch.abs(true_chamfer - loss_chamfer)



    print("El error relativo medio entre ambas distancias es", relative_error/1000)
    print("El error absoluto medio entre ambas distancias es", absolute_error/1000)