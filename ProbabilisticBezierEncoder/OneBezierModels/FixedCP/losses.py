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
    distance_map = distance_map/torch.max(distance_map)
    return torch.sum(distance_map * im[:, 0]/torch.sum(im[:, 0], dim=(1, 2))) / im.shape[0]

def chamfer_loss(control_points, num_cps, im, distance_im, grid):
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

    # Renderizamos las curvas de bezier predichas
    pred_images = torch.zeros_like(im)
    for i in range(batch_size):
        pred_images[i, 0, pred_seq[i, :, 0], pred_seq[i, :, 1]] = 1

    # Creamos los mapas de distancias de las curvas predichas
    pred_seq = pred_seq.unsqueeze(-2).unsqueeze(-2)
    distance_map = torch.sqrt(torch.sum((grid-pred_seq)**2, dim=-1))
    distance_map, _ = torch.min(distance_map, dim=1)

    return torch.sum(im[:, 0]*distance_map/torch.sum(im[:, 0], dim=(1, 2)).view(-1, 1, 1) + pred_images[:, 0]*distance_im[:, 0]/torch.sum(pred_images[:, 0], dim=(1, 2)).view(-1, 1, 1)) / batch_size