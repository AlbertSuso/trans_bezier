import torch
import numpy as np

from ProbabilisticBezierEncoder.MultiBezierModels.FixedCP.dataset_generation import bezier
from Utils.curvature_penalization import curvature, acceleration_curvature
from Utils.repulsion import repulsion

def loss_function(control_points, im, distance_im, covariance, probabilistic_map_generator, grid): #, repulsion_coef=0.1, dist_thresh=4.5, second_term=True):
    batch_size = control_points.shape[2]
    num_cp = control_points.shape[1]

    probability_map = torch.empty((0, batch_size, 64, 64), dtype=torch.float32, device=control_points.device)
    pred_seq = torch.empty((batch_size, 0, 1, 1, 2), dtype=torch.float32, device=control_points.device)

    #curvature_penalizations = torch.empty((batch_size, 0), dtype=torch.float32, device=control_points.device)

    num_cps = num_cp*torch.ones(batch_size, dtype=torch.long, device=control_points.device)

    for bezier_cp in control_points:
        # Calculamos el mapa probabilistico de esta curva y lo concatenamos con los de las demás curvas
        partial_probability_map = probabilistic_map_generator(bezier_cp, num_cps, covariance)
        probability_map = torch.cat((probability_map, partial_probability_map.unsqueeze(0)), dim=0)

        # Calculamos la secuencia de puntos de esta curva y la concatenamos con las de las demás curvas
        partial_pred_seq = bezier(bezier_cp, num_cps,
                                  torch.linspace(0, 1, 150, device=num_cps.device).unsqueeze(0), device=num_cps.device)
        pred_seq = torch.cat((pred_seq, partial_pred_seq.unsqueeze(-2).unsqueeze(-2)), dim=1)

        # Calculamos la curvatura media o máxima de las curvas predichas y la almacenamos
        #new_curvatures = curvature(partial_pred_seq, mode='max')
        #curvature_penalizations = torch.cat((curvature_penalizations, new_curvatures.unsqueeze(1)), dim=1)

    # Calculamos los mapas probabilisticos y de distancias del conjunto de curvas
    pmap, _ = torch.max(probability_map, dim=0)
    dmap = torch.sqrt(torch.sum((grid - pred_seq) ** 2, dim=-1))
    dmap, _ = torch.min(dmap, dim=1)

    # Calculamos la penalización por curvatura
    #curvature_penalizations = torch.mean(curvature_penalizations)

    # Calculamos la fake chamfer_distance
    fake_chamfer = torch.sum(im[:, 0]*dmap/torch.sum(im[:, 0], dim=(1, 2)).view(-1, 1, 1)+pmap*distance_im[:, 0]/torch.sum(pmap, dim=(1, 2)).view(-1, 1, 1))/batch_size #+ curv_pen_coef*curvature_penalizations

    """repulsion_penalty = 0
    if repulsion_coef > 0:
        repulsion_penalty = repulsion(control_points, dist_thresh=dist_thresh, second_term=second_term)"""

    return fake_chamfer# + repulsion_coef*repulsion_penalty


def new_loss(pred_cp, groundtruth_im, groundtruth_seq, grid):
    """
    pred_cp.shape = (num_beziers, num_cp, batch_size, 2)
    groundtruth_seq.shape = (bs, max_N, 2)
    groundtruth_im.shape = (bs, 1, 64, 64)
    grid.shape = (1, 1, 64, 64, 2)

    As the training images will have different number of points belonging to the curves, max_N will be the highest number
    of points among all the dataset images. That means that for almost all the images we will need to padd the tensor
    groundtruth_seq (to fill the max_N coordinates), we will do the padding with a constant whose distance to any point
    of the canvas is high enough to don't be considered in the computation of the chamfer distance.
    """
    batch_size = pred_cp.shape[2]
    num_cp = pred_cp.shape[1]
    groundtruth_im = groundtruth_im[:, 0]

    # Computation of the sequence of coordinates of the predicted bézier curves
    num_cps = num_cp * torch.ones(batch_size, dtype=torch.long, device=pred_cp.device)
    pred_seq = torch.empty((batch_size, 0, 1, 2), dtype=torch.float32, device=pred_cp.device)
    for bezier_cp in pred_cp:
        partial_pred_seq = bezier(bezier_cp, num_cps,
                                  torch.linspace(0, 1, 150, device=num_cps.device).unsqueeze(0), device=num_cps.device)
        pred_seq = torch.cat((pred_seq, partial_pred_seq.unsqueeze(-2)), dim=1)


    """Computation of the first side of the chamfer distance sum(pred_im*dmap(original_im))/sum(pred_im)"""
    # pred_seq.shape = (bs, N, 1, 2)
    groundtruth_seq = groundtruth_seq.unsqueeze(1) #groundtruth_seq.shape = (bs, 1, max_N, 2)
    # Computation of the fake chamfer distance
    temp = torch.sqrt(torch.sum((pred_seq - groundtruth_seq) ** 2, dim=-1)) #temp.shape(bs, N, max_N)
    fact1 = torch.mean(torch.min(temp, dim=-1), dim=-1)

    """Computation of the second side of the chamfer distance sum(dmap(pred_im)*original_im)/sum(original_im)"""
    # Computation of the distance map
    dmap = torch.sqrt(torch.sum((grid - pred_seq.unsqueeze(-2)) ** 2, dim=-1))
    dmap, _ = torch.min(dmap, dim=1)
    fact2 = torch.sum(groundtruth_im * dmap, dim=(1, 2)) / torch.sum(groundtruth_im, dim=(1, 2))

    return torch.mean(fact1 + fact2)




