import torch
import numpy as np

from ProbabilisticBezierEncoder.MultiBezierModels.FixedCP.dataset_generation import bezier
from Utils.curvature_penalization import curvature, acceleration_curvature
# from Utils.repulsion import repulsion

def loss_function(control_points, im, distance_im, covariance, probabilistic_map_generator, grid):
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
    return fake_chamfer



