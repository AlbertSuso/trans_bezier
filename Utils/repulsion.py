import torch

def repulsion(control_points, dist_thresh=4.5, second_term=True):
    """
    control_points.shape = (num_beziers, batch_size, num_cp, 2)
    """
    rep = 0
    for bez_i in range(control_points.shape[0]):
        for bez_j in range(bez_i+1, control_points.shape[0]):
            for cp_i in range(control_points.shape[2]):
                for cp_j in range(control_points.shape[2]):
                    resta = control_points[bez_i, :, cp_i] - control_points[bez_j, :, cp_j]
                    dist = torch.sum(resta * resta, dim=-1)
                    maxx = torch.max(torch.zeros_like(dist), 1 - dist / dist_thresh)
                    if second_term:
                        rep += maxx * (1 - dist / dist_thresh)
                    else:
                        rep += maxx
    return torch.mean(rep)