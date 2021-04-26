import torch

def repusion(control_points, canvas_size):
    """
    control_points.shape = (num_beziers, batch_size, num_cp, 2)
    """
    norm_coef = canvas_size*canvas_size
    rep = 0
    for l in range(control_points.shape[0]):
        for m in range(control_points.shape[0]):
            for i in range(control_points.shape[2]):
                for j in range(control_points.shape[2]):
                    resta = control_points[l, :, i] - control_points[m, :, j]
                    dist = torch.sum(resta*resta, dim=-1)
                    maxx, _ = torch.max(torch.zeros_like(dist), 1-dist/norm_coef)
                    rep += maxx.unsqueeze(-1)*resta/norm_coef
    return torch.sum(rep*rep)/control_points.shape[1]