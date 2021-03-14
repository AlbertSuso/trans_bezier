import torch

from ProbabilisticBezierEncoder.MultiBezierModels.FixedCP.dataset_generation import bezier

def get_pmap_rewards(control_points, num_cp, num_beziers, im, loss_im, actual_covariances, probabilistic_map_generator):
    """
    control_points.shape = (num_cp*max_beziers, batch_size, 2)
    num_cp = scalar
    num_beziers.shape = (batch_size)
    actual_covariances.shape = (num_cp, batch_size, 2, 2)
    im.shape = (batch_size, 1, 64, 64)
    loss_im.shape = (batch_size 1, 64, 64)
    """
    batch_size = im.shape[0]

    probability_map = torch.empty((0, batch_size, 64, 64), device=im.device)
    pmap_rewards = torch.empty((0, batch_size), device=im.device)

    i = 0
    not_finished = num_beziers > i
    to_end = torch.sum(not_finished)
    while to_end:
        num_cps = torch.zeros_like(num_beziers)
        num_cps[not_finished] = num_cp
        partial_probability_map = probabilistic_map_generator(control_points[num_cp*i:num_cp*i+num_cp], num_cps, actual_covariances)

        probability_map = torch.cat((probability_map, partial_probability_map.unsqueeze(0)), dim=0)

        #Calculamos el reward obtenido después de dibujar esta curva
        reduced_pmap, _ = torch.max(probability_map, dim=0)
        new_rewards = torch.sum(reduced_pmap * loss_im[:, 0] / torch.sum(im[:, 0], dim=(1, 2)).view(-1, 1, 1), dim=(1, 2))
        pmap_rewards = torch.cat((pmap_rewards, new_rewards.unsqueeze(0)), dim=0)

        i += 1
        not_finished = num_beziers > i
        to_end = torch.sum(not_finished)

    return pmap_rewards


def get_dmap_rewards(control_points, num_cp, num_beziers, im, grid, distance='l2'):
    """
    control_points.shape = (num_cp*max_beziers, batch_size, 2)
    num_cp = scalar
    num_beziers.shape = (batch_size)
    im.shape = (batch_size, 1, 64, 64)
    grid.shape = (1, 1, 64, 64, 2)
    """
    batch_size = im.shape[0]
    pred_seq = torch.empty((batch_size, 0, 1, 1, 2), device=im.device)
    dmap_rewards = torch.empty((batch_size, 0), device=im.device)

    to_end = True
    i = 0
    while to_end:
        not_finished = num_beziers > i
        to_end = torch.sum(not_finished)

        num_cps = torch.zeros_like(num_beziers)
        num_cps[not_finished] = num_cp
        partial_pred_seq = bezier(control_points[num_cp*i:num_cp*i+num_cp], num_cps,
                                  torch.linspace(0, 1, 150, device=num_cps.device).unsqueeze(0), device=num_cps.device)
        pred_seq = torch.cat((pred_seq,  partial_pred_seq.unsqueeze(-2).unsqueeze(-2)), dim=1)

        # Calculamos el reward obtenido después de dibujar esta curva
        new_dmap = torch.sqrt(torch.sum((grid - pred_seq) ** 2, dim=-1))
        new_dmap, _ = torch.min(new_dmap, dim=1)
        if distance == 'quadratic':
            new_dmap = new_dmap * new_dmap
        elif distance == 'exp':
            new_dmap = torch.exp(new_dmap)
        new_rewards = -torch.sum(new_dmap * im[:, 0] / torch.sum(im[:, 0], dim=(1, 2)), dim=(1, 2))
        dmap_rewards = torch.cat((dmap_rewards, new_rewards), dim=1)

        i += 1

    return dmap_rewards

def get_chamfer_rewards(control_points, num_cp, num_beziers, im, distance_im, covariance, probabilistic_map_generator, grid):
    """
    control_points.shape = (num_cp*max_beziers, batch_size, 2)
    num_cp = scalar
    num_beziers.shape = (batch_size)
    actual_covariances.shape = (num_cp, batch_size, 2, 2)
    im.shape = (batch_size, 1, 64, 64)
    loss_im.shape = (batch_size 1, 64, 64)
    """
    batch_size = im.shape[0]

    probability_map = torch.empty((0, batch_size, 64, 64), dtype=torch.float32, device=im.device)
    pred_seq = torch.empty((batch_size, 0, 1, 1, 2), dtype=torch.float32, device=im.device)
    chamfer_rewards = torch.empty((0, batch_size), dtype=torch.float32, device=im.device)

    i = 0
    not_finished = num_beziers > i
    to_end = torch.sum(not_finished)
    while to_end:
        num_cps = torch.zeros_like(num_beziers)
        num_cps[not_finished] = num_cp

        partial_probability_map = probabilistic_map_generator(control_points[num_cp*i:num_cp*i+num_cp], num_cps, covariance)
        probability_map = torch.cat((probability_map, partial_probability_map.unsqueeze(0)), dim=0)

        partial_pred_seq = bezier(control_points[num_cp * i:num_cp * i + num_cp], num_cps,
                                  torch.linspace(0, 1, 150, device=num_cps.device).unsqueeze(0), device=num_cps.device)
        pred_seq = torch.cat((pred_seq, partial_pred_seq.unsqueeze(-2).unsqueeze(-2)), dim=1)

        #Calculamos el reward obtenido después de dibujar esta curva
        pmap, _ = torch.max(probability_map, dim=0)
        dmap = torch.sqrt(torch.sum((grid - pred_seq) ** 2, dim=-1))
        dmap, _ = torch.min(dmap, dim=1)


        new_rewards = -torch.sum(im[:, 0]*dmap/torch.sum(im[:, 0], dim=(1, 2)).view(-1, 1, 1)+pmap*distance_im[:, 0]/torch.sum(pmap, dim=(1, 2)).view(-1, 1, 1), dim=(1, 2))
        chamfer_rewards = torch.cat((chamfer_rewards, new_rewards.unsqueeze(0)), dim=0)

        i += 1
        not_finished = num_beziers > i
        to_end = torch.sum(not_finished)

    return chamfer_rewards



def loss_function(epoch, control_points, num_beziers, probabilities, num_cp, im, distance_im, loss_im, grid, actual_covariances,
         probabilistic_map_generator, loss_type='pmap', distance='l2', gamma=0.9):
    if loss_type == 'pmap':
        rewards = get_pmap_rewards(control_points, num_cp, num_beziers, im, loss_im, actual_covariances, probabilistic_map_generator)
    elif loss_type == 'dmap':
        rewards = get_dmap_rewards(control_points, num_cp, num_beziers, im, grid, distance=distance)
    else:
        rewards = get_chamfer_rewards(control_points, num_cp, num_beziers, im, distance_im, actual_covariances, probabilistic_map_generator, grid)

    # Calculamos los difference rewards y el cummulative_reward
    difference_rewards = rewards
    cummulative_reward = 0

    for i in range(difference_rewards.shape[0]-1, 0, -1):
        difference_rewards[i] = difference_rewards[i] - difference_rewards[i-1]
        cummulative_reward = gamma*cummulative_reward + difference_rewards[i]
    cummulative_reward = gamma*cummulative_reward + difference_rewards[0]

    model_loss = -cummulative_reward
    if epoch <= 50:
        return torch.mean(model_loss)

    reinforcement_loss = -torch.sum(difference_rewards.detach() * torch.log(probabilities), dim=0)
    return torch.mean(model_loss + reinforcement_loss)


