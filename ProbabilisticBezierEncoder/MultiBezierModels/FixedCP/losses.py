import torch

from ProbabilisticBezierEncoder.MultiBezierModels.FixedCP.dataset_generation import bezier

def get_pmap_rewards(control_points, num_cp, num_beziers, im, loss_im, actual_covariances, probabilistic_map_generator,
                     distance='l2', gamma=0.9):
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

    to_end = True
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
        if distance == 'quadratic':
            reduced_pmap = reduced_pmap * reduced_pmap
        elif distance == 'exp':
            reduced_pmap = torch.exp(reduced_pmap)
        new_rewards = torch.sum(reduced_pmap * loss_im[:, 0] / torch.sum(im[:, 0], dim=(1, 2)).view(-1, 1, 1), dim=(1, 2))
        pmap_rewards = torch.cat((pmap_rewards, new_rewards.unsqueeze(0)), dim=0)

        i += 1
        not_finished = num_beziers > i
        to_end = torch.sum(not_finished)

    # Calculamos los cummulative rewards de cada curva
    cummulative_rewards = pmap_rewards
    cummulative_rewards[:, -1] -= cummulative_rewards[:, -2]
    for i in range(cummulative_rewards.shape[1] - 2, 0, -1):
        cummulative_rewards[:, i] += -cummulative_rewards[:, i - 1] + gamma * cummulative_rewards[:, i + 1]
    cummulative_rewards[:, 0] += gamma * cummulative_rewards[:, 1]

    return cummulative_rewards.permute(1, 0) #shape=(batch_size, max_beziers)


def get_dmap_rewards(control_points, num_cp, num_beziers, im, grid, distance='l2', gamma=0.9):
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

    # Calculamos los cummulative rewards de cada curva
    cummulative_rewards = dmap_rewards
    cummulative_rewards[:, -1] -= cummulative_rewards[:, -2]
    for i in range(cummulative_rewards.shape[1] - 2, 0, -1):
        cummulative_rewards[:, i] += -cummulative_rewards[:, i - 1] + gamma * cummulative_rewards[:, i + 1]
    cummulative_rewards[:, 0] += gamma * cummulative_rewards[:, 1]

    return cummulative_rewards  # shape=(batch_size, max_beziers)


def loss_function(control_points, num_beziers, probabilities, num_cp, im, loss_im, grid, actual_covariances,
         probabilistic_map_generator, map_type='pmap', distance='l2', gamma=0.9):
    if map_type == 'pmap':
        cummulative_rewards = get_pmap_rewards(control_points, num_cp, num_beziers, im, loss_im,
                                               actual_covariances, probabilistic_map_generator, distance=distance, gamma=gamma)
    else:
        cummulative_rewards = get_dmap_rewards(control_points, num_cp, num_beziers, im, grid,
                                               distance=distance, gamma=gamma)

    model_loss = -cummulative_rewards[:, 0]
    reinforcement_loss = -torch.sum(cummulative_rewards.detach() * torch.log(probabilities.permute(1, 0)), dim=-1)

    return torch.mean(model_loss + reinforcement_loss)


