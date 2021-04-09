import torch

def curvature(seq, mode='max'):
    """
    seq.shape=(batch_size, seq_len, 2)
    """
    second_derivatives = seq[:, 2:] - 2*seq[:, 1:-1] + seq[:, -2]
    curvatures = torch.sum(second_derivatives*second_derivatives, dim=-1)
    if mode == 'max':
        return torch.max(curvatures, dim=-1)
    return torch.mean(curvatures, dim=-1)

def acceleration_curvature(seq, mode='max'):
    """
        seq.shape=(batch_size, seq_len, 2)
    """
    fourth_derivatives = seq[:, 4:] - 4*seq[:, 3:-1] + 6*seq[:, 2:-2] - 4*seq[:, 1:-3] + seq[:, :-4]
    curvatures = torch.sum(fourth_derivatives * fourth_derivatives, dim=-1)
    if mode == 'max':
        return torch.max(curvatures, dim=-1)
    return torch.mean(curvatures, dim=-1)