import torch
import torch.nn as nn
import numpy as np

from scipy.special import binom


class ProbabilisticMap(nn.Module):
    def __init__(self, map_sizes=(64, 64, 50)):
        super().__init__()
        self._map_width = map_sizes[0]
        self._map_height = map_sizes[1]
        self._map_temporalSize = map_sizes[2]

        positions = torch.empty((1, self._map_width, self._map_height, 1, 2), dtype=torch.float32)
        for x in range(self._map_width):
            positions[:, x, :, 0, 0] = x
        for y in range(self._map_height):
            positions[:, :, y, 0, 1] = y
        self.register_buffer('positions', positions)

    def _normal_dist_parameters(self, t, cp_means, cp_covariances):
        """
        Calcula la media y la matriz de covariancias de la distribucion normal que simean = torch.zeros_like(cp_means[0])gue la curva de bezier
        probabilistica en los tiempos del array de tiempos "t"

        cp_means.shape=(num_cp, batch_size, cp_size)
        cp_covariances.shape = (num_cp, batch_size, cp_size, cp_size)
        t.shape = (temporal_size)
        """
        # Añadimos una dimension para el tiempo.
        cp_means = cp_means.unsqueeze(2)
        cp_means = cp_means.repeat(1, 1, self._map_temporalSize, 1)
        cp_covariances = cp_covariances.unsqueeze(2)
        cp_covariances = cp_covariances.repeat(1, 1, self._map_temporalSize, 1, 1)
        # Adaptamos la shape de t para poder vectorizar las operaciones
        t = t.view(1, -1, 1)

        N = cp_means.shape[0] - 1
        t_inv = 1 - t

        mean = torch.zeros_like(cp_means[0])
        covariance = torch.zeros_like(cp_covariances[0])

        for i, (cp_mean, cp_covariance) in enumerate(zip(cp_means, cp_covariances)):
            mean += binom(N, i) * t ** i * t_inv ** (N - i) * cp_mean
            covariance += (binom(N, i) * t.unsqueeze(-1) ** i * t_inv.unsqueeze(-1) ** (N - i)) ** 2 * cp_covariance

        return mean, covariance

    def normal2d(self, mean, covariance):
        """
        Devuelve un batch de mapas 3D de shape (64, 64, temporal_size), siendo cada slice (64, 64, t)
        la distribucion de probabilidades de la curva de bezier en tiempo t.

        mean.shape = (batch_size, cp_size)
        cov.shape = (batch_size, cp_size, cp_size)
        """
        # TODOS LOS UNSQUEEZE SON PARA QUE LAS DIMENSIONES CUADREN Y SE PUEDAN VECTORIZAR LAS OPERACIONES
        # Esto es debido a que p.shape=(1, 64, 64, 2). Por tanto las 3 primeras componentes de cada tensor son
        # batch_size, grid_height y grid_width respectivamente. En realidad las operaciones queremos hacerlas
        # sobre las dos ultimas componentes, y acabar obteniendo un mapa con map.shape=(batch_size, 64, 64)
        mean = mean.unsqueeze(1).unsqueeze(1)
        cov_inv = torch.inverse(covariance).unsqueeze(1).unsqueeze(1)
        divisor = torch.sqrt((2 * np.pi) ** 2 * torch.det(covariance)).unsqueeze(1).unsqueeze(1)

        up = torch.matmul(self.positions.unsqueeze(-2) - mean.unsqueeze(-2), cov_inv)
        up = torch.matmul(up, self.positions.unsqueeze(-1) - mean.unsqueeze(-1))
        up = torch.exp(-0.5 * up)
        return up[:, :, :, :, 0, 0]/divisor


    def forward(self, cp_means, cp_covariances):
        """
        A partir de las medias de los puntos de control y sus matrices de covariancias, genera un batch de mapas
        probabilisticos de tamaño (batch_size, image_height, image_width, temporal_size).
        En concreto, cada mapa 2D (i, image_height, image_width, t) es la distribucion de probabilidades de la
        curva de bezier "i" en el tiempo "t".

        cp_means.shape = (num_cp, batch_size, 2)
        cp_covariances.shape = (num_cp, batch_size, 2, 2)
        """
        mean, covariance = self._normal_dist_parameters(torch.linspace(0, 1, self._map_temporalSize, device=cp_means.device),
                                                        cp_means, cp_covariances)
        return self.normal2d(mean, covariance)


if __name__ == '__main__':
    import os
    import time
    import matplotlib.pyplot as plt

    basedir = "/home/albert/PycharmProjects/trans_bezier/Datasets/OneBezierDatasets/Training"
    images = torch.load(os.path.join(basedir, "images/fixedCP3"))
    sequences = torch.load(os.path.join(basedir, "sequences/fixedCP3"))

    batch_size = 64
    idx = 170
    im = images[idx:idx+batch_size]
    seq = sequences[:-1, idx:idx+batch_size]

    #padding_mask = padding_masks[idx]
    #num_cp = padding_mask.shape[0] - torch.sum(padding_mask) - 1
    num_cp = 3

    cp_means = torch.empty((num_cp, batch_size, 2))

    for i in range(num_cp):
        cp_means[i, :, 0] = seq[i] // 64
        cp_means[i, :, 1] = seq[i] % 64

    cp_covariance = torch.tensor([[[3, 0], [0, 3]] for i in range(num_cp)], dtype=torch.float32)
    cp_covariances = torch.empty((num_cp, batch_size, 2, 2))
    for i in range(batch_size):
        cp_covariances[:, i, :, :] = cp_covariance

    cp_means = cp_means.cuda()
    cp_covariances = cp_covariances.cuda()
    map_maker = ProbabilisticMap(map_sizes=(64, 64, 50)).cuda()

    print("AQUI EMPIEZA DE VERDAD LA PRUEBA")
    print("cp_means.shape=", cp_means.shape)
    print("cp_covariances.shape=", cp_covariances.shape)
    t0 = time.time()
    map = map_maker(cp_means, cp_covariances)
    print("map.shape=", map.shape)
    reduced_map, _ = torch.max(map, dim=3)
    reduced_map = reduced_map/torch.max(reduced_map)
    print("reduced_map.shape=", reduced_map.shape)

    print("Tiempo transcurrido:", time.time()-t0)

    for i in range(10):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(im[i, 0], cmap='gray')
        plt.title("Deterministic Image")
        plt.subplot(1, 2, 2)
        plt.imshow(reduced_map[i].cpu(), cmap='gray')
        plt.title("Probability distribution")
        plt.show()