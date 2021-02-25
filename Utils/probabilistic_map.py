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

    def _normal_dist_parameters(self, t, cp_means, num_cps, cp_covariances):
        """
        Calcula la media y la matriz de covariancias de la distribucion normal que simean = torch.zeros_like(cp_means[0])gue la curva de bezier
        probabilistica en los tiempos del array de tiempos "t"

        cp_means.shape=(num_cp, batch_size, 2)
        cp_covariances.shape = (num_cp, batch_size, 2, 2)
        t.shape = (temporal_size)
        num_cps.shape=(batch_size)
        """
        # Calculamos los coeficientes binomiales necesarios para la computacion de los parametros
        binomial_coefs = torch.zeros_like(cp_means[:, :, 0]).unsqueeze(-1)
        for i, num_cp in enumerate(num_cps):
            binomial_coefs[:num_cp, i, 0] = binom((num_cp - 1).cpu(), [k for k in range(num_cp)])

        # Añadimos una dimension para el tiempo.
        cp_means = cp_means.unsqueeze(2)
        cp_means = cp_means.repeat(1, 1, self._map_temporalSize, 1)
        cp_covariances = cp_covariances.unsqueeze(2)
        cp_covariances = cp_covariances.repeat(1, 1, self._map_temporalSize, 1, 1)
        # Adaptamos la shape de t para poder vectorizar las operaciones
        t = t.view(1, -1)
        t_inv = 1 - t
        num_cps = num_cps.unsqueeze(-1)

        mean = torch.zeros_like(cp_means[0])
        covariance = torch.zeros_like(cp_covariances[0])

        for i, (cp_mean, cp_covariance) in enumerate(zip(cp_means, cp_covariances)):
            pot = (num_cps - 1 - i)
            pot[pot < 0] = 0

            mean += (binomial_coefs[i] * t ** i * t_inv ** pot).unsqueeze(-1) * cp_mean
            covariance += (binomial_coefs[i] * t ** i * t_inv ** pot).unsqueeze(-1).unsqueeze(-1) ** 2 * cp_covariance

        return mean, covariance

    def _normal2d(self, mean, covariance):
        """
        Devuelve un batch de mapas 3D de shape (64, 64, temporal_size), siendo cada slice (64, 64, t)
        la distribucion de probabilidades de la curva de bezier en tiempo t.

        mean.shape = (batch_size, temporal_size, 2) (2=cp_size)
        cov.shape = (batch_size, temporal_size, 2, 2)
        """
        batch_size = mean.shape[0]
        # Separamos aquellos elementos del batch válidos (que tienen >0 control points)
        valids1 = torch.sum(mean == 0, dim=(1, 2)) != 2*mean.shape[1]
        valids2 = torch.sum(covariance == 0, dim=(1, 2, 3)) != 4*mean.shape[1]
        valids = valids1+valids2

        # TODOS LOS UNSQUEEZE SON PARA QUE LAS DIMENSIONES CUADREN Y SE PUEDAN VECTORIZAR LAS OPERACIONES
        # Esto es debido a que p.shape=(1, 64, 64, 2). Por tanto las 3 primeras componentes de cada tensor son
        # batch_size, grid_height y grid_width respectivamente. En realidad las operaciones queremos hacerlas
        # sobre las dos ultimas componentes, y acabar obteniendo un mapa con map.shape=(batch_size, 64, 64)
        mean = mean[valids].unsqueeze(1).unsqueeze(1)
        cov_inv = torch.inverse(covariance[valids]).unsqueeze(1).unsqueeze(1)

        divisor = torch.ones((batch_size, 1, 1, self._map_temporalSize), device=mean.device)
        divisor[valids] = 2 * np.pi * torch.sqrt(torch.det(covariance[valids])).unsqueeze(1).unsqueeze(1)

        up = torch.zeros((batch_size, self._map_height, self._map_width, self._map_temporalSize, 1, 1), device=mean.device)
        result = torch.matmul(self.positions.unsqueeze(-2) - mean.unsqueeze(-2), cov_inv)
        result = torch.matmul(result, self.positions.unsqueeze(-1) - mean.unsqueeze(-1))
        up[valids] = torch.exp(-0.5 * result)
        return up[:, :, :, :, 0, 0]/divisor


    def forward(self, cp_means, num_cps, cp_covariances, mode="p"):
        """
        A partir de las medias de los puntos de control y sus matrices de covariancias, genera un batch de mapas
        probabilisticos de tamaño (batch_size, image_height, image_width, temporal_size).
        En concreto, cada mapa 2D (i, image_height, image_width, t) es la distribucion de probabilidades de la
        curva de bezier "i" en el tiempo "t".

        cp_means.shape = (num_cp, batch_size, 2)
        num_cps.shape = (batch_size)
        cp_covariances.shape = (num_cp, batch_size, 2, 2)

        output.shape = (batch_size, image_height, image_width, temporal_size)
        """
        mean, covariance = self._normal_dist_parameters(torch.linspace(0, 1, self._map_temporalSize, device=cp_means.device),
                                                        cp_means, num_cps, cp_covariances)
        probability_map = self._normal2d(mean, covariance)

        reduced_map, max_idxs = torch.max(probability_map, dim=-1)

        if mode == 'p':
            return reduced_map
        elif mode == 'd':
            # Ponemos un valor pequeño en lugar de los valores nulos del mapa de probabilidades (que son nulos debido a la falta de precisión
            # numérica de las operaciones float32
            reduced_map[reduced_map == 0] = reduced_map[reduced_map == 0] + torch.min(reduced_map[reduced_map != 0])

            # Creamos un array variances de shape variances.shape=(batch_size, 64, 64) tal que variances[i, y, x] es la variancia de la distribución normal que sigue el punto (y, x)
            # del mapa probabilistico reducido
            variances = torch.empty((batch_size, self._map_height, self._map_width), device=cp_means.device)
            for i in range(batch_size):
                variances[i] = covariance[i, max_idxs[i], 0, 0]

            distance_map = torch.sqrt(-2*variances*torch.log(2*np.pi*variances*reduced_map))
            return distance_map
        else:
            raise IndexError


if __name__ == '__main__':
    import os
    import time
    import matplotlib.pyplot as plt

    basedir = "/home/asuso/PycharmProjects/trans_bezier/Datasets/OneBezierDatasets/Training"
    images = torch.load(os.path.join(basedir, "images/fixedCP3"))
    sequences = torch.load(os.path.join(basedir, "sequences/fixedCP3"))

    map_maker = ProbabilisticMap(map_sizes=(64, 64, 50)) #.cuda()

    batch_size = 1
    idx = 166
    im = images[idx:idx+batch_size] #.cuda()
    seq = sequences[:-1, idx:idx+batch_size]

    #padding_mask = padding_masks[idx]
    #num_cp = padding_mask.shape[0] - torch.sum(padding_mask) - 1
    num_cp = 3

    cp_means = torch.empty((num_cp, batch_size, 2))
    for i in range(num_cp):
        cp_means[i, :, 0] = seq[i] // 64
        cp_means[i, :, 1] = seq[i] % 64

    cp_covariance = torch.tensor([[[1, 0], [0, 1]] for i in range(num_cp)], dtype=torch.float32)
    cp_covariances = torch.empty((num_cp, batch_size, 2, 2))
    for i in range(batch_size):
        cp_covariances[:, i, :, :] = cp_covariance

    cp_means = cp_means #.cuda()
    cp_covariances = cp_covariances #.cuda()

    t0 = time.time()
    map = map_maker(cp_means, num_cp*torch.ones(batch_size, dtype=torch.long, device='cpu'), cp_covariances, mode='d')
    print("Proporción de infs en el mapa es", torch.sum(torch.isinf(map))/(64*64))
    print(torch.max(map))
    print(map.shape)

    plt.imshow(map[0].cpu(), cmap='gray')
    plt.show()




    """max_map, _ = torch.max(map, dim=3)
    sum_map = torch.sum(map, dim=3)

    max_loss = -torch.sum(im[:, 0] * max_map / torch.sum(im[:, 0], dim=(1, 2)).view(-1, 1, 1))
    print("La max_loss obtenida es", max_loss)

    sum_loss = -torch.sum(im[:, 0] * sum_map / torch.sum(im[:, 0], dim=(1, 2)).view(-1, 1, 1))
    print("La sum_loss obtenida es", sum_loss)

    print("El valor maximo del max_map es", torch.max(max_map))
    print("El valor minimo del max_map es", torch.min(max_map))

    print(max_map.shape)
    print(torch.sum(max_map == 0))

    for i in range(5):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(im[i, 0].cpu(), cmap='gray')
        plt.title("Deterministic Image")
        plt.subplot(1, 2, 2)
        plt.imshow(max_map[i].cpu(), cmap='gray')
        plt.title("Probability distribution")
        plt.show()"""
