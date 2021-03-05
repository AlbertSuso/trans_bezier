import torch
import os
import time
import matplotlib.pyplot as plt

from torch.optim import Adam
from Utils.feature_extractor import ResNet18
from ProbabilisticBezierEncoder.OneBezierModels.FixedCP.transformer import Transformer
from ProbabilisticBezierEncoder.OneBezierModels.FixedCP.losses import pmap_loss, dmap_loss, chamfer_loss
from Utils.chamfer_distance import chamfer_distance, generate_loss_images, generate_distance_images
from Utils.probabilistic_map import ProbabilisticMap
from ProbabilisticBezierEncoder.OneBezierModels.FixedCP.training import step_decay
from ProbabilisticBezierEncoder.OneBezierModels.FixedCP.dataset_generation import bezier

device = "cuda:0"
# device = "cpu"

num_cp = 5
num_tl = 5

dataset_basedir = "/home/asuso/PycharmProjects/trans_bezier"
state_dict_basedir = "/home/asuso/PycharmProjects/trans_bezier"
best_loss = float('inf')

images = torch.load(os.path.join(dataset_basedir, "Datasets/OneBezierDatasets/Training/images/fixedCP"+str(num_cp)))

grid = torch.empty((1, 1, images.shape[2], images.shape[3], 2), dtype=torch.float32)
for i in range(images.shape[2]):
    grid[0, 0, i, :, 0] = i
    grid[0, 0, :, i, 1] = i

model = Transformer(64, feature_extractor=ResNet18, num_transformer_layers=num_tl,
                    num_cp=num_cp, transformer_encoder=True).to(device)
model.load_state_dict(torch.load(state_dict_basedir+"/state_dicts/ProbabilisticBezierEncoder/OneBezierModels/FixedCP/losschamfer_distancequadratic_penalization0.1"))
# model.load_state_dict(torch.load(state_dict_basedir+"/state_dicts/ProbabilisticBezierEncoder/OneBezierModels/FixedCP/losspmap_distancel2"))
optimizer = Adam(model.parameters(), lr=1e-5)

probabilistic_map_generator = ProbabilisticMap((model.image_size, model.image_size, 50)).to(device)
cp_covariances = torch.tensor([ [[1, 0], [0, 1]] for i in range(model.num_cp)], dtype=torch.float32).to(device).unsqueeze(1)
grid = grid.to(device)

idx = 161
num_samples = 1
times = torch.empty(num_samples, dtype=torch.float32, device=device)
best_chamfers = torch.empty(num_samples, dtype=torch.float32, device=device)
best_chamfer_epochs = torch.empty(num_samples, dtype=torch.float32, device=device)
model.eval()
for n in range(num_samples):
    im = images[idx+n].unsqueeze(0)
    loss_im = generate_loss_images(im, weight=5).to(device)
    dist_im = generate_distance_images(im).to(device)
    im = im.to(device)


    best_chamfer = float('inf')
    best_chamfer_epoch = 0
    best_image = None
    t0 = time.time()
    for i in range(60):
        #actual_covariances = cp_covariances * step_decay(25, i, 0.5, 6, 1).to(device)

        control_points, num_cps = model(im)

        im_seq = bezier(control_points, num_cps, torch.linspace(0, 1, 150, device=device).unsqueeze(0),
                        device=device)
        im_seq = torch.round(im_seq).long()
        predicted_im = torch.zeros_like(im)
        predicted_im[0, 0, im_seq[0, :, 0], im_seq[0, :, 1]] = 1

        chamfer_dist = chamfer_distance(predicted_im[0].cpu().numpy(), im[0].cpu().numpy())
        if chamfer_dist < best_chamfer:
            # print("Epoch {} --> chamfer_distance={}".format(i+1, chamfer_dist))
            best_chamfer = chamfer_dist
            best_chamfer_epoch = i+1
            best_image = predicted_im

            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(im[0, 0].cpu(), cmap='gray')
            plt.title("Original")
            plt.subplot(1, 2, 2)
            plt.imshow(predicted_im[0, 0].cpu(), cmap='gray')
            plt.title("Predicted epoch {}".format(i+1))
            plt.show()

        # loss = pmap_loss(control_points, num_cps, actual_covariances, im, loss_im, probabilistic_map_generator, mode='p')
        loss = chamfer_loss(control_points, num_cps, im, dist_im, cp_covariances, probabilistic_map_generator, grid)
        # loss = dmap_loss(control_points, num_cps, im, grid, distance='l2')

        loss.backward()
        optimizer.step()
        model.zero_grad()

    times[n] = time.time()-t0
    best_chamfers[n] = best_chamfer[0]
    best_chamfer_epochs[n] = best_chamfer_epoch
    #print("\nTiempo transcurrido --->", time.time()-t0)

    """plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im[0, 0].cpu(), cmap='gray')
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(best_image[0, 0].cpu(), cmap='gray')
    plt.title("Predicted epoch {}".format(i+1))
    plt.show()"""

model.train()


print("El tiempo medio requerido ha sido de", torch.mean(times))
print("La best_chamfer media obtenida es de", torch.mean(best_chamfers))
print("La época media en la que se ha alcanzado la best_chamfer es", torch.mean(best_chamfer_epochs))
print("El porcentaje de chamfer distances por debajo de 0.3 es", 100*torch.sum(best_chamfers < 0.3)/num_samples)
print("El porcentaje de chamfer distances por debajo de 0.15 es", 100*torch.sum(best_chamfers < 0.15)/num_samples)
print("El porcentaje de chamfer distances por debajo de 0.1 es", 100*torch.sum(best_chamfers < 0.1)/num_samples)
print("El porcentaje de chamfer distances por debajo de 0.075 es", 100*torch.sum(best_chamfers < 0.075)/num_samples)
print("El porcentaje de chamfer distances por debajo de 0.05 es", 100*torch.sum(best_chamfers < 0.05)/num_samples)
