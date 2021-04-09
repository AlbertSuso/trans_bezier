import torch
import os
import time
import matplotlib.pyplot as plt

from torch.optim import Adam
from Utils.feature_extractor import ResNet18
from ProbabilisticBezierEncoder.OneBezierModels.FixedCP.transformer import Transformer as Transformer_OneBezier
from ProbabilisticBezierEncoder.MultiBezierModels.FixedCP.transformer import Transformer as Transformer_MultiBezier_FixedCP
from ProbabilisticBezierEncoder.MultiBezierModels.ParallelVersion.transformer import Transformer as Transformer_MultiBezier_Parallel
from ProbabilisticBezierEncoder.OneBezierModels.FixedCP.losses import pmap_loss, dmap_loss, chamfer_loss
from ProbabilisticBezierEncoder.MultiBezierModels.FixedCP.losses import loss_function as loss_MultiBezier_FixedCP
from ProbabilisticBezierEncoder.MultiBezierModels.ParallelVersion.losses import loss_function as loss_MultiBezier_ParallelVersion
from Utils.chamfer_distance import chamfer_distance, generate_loss_images, generate_distance_images
from Utils.probabilistic_map import ProbabilisticMap
from ProbabilisticBezierEncoder.OneBezierModels.FixedCP.training import step_decay
from ProbabilisticBezierEncoder.OneBezierModels.FixedCP.dataset_generation import bezier as OneBezier_bezier
from ProbabilisticBezierEncoder.MultiBezierModels.FixedCP.dataset_generation import bezier as MultiBezier_bezier
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = "cuda:0"
# device = "cpu"

# modelo = "OneBezier"
modelo = "MultiBezier_FixedCP"
# modelo = "MultiBezier_ParallelVersion"

max_beziers = 3
num_cp = 3
num_tl = 6 # Para multi bezier fixedCP
# num_tl = 5 # Para OneBezier

dataset_basedir = "/home/asuso/PycharmProjects/trans_bezier"
state_dict_basedir = "/home/asuso/PycharmProjects/trans_bezier"
best_loss = float('inf')

if modelo == "OneBezier":
    images = torch.load(os.path.join(dataset_basedir, "Datasets/OneBezierDatasets/Training/images/fixedCP"+str(num_cp)))
    bezier_rend = OneBezier_bezier
else:
    images = torch.load(os.path.join(dataset_basedir, "Datasets/MultiBezierDatasets/Training/images/fixedCP"+str(num_cp)+"_maxBeziers"+str(max_beziers)+"imSize64"))
    bezier_rend = MultiBezier_bezier

grid = torch.empty((1, 1, images.shape[2], images.shape[3], 2), dtype=torch.float32)
for i in range(images.shape[2]):
    grid[0, 0, i, :, 0] = i
    grid[0, 0, :, i, 1] = i

if modelo == "OneBezier":
    model = Transformer_OneBezier(64, feature_extractor=ResNet18, num_transformer_layers=num_tl,
                                  num_cp=num_cp, transformer_encoder=True).to(device)
    state_dict_path = state_dict_basedir + "/state_dicts/ProbabilisticBezierEncoder/OneBezierModels/FixedCP/losschamfer_distancequadratic_penalization0.1"

elif modelo == "MultiBezier_FixedCP":
    model = Transformer_MultiBezier_FixedCP(64, feature_extractor=ResNet18, num_transformer_layers=num_tl, num_cp=num_cp, max_beziers=max_beziers).to(device)
    state_dict_path = state_dict_basedir+"/state_dicts/ProbabilisticBezierEncoder/MultiBezierModels/FixedCP/"+str(num_cp)+"CP_maxBeziers"+str(max_beziers)+"losschamfer"

elif modelo == "MultiBezier_ParallelVersion":
    model = Transformer_MultiBezier_Parallel(64, feature_extractor=ResNet18, num_transformer_layers=num_tl,
                                             num_cp=num_cp, max_beziers=max_beziers).to(device)
    state_dict_path = state_dict_basedir+"/state_dicts/ProbabilisticBezierEncoder/MultiBezierModels/ParallelVersion/"+str(num_cp)+"CP_maxBeziers"+str(max_beziers)

else:
    raise Exception


probabilistic_map_generator = ProbabilisticMap((model.image_size, model.image_size, 50)).to(device)
cp_covariances = torch.tensor([ [[1, 0], [0, 1]] for i in range(model.num_cp)], dtype=torch.float32).to(device).unsqueeze(1)
grid = grid.to(device)

idx = 168 #161
num_samples = 1
times = torch.empty(num_samples, dtype=torch.float32, device=device)
best_chamfers = torch.empty(num_samples, dtype=torch.float32, device=device)
best_chamfer_epochs = torch.empty(num_samples, dtype=torch.float32, device=device)
losses = torch.empty((num_samples, 100), dtype=torch.float32, device=device)
model.eval()
for n in range(num_samples):
    print("Sample", n)
    model.load_state_dict(torch.load(state_dict_path))
    optimizer = Adam(model.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=10 ** (-0.5), patience=4, min_lr=1e-10)
    im = images[idx+n].unsqueeze(0)
    loss_im = generate_loss_images(im, weight=5).to(device)
    dist_im = generate_distance_images(im).to(device)
    im = im.to(device)


    best_chamfer = float('inf')
    best_chamfer_epoch = 0
    best_image = None
    t0 = time.time()
    for i in range(100):
        #actual_covariances = cp_covariances * step_decay(25, i, 0.5, 6, 1).to(device)

        if modelo == "OneBezier":
            control_points, num_cps = model(im)
        elif modelo == "MultiBezier_FixedCP":
            control_points, probabilities = model(im)
            num_cps = model.num_cp + torch.zeros(1, dtype=torch.long, device=control_points.device)
        elif modelo == "MultiBezier_ParallelVersion":
            control_points = model(im)
            num_cps = model.num_cp + torch.zeros(1, dtype=torch.long, device=control_points.device)

        num_beziers = 1 if modelo == "OneBezier" else max_beziers
        predicted_im = torch.zeros_like(im)
        for j in range(num_beziers):
            im_seq = bezier_rend(control_points[3*j:3*(j+1)], num_cps, torch.linspace(0, 1, 150, device=device).unsqueeze(0), device=device)
            im_seq = torch.round(im_seq).long()
            predicted_im[0, 0, im_seq[0, :, 0], im_seq[0, :, 1]] = 1

        chamfer_dist = chamfer_distance(predicted_im[0].cpu().numpy(), im[0].cpu().numpy())
        # scheduler.step(chamfer_dist)
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

        if modelo == "OneBezier":
            loss = chamfer_loss(control_points, num_cps, im, dist_im, cp_covariances, probabilistic_map_generator, grid)
        elif modelo == "MultiBezier_FixedCP":
            epoch = 10 # Si queremos entrenar la predicción del numero de beziers, hay que poner epoch>=50
            loss = loss_MultiBezier_FixedCP(epoch, control_points, model.max_beziers+torch.zeros(1, dtype=torch.long, device=control_points.device),
                                            probabilities, model.num_cp, im, dist_im, loss_im, grid, cp_covariances,
                                            probabilistic_map_generator, loss_type="chamfer", distance='l2', gamma=0.9)
        elif modelo == "MultiBezier_ParallelVersion":
            loss = loss_MultiBezier_ParallelVersion(control_points, im, dist_im, cp_covariances, probabilistic_map_generator, grid)

        loss.backward()
        optimizer.step()
        model.zero_grad()
        losses[n, i] = loss.detach()

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

mean_loss = torch.mean(losses, dim=0)
plt.plot(mean_loss.cpu())
plt.show()
