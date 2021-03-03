import torch
import os
import matplotlib.pyplot as plt

from torch.optim import Adam
from Utils.feature_extractor import ResNet18
from ProbabilisticBezierEncoder.OneBezierModels.FixedCP.transformer import Transformer
from ProbabilisticBezierEncoder.OneBezierModels.FixedCP.losses import dmap_loss, pmap_loss
from Utils.chamfer_distance import chamfer_distance, generate_loss_images
from Utils.probabilistic_map import ProbabilisticMap
from ProbabilisticBezierEncoder.OneBezierModels.FixedCP.training import step_decay
from ProbabilisticBezierEncoder.OneBezierModels.FixedCP.dataset_generation import bezier


num_cp = 5
num_tl = 5

dataset_basedir = "/home/asuso/PycharmProjects/trans_bezier"
state_dict_basedir = "/home/asuso/PycharmProjects/trans_bezier"
best_loss = float('inf')

images = torch.load(os.path.join(dataset_basedir, "Datasets/OneBezierDatasets/Training/images/fixedCP"+str(num_cp)))
loss_images = generate_loss_images(images)
im = images[157].unsqueeze(0).cuda()
loss_im = loss_images[157].unsqueeze(0).cuda()

model = Transformer(64, feature_extractor=ResNet18, num_transformer_layers=num_tl,
                    num_cp=num_cp, transformer_encoder=True).cuda()
model.load_state_dict(torch.load(state_dict_basedir+"/state_dicts/ProbabilisticBezierEncoder/OneBezierModels/fixedCP/lossdmap_distancel2_preEntrenat"))
optimizer = Adam(model.parameters(), lr=1e-3)

probabilistic_map_generator = ProbabilisticMap((model.image_size, model.image_size, 50)).cuda()
cp_covariances = torch.tensor([ [[1, 0], [0, 1]] for i in range(model.num_cp)], dtype=torch.float32).cuda().unsqueeze(1)


for i in range(10000):
    actual_covariances = cp_covariances * step_decay(25, i, 0.5, 6, 1).cuda()

    control_points, num_cps = model(im)
    loss = pmap_loss(control_points, num_cps, actual_covariances, im, loss_im, probabilistic_map_generator, mode='p')

    loss.backward()
    optimizer.step()
    model.zero_grad()

    if i % 1000 == 999:
        print("loss={}\n".format(loss))

        im_seq = bezier(control_points, num_cps, torch.linspace(0, 1, 150, device=control_points.device).unsqueeze(0),
                        device='cuda')
        im_seq = torch.round(im_seq).long()
        predicted_im = torch.zeros_like(im)
        predicted_im[0, 0, im_seq[0, :, 0], im_seq[0, :, 1]] = 1

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(im[0, 0].cpu(), cmap='gray')
        plt.title("Original")
        plt.subplot(1, 2, 2)
        plt.imshow(predicted_im[0, 0].cpu(), cmap='gray')
        plt.title("Predicted epoch {}".format(i))
        plt.show()


