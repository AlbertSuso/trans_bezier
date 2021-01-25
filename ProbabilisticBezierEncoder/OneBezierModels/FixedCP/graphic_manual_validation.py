import torch
import os
import matplotlib.pyplot as plt

from DeterministicBezierEncoder.OneBezierModels.FixedCP.transformer import Transformer
from Utils.feature_extractor import ResNet18
from DeterministicBezierEncoder.OneBezierModels.FixedCP.dataset_generation import bezier
from Utils.chamfer_distance import chamfer_distance


basedir = "/home/albert/PycharmProjects/trans_bezier"
image_size = 64
num_cp = 3


model = Transformer(image_size, feature_extractor=ResNet18, num_transformer_layers=4, num_cp=num_cp, transformer_encoder=True).cuda()
model.load_state_dict(torch.load(basedir+"/state_dicts/DeterministicBezierEncoder/OneBezierModels/FixedCP/"+str(num_cp)+"CP_exp0"))
model.eval()

images = torch.load(os.path.join(basedir, "Datasets/OneBezierDatasets/Training/images/provaCP"+str(num_cp)))
sequences = torch.load(os.path.join(basedir, "Datasets/OneBezierDatasets/Training/sequences/provaCP"+str(num_cp)))

idx = 157
tgt_im = images[idx].unsqueeze(0).cuda()
tgt_seq = sequences[:-1, idx].cuda()

tgt_control_points = torch.empty((tgt_seq.shape[0], 2))

for i, cp in enumerate(tgt_seq):
    tgt_control_points[i, 0] = cp // image_size
    tgt_control_points[i, 1] = cp % image_size

pred_im = torch.zeros_like(tgt_im)

control_points = model.predict(tgt_im)
print("Los puntos de control predichos son", control_points)

resolution = 150
for j, t in enumerate(torch.linspace(0, 1, resolution)):
    output = bezier(control_points, t)
    output = torch.round(output).long()
    pred_im[0, 0, output[0], output[1]] = 1

target = torch.empty((3, 64, 64))
predicted = torch.empty((3, 64, 64))

target[:] = tgt_im
predicted[:] = pred_im

for cp_tgt in tgt_control_points:
    target[:, int(cp_tgt[0]), int(cp_tgt[1])] = 0
    target[0, int(cp_tgt[0]), int(cp_tgt[1])] = 1

for cp_pred in control_points.cpu():
    predicted[:, int(cp_pred[0]), int(cp_pred[1])] = 0
    predicted[0, int(cp_pred[0]), int(cp_pred[1])] = 1

print("La chamfer distance es", chamfer_distance(target.numpy(), predicted.numpy()))

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(target.transpose(0, 1).transpose(1, 2))
plt.title("Target\n{}".format(tgt_control_points))
plt.subplot(1, 2, 2)
plt.imshow(predicted.transpose(0, 1).transpose(1, 2))
plt.title("Predicted\n{}".format(control_points.cpu()))
plt.show()

model.train()

