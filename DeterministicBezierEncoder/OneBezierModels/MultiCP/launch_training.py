import argparse
import os
import torch

from torch.optim import Adam
from DeterministicBezierEncoder.OneBezierModels.MultiCP.transformer import Transformer
from DeterministicBezierEncoder.OneBezierModels.MultiCP.training import train_one_bezier_transformer
from Utils.feature_extractor import ResNet18


# dataset_basedir = "/data2fast/users/asuso"
dataset_basedir = "/home/asuso/PycharmProjects/trans_bezier"

# state_dict_basedir = "/data1slow/users/asuso/trans_bezier"
state_dict_basedir = "/home/asuso/PycharmProjects/trans_bezier"

"""SELECTION OF HYPERPARAMETERS"""

parser = argparse.ArgumentParser()
parser.add_argument('-n_exp', '--num_experiment', type=int)
parser.add_argument('-new', '--new_model', type=bool)

parser.add_argument('-trans_encoder', '--transformer_encoder', type=bool)
parser.add_argument('-ntl', '--num_transformer_layers', type=int)
parser.add_argument('-ncp', '--num_control_points', type=int)

parser.add_argument('-bs', '--batch_size', type=int)
parser.add_argument('-e', '--num_epochs', type=int)
parser.add_argument('-lr', '--learning_rate', type=float)

# Para proseguir un entrenamiento ya iniciado. Falta implementar el resto.
parser.add_argument('-sd', '--state_dicts', type=str)

args = parser.parse_args()

num_experiment = args.num_experiment if args.num_experiment is not None else 0
new_model = args.new_model if args.new_model is not None else True

transformer_encoder = args.transformer_encoder if args.transformer_encoder is not None else True
num_transformer_layers = args.num_transformer_layers if args.num_transformer_layers is not None else 6
num_control_points = args.num_control_points if args.num_control_points is not None else 3

batch_size = args.batch_size if args.batch_size is not None else 64
num_epochs = args.num_epochs if args.num_epochs is not None else 100
learning_rate = args.learning_rate if args.learning_rate is not None else 0.00005
state_dicts_path = args.state_dicts

"""LOADING DATASET"""
images = torch.load(os.path.join(dataset_basedir, "Datasets/OneBezierDatasets/Training/images/multiCP"+str(num_control_points)))
sequences = torch.load(os.path.join(dataset_basedir, "Datasets/OneBezierDatasets/Training/sequences/multiCP"+str(num_control_points)))
tgt_padding_masks = torch.load(os.path.join(dataset_basedir, "Datasets/OneBezierDatasets/Training/padding_masks/multiCP"+str(num_control_points)))
dataset = (images, sequences, tgt_padding_masks)

"""INSTANTIATION OF THE MODEL"""
image_size = 64
model = Transformer(image_size, feature_extractor=ResNet18, num_transformer_layers=num_transformer_layers,
                    num_cp=num_control_points, transformer_encoder=transformer_encoder)
if not new_model:
    model.load_state_dict(torch.load(state_dict_basedir+"/state_dicts/DeterministicBezierEncoder/OneBezierModels/multiCP/"+str(model.num_cp)+"CP_exp"+str(num_experiment)))

"""SELECT OPTIMIZATOR AND RUN TRAINING"""
optimizer = Adam

train_one_bezier_transformer(model, dataset, batch_size, num_epochs, optimizer,
                             num_experiment, lr=learning_rate, cuda=True, debug=True)

print("FINISHED TRAINING WITH EXIT")
