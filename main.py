import argparse
import os
import torch

from torch.optim import Adam
from ProbabilisticBezierEncoder.OneBezierModels.FixedCP.transformer import Transformer
from ProbabilisticBezierEncoder.OneBezierModels.FixedCP.training import train_one_bezier_transformer
from Utils.feature_extractor import ResNet18


dataset_basedir = "/data2fast/users/asuso"
# dataset_basedir = "/home/asuso/PycharmProjects/trans_bezier"

state_dict_basedir = "/data1slow/users/asuso/trans_bezier"
# state_dict_basedir = "/home/asuso/PycharmProjects/trans_bezier"

"""SELECTION OF HYPERPARAMETERS"""

parser = argparse.ArgumentParser()
parser.add_argument('-n_exp', '--num_experiment', type=int)
parser.add_argument('-new', '--new_model', type=bool)

parser.add_argument('-ntl', '--num_transformer_layers', type=int)
parser.add_argument('-ncp', '--num_control_points', type=int)

parser.add_argument('-bs', '--batch_size', type=int)
parser.add_argument('-e', '--num_epochs', type=int)
parser.add_argument('-lr', '--learning_rate', type=float)

parser.add_argument('-ltype', '--loss_type', type=str)
parser.add_argument('-dtype', '--distance_type', type=str)

# parser.add_argument('-pred_var', '--predict_variance', type=bool)
parser.add_argument('-cpv', '--cp_variance', type=int)
parser.add_argument('-vdrop', '--variance_drop', type=float)
parser.add_argument('-edrop', '--epochs_drop', type=int)
parser.add_argument('-minv', '--min_variance', type=float)
parser.add_argument('-pcoef', '--penalization_coef', type=float)


args = parser.parse_args()

num_experiment = args.num_experiment if args.num_experiment is not None else 0
new_model = args.new_model if args.new_model is not None else True

num_transformer_layers = args.num_transformer_layers if args.num_transformer_layers is not None else 5
num_control_points = args.num_control_points if args.num_control_points is not None else 5

batch_size = args.batch_size if args.batch_size is not None else 32
num_epochs = args.num_epochs if args.num_epochs is not None else 200
learning_rate = args.learning_rate if args.learning_rate is not None else 0.00005

loss_type = args.loss_type if args.loss_type is not None else 'chamfer'
distance_type = args.distance_type if args.distance_type is not None else 'l2'

# predict_variance = args.predict_variance if args.predict_variance is not None else True
cp_variance = args.cp_variance if args.cp_variance is not None else 25
variance_drop = args.variance_drop if args.variance_drop is not None else 0.5
epochs_drop = args.epochs_drop if args.epochs_drop is not None else 5
min_variance = args.min_variance if args.min_variance is not None else 1
penalization_coef = args.penalization_coef if args.penalization_coef is not None else 0.1

"""LOADING DATASET"""
# images = torch.load(os.path.join(dataset_basedir, "Datasets/MNIST/thinned_relocated"))
images = torch.load(os.path.join(dataset_basedir, "Datasets/OneBezierDatasets/Training/images/fixedCP"+str(num_control_points)))
# sequences = torch.load(os.path.join(dataset_basedir, "Datasets/OneBezierDatasets/Training/sequences/fixedCP"+str(num_control_points)))
dataset = images

"""INSTANTIATION OF THE MODEL"""
image_size = 64
model = Transformer(image_size, feature_extractor=ResNet18, num_transformer_layers=num_transformer_layers,
                    num_cp=num_control_points, transformer_encoder=True)
if not new_model:
    model.load_state_dict(torch.load(state_dict_basedir+"/state_dicts/ProbabilisticBezierEncoder/OneBezierModels/fixedCP/"+str(model.num_cp)+"CP_exp"+str(num_experiment)))

"""SELECT OPTIMIZATOR AND RUN TRAINING"""
optimizer = Adam

"""SELECT LOSS AND LOSS_DISTANCE"""
loss_mode = (loss_type, distance_type)

train_one_bezier_transformer(model, dataset, batch_size, num_epochs, optimizer, loss_mode,
                             num_experiment, cp_variance, variance_drop, epochs_drop, min_variance, penalization_coef,
                             lr=learning_rate, cuda=True, debug=True)

print("FINISHED TRAINING WITH EXIT")
