import argparse
import os
import torch

from torch.optim import Adam
from ProbabilisticBezierEncoder.MultiBezierModels.ParallelVersion.transformer import Transformer
from ProbabilisticBezierEncoder.MultiBezierModels.ParallelVersion.training import train_one_bezier_transformer
from Utils.feature_extractor import ResNet18


# dataset_basedir = "/data2fast/users/asuso"
dataset_basedir = "/home/asuso/PycharmProjects/trans_bezier"

# state_dict_basedir = "/data1slow/users/asuso/trans_bezier"
state_dict_basedir = "/home/asuso/PycharmProjects/trans_bezier"

"""SELECTION OF HYPERPARAMETERS"""

parser = argparse.ArgumentParser()
parser.add_argument('-n_exp', '--num_experiment', type=int)
parser.add_argument('-new', '--new_model', type=bool)

parser.add_argument('-ntl', '--num_transformer_layers', type=int)
parser.add_argument('-ncp', '--num_control_points', type=int)
parser.add_argument('-max_bez', '--max_beziers', type=int)

parser.add_argument('-bs', '--batch_size', type=int)
parser.add_argument('-e', '--num_epochs', type=int)
parser.add_argument('-lr', '--learning_rate', type=float)

parser.add_argument('-cpc', '--curv_pen_coef', type=float)
parser.add_argument('-rc', '--rep_coef', type=float)
parser.add_argument('-dt', '--dist_thresh', type=float)
parser.add_argument('-st', '--second_term', type=bool)


args = parser.parse_args()

num_experiment = args.num_experiment if args.num_experiment is not None else 0
new_model = args.new_model if args.new_model is not None else True

num_transformer_layers = args.num_transformer_layers if args.num_transformer_layers is not None else 6
num_control_points = args.num_control_points if args.num_control_points is not None else 3
max_beziers = args.max_beziers if args.max_beziers is not None else 2

batch_size = args.batch_size if args.batch_size is not None else 16
num_epochs = args.num_epochs if args.num_epochs is not None else 200
learning_rate = args.learning_rate if args.learning_rate is not None else 0.00005

curv_pen_coef = args.curv_pen_coef if args.curv_pen_coef is not None else 0.01
rep_coef = args.rep_coef if args.rep_coef is not None else 0.1
dist_thresh = args.dist_thresh if args.dist_thresh is not None else 4.5
second_term = args.second_term if args.second_term is not None else True

"""LOADING DATASET"""
# images = torch.load(os.path.join(dataset_basedir, "Datasets/MNIST/thinned_relocated"))
images = torch.load(os.path.join(dataset_basedir, "Datasets/MultiBezierDatasets/Training/images/fixedCP"+str(num_control_points)+"_maxBeziers"+str(max_beziers)+"imSize64"))
# sequences = torch.load(os.path.join(dataset_basedir, "Datasets/OneBezierDatasets/Training/sequences/fixedCP"+str(num_control_points)))
dataset = images

"""INSTANTIATION OF THE MODEL"""
image_size = 64
model = Transformer(image_size, feature_extractor=ResNet18, num_transformer_layers=num_transformer_layers,
                    num_cp=num_control_points, max_beziers=max_beziers)
if not new_model:
    model.load_state_dict(torch.load(state_dict_basedir+"/state_dicts/ProbabilisticBezierEncoder/MultiBezierModels/ParallelVersion/"+str(model.num_cp)+"CP_maxBeziers"+str(max_beziers)))

"""SELECT OPTIMIZATOR AND RUN TRAINING"""
optimizer = Adam

train_one_bezier_transformer(model, dataset, batch_size, num_epochs, optimizer, num_experiment, lr=learning_rate, curv_pen_coef=curv_pen_coef,
                             rep_coef=rep_coef, dist_thresh=dist_thresh, second_term=second_term, cuda=True, debug=True)

print("FINISHED TRAINING WITH EXIT")
