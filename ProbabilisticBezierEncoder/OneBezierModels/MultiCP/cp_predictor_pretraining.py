import os
import torch
import torch.nn.functional as F

from torch.optim import Adam
from Utils.feature_extractor import NumCP_predictor12, NumCP_predictor18

def train_cp_predictor(model, dataset, lr=1e-3, epochs=7, batch_size=64, cuda=True):
    images, tgt_padding_masks = dataset
    min_cp = 2
    num_cp = (tgt_padding_masks.shape[1] - 1 - min_cp) - torch.sum(tgt_padding_masks, dim=-1)

    if cuda:
        images = images.cuda()
        num_cp = num_cp.cuda()
        model = model.cuda()

    # Separamos en training y validation
    im_training = images[:40000]
    im_validation = images[40000:]
    num_cp_training = num_cp[:40000]
    num_cp_validation = num_cp[40000:]

    # Definimos el optimizer
    optimizer = Adam(model.parameters(), lr=lr)


    print("Pre-entrenamiento del predictor de num_cp")
    for epoch in range(epochs):
        print("Beginning epoch number", epoch + 1)
        # actual_covariances = cp_covariances * step_decay(cp_variance, epoch, var_drop, epochs_drop, min_variance).to(cp_covariances.device)
        total_loss = 0
        accuracy = 0
        for i in range(0, len(im_training) - batch_size + 1, batch_size):
            # Obtenemos el batch
            im = im_training[i:i + batch_size]
            groundtruth = num_cp_training[i: i+batch_size]

            # Aplicamos el modelo
            probabilities = model(im)

            # Calculamos la loss
            loss = F.cross_entropy(probabilities, groundtruth)

            # Actualizamos la accuracy y la total_loss
            accuracy += torch.sum(torch.argmax(probabilities, dim=-1) == groundtruth)
            total_loss += loss

            # Realizamos un paso de descenso del gradiente
            loss.backward()
            optimizer.step()
            model.zero_grad()

        print("EPOCA", epoch+1, "La training loss es", total_loss/40000)
        print("EPOCA", epoch+1, "La training accuracy es", accuracy/40000)

        total_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for j in range(0, len(im_validation) - batch_size + 1, batch_size):
                # Obtenemos el batch
                im = im_validation[j:j + batch_size]
                groundtruth = num_cp_validation[j: j + batch_size]

                # Aplicamos el modelo
                probabilities = model(im)

                # Actualizamos la total_loss
                total_loss += F.cross_entropy(probabilities, groundtruth)

                # Actualizamos la accuracy
                accuracy += torch.sum(torch.argmax(probabilities, dim=-1) == groundtruth)

            print("EPOCA", epoch + 1, "La validation loss es", total_loss / 10000)
            print("EPOCA", epoch + 1, "La validation accuracy es", accuracy / 10000)
        model.train()
    return model.cpu()

if __name__ =='__main__':
    # dataset_basedir = "/data2fast/users/asuso"
    dataset_basedir = "/home/asuso/PycharmProjects/trans_bezier"

    max_cp = 5
    lr = 1e-3
    epochs = 20
    batch_size = 64

    """LOADING DATASET"""
    images = torch.load(os.path.join(dataset_basedir, "Datasets/OneBezierDatasets/Training/images/multiCP"+str(max_cp)+"_larger"))
    sequences = torch.load(os.path.join(dataset_basedir, "Datasets/OneBezierDatasets/Training/sequences/multiCP"+str(max_cp)+"_larger"))
    tgt_padding_masks = torch.load(os.path.join(dataset_basedir, "Datasets/OneBezierDatasets/Training/padding_masks/multiCP"+str(max_cp)+"_larger"))
    dataset = (images, tgt_padding_masks)

    for net in [NumCP_predictor12, NumCP_predictor18]:
        model = net(max_cp=max_cp)
        train_cp_predictor(model, dataset, lr=lr, epochs=epochs, batch_size=batch_size, cuda=True)

