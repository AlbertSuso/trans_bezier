import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from Utils.feature_extractor import ResNet12
from Utils.positional_encoder import PositionalEncoder

class TransformerEncoder(nn.Module):
    def __init__(self, feature_extractor, num_layers=3):
        super().__init__()
        self._embedder = feature_extractor
        self.d_model = self._embedder.d_model

        # encoder_layer with d_model, nhead, dim_feedforward (implementado tal cual en el paper)
        encoder_layer = nn.TransformerEncoderLayer(self.d_model, 8, 4*self.d_model)
        self._encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) # 4 capes


    def forward(self, image):
        #image.shape = (batch_size, 1, 64, 64)
        features = self._embedder(image)
        #features.shape = (batch_size, num_feature_maps=d_model, X, X)
        #features = features.view(image.shape[0], self._embedder.d_model, -1).transpose(1, 2).transpose(0, 1)
        #features.shape = (seq_len, batch_size, d_model)
        return self._encoder(features)
        #return.shape = (seq_len, batch_size, d_model)

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_layers=3):
        super().__init__()
        self.d_model = d_model

        # embedder para el token de "seguir"
        self._embedder = nn.Embedding(2, d_model)
        self._positional_encoder = PositionalEncoder(d_model)

        # decoder_layer with d_model, nhead, dim_feedforward (implementado tal cual en el paper)
        decoder_layer = nn.TransformerDecoderLayer(d_model, 8, 4*d_model)
        self._decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)


    def _generate_tgt_mask(self, tgt_seq_len, type=1, device='cuda'):
        mask = None
        # La primera fila se enmascara totalmente, y la última posición de la última fila también se enmascara
        if type == 0:
            mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=device))
            mask = mask.float().masked_fill(mask == 1, float('-inf'))
        # La primera posición de la primera fila no se enmascara, y la última posición de la última fila tampoco
        elif type == 1:
            mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=device)).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask


    def forward(self, tgt, memory, tgt_key_padding_mask):
        # tgt.shape= (seq_len, batch_size, 1)
        x = self._embedder(tgt)
        # x.shape = (seq_len, batch_size, d_model)
        x = self._positional_encoder(x)
        return self._decoder(x, memory, tgt_mask=self._generate_tgt_mask(tgt.shape[0], device=x.device), tgt_key_padding_mask=tgt_key_padding_mask)

class NumBezierPredictor(nn.Module):
    def __init__(self, feature_extractor=ResNet12, num_transformer_layers=3, num_cp=3):
        super().__init__()
        self.num_cp = num_cp

        self._encoder = TransformerEncoder(num_layers=num_transformer_layers, feature_extractor=feature_extractor())
        self.d_model = self._encoder.d_model
        self._decoder = TransformerDecoder(self.d_model, num_layers=num_transformer_layers)

        self._out_probabilites = nn.Linear(self.d_model, 2)

    def forward(self, image_inputs, tgt_seq, tgt_key_padding_mask):
        # Aplicamos el encoder
        memory = self._encoder(image_inputs)

        # Shifteamos en 1 la tgt_seq y le añadimos un token de "seguir"
        input_tgt_seq = torch.empty_like(tgt_seq)
        input_tgt_seq[0, :] = 0
        input_tgt_seq[1:] = tgt_seq[:-1]

        # Aplicamos el decoder
        output = self._decoder(input_tgt_seq, memory, tgt_key_padding_mask)
        # output.shape = (tgt_seq_len, batch_size, d_model)

        # Devolvemos las probabilidades sin pasar por la softmax, porque la cross-entropy loss ya la aplica automáticamente
        return self._out_probabilites(output) #return.shape = (tgt_seq_len, batch_size, num_probabilites)

    def predict(self, image_inputs, max_beziers = 5):
        batch_size = image_inputs.shape[0]

        memory = self._encoder(image_inputs)

        seq = torch.zeros((1, batch_size), dtype=torch.long, device=image_inputs.device)
        not_finished = torch.ones(batch_size, dtype=torch.bool, device=image_inputs.device)
        num_beziers = torch.ones(batch_size, device=image_inputs.device)

        # Mientras aún queden secuencias por finalizar
        i = 0
        while torch.sum(not_finished) > 0 and i < max_beziers+1:
            i += 1
            # Ejecutamos el decoder y la capa lineal para obtener un nuevo batch de tokens
            output = self._decoder(seq, memory, None)
            output = output[-1]
            probabilities = self._out_probabilites(output)
            new_tokens = torch.argmax(probabilities, dim=-1)

            # Añadimos los nuevos tokens a la secuencia
            seq = torch.cat((seq, new_tokens.unsqueeze(0)), dim=0)
            # Actualizamos la lista de secuencias no finalizadas
            not_finished[new_tokens == 1] = False
            # Actualizamos el tensor que cuenta el numero de beziers de cada imagen del batch
            num_beziers[not_finished] += 1

        return num_beziers

def train_num_beziers_predictor(model, dataset, lr=1e-3, epochs=7, batch_size=64, cuda=True):
    images, num_beziers = dataset
    dataset_size = images.shape[0]
    max_beziers = torch.max(num_beziers)

    tgt_seq = torch.zeros((torch.max(num_beziers)+1, dataset_size), dtype=torch.long)
    tgt_padding_masks = torch.zeros((dataset_size, max_beziers+1))
    for i, num_bezier in enumerate(num_beziers):
        tgt_seq[num_bezier:, i] = 1
        tgt_padding_masks[i, num_bezier:] = 1
    tgt_padding_masks = tgt_padding_masks.bool()

    if cuda:
        images = images.cuda()
        tgt_seq = tgt_seq.cuda()
        tgt_padding_masks = tgt_padding_masks.cuda()
        num_beziers = num_beziers.cuda()

        model = model.cuda()

    # Separamos en training y validation
    im_training = images[:40000]
    im_validation = images[40000:]
    tgt_seq_training = tgt_seq[:, :40000]
    tgt_seq_validation = tgt_seq[:, 40000:]
    padding_masks_training = tgt_padding_masks[:40000]
    padding_masks_validation = tgt_padding_masks[40000:]
    num_beziers_training = num_beziers[:40000]
    num_beziers_validation = num_beziers[40000:]

    # Definimos el optimizer
    optimizer = Adam(model.parameters(), lr=lr)


    print("Pre-entrenamiento del predictor de num_cp")
    for epoch in range(epochs):
        t0 = time.time()
        print("Beginning epoch number", epoch + 1)
        # actual_covariances = cp_covariances * step_decay(cp_variance, epoch, var_drop, epochs_drop, min_variance).to(cp_covariances.device)
        total_loss = 0
        for i in range(0, len(im_training) - batch_size + 1, batch_size):
            # Obtenemos el batch
            im = im_training[i:i+batch_size]
            seq = tgt_seq_training[:, i:i+batch_size]
            padding_masks = padding_masks_training[i:i+batch_size]

            # Aplicamos el modelo
            probabilities = model(im, seq, padding_masks)

            # Calculamos la loss
            loss = 0
            for k in range(batch_size):
                loss += F.cross_entropy(probabilities[:, k], seq[:, k])

            # Actualizamos la total_loss
            total_loss += loss

            # Realizamos un paso de descenso del gradiente
            loss.backward()
            optimizer.step()
            model.zero_grad()

        print("La training loss es", total_loss/40000)
        total_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for j in range(0, len(im_validation) - batch_size + 1, batch_size):
                # Obtenemos el batch
                im = im_validation[j:j+batch_size]
                seq = tgt_seq_validation[:, j:j+batch_size]
                padding_masks = padding_masks_validation[j:j+batch_size]
                num_beziers = num_beziers_validation[j:j+batch_size]

                # Aplicamos el modelo
                probabilities = model(im, seq, padding_masks)

                # Actualizamos la total_loss
                for k in range(batch_size):
                    total_loss += F.cross_entropy(probabilities[:, k], seq[:, k])

                # Actualizamos la accuracy
                pred_num_beziers = model.predict(im, max_beziers=max_beziers)
                # print(pred.shape)
                # print(num_beziers.shape)
                accuracy += torch.sum(pred_num_beziers == num_beziers)
        print("La validation loss es", total_loss / 10000)
        print("La validation accuracy es", accuracy/10000)
        model.train()
        print("Tiempo por epoca de", time.time()-t0)
    return model.cpu()

if __name__ =='__main__':
    # dataset_basedir = "/data2fast/users/asuso"
    dataset_basedir = "/home/asuso/PycharmProjects/trans_bezier"

    num_cp = 3
    max_beziers = 2
    im_size = 64
    model = NumBezierPredictor(num_cp=num_cp)

    """LOADING DATASET"""
    images = torch.load(os.path.join(dataset_basedir, "Datasets/MultiBezierDatasets/Training/images/fixedCP"+str(num_cp)+"_maxBeziers"+str(max_beziers)+"imSize"+str(im_size)))
    num_beziers = torch.load(os.path.join(dataset_basedir, "Datasets/MultiBezierDatasets/Training/num_beziers/fixedCP"+str(num_cp)+"_maxBeziers"+str(max_beziers)+"imSize"+str(im_size)))
    dataset = (images, num_beziers)

    lr = 1e-3
    epochs = 20
    batch_size = 64

    train_num_beziers_predictor(model, dataset, lr=lr, epochs=epochs, batch_size=batch_size, cuda=True)
