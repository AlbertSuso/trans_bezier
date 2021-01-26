"""PROGRAM TO MAKE SILLY COMPROVATIONS"""

import torch
import torch.nn as nn

from Utils.feature_extractor import ResNet18
from Utils.positional_encoder import PositionalEncoder


class TransformerEncoder(nn.Module):
    def __init__(self, feature_extractor, num_layers=6):
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
    def __init__(self, d_model, image_size, num_layers=6):
        super().__init__()
        self.d_model = d_model

        # la dimensió input es una per cada possible punt de control + 1 per inici de seq
        self._embedder = nn.Embedding(1+image_size*image_size, d_model)
        self._positional_encoder = PositionalEncoder(d_model)

        # decoder_layer with d_model, nhead, dim_feedforward (implementado tal cual en el paper)
        decoder_layer = nn.TransformerDecoderLayer(d_model, 8, 4*d_model)
        self._decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)


    def _generate_tgt_mask(self, tgt_seq_len, type=1, cuda=True):
        mask = None
        device = 'cuda' if cuda else 'cpu'
        # La primera fila se enmascara totalmente, y la última posición de la última fila también se enmascara
        if type == 0:
            mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=device))
            mask = mask.float().masked_fill(mask == 1, float('-inf'))
        # La primera posición de la primera fila no se enmascara, y la última posición de la última fila tampoco
        elif type == 1:
            mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=device)).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask


    def forward(self, tgt, memory):
        # tgt.shape= (seq_len, batch_size, 1)
        x = self._embedder(tgt)
        # x.shape = (seq_len, batch_size, d_model)
        """Aqui, o quizá antes del embedder, iría el shifting. Supongo que se añade un primer token artificial y se elimina el ultimo token de la secuencia.
        Así si que tiene sentido la mascara que antes no me cuadraba"""
        x = self._positional_encoder(x)
        return self._decoder(x, memory, tgt_mask=self._generate_tgt_mask(tgt.shape[0]))

class Transformer(nn.Module):
    def __init__(self, image_size, feature_extractor=ResNet18, num_transformer_layers=6, num_cp=3, transformer_encoder=True):
        super().__init__()
        self.image_size = image_size
        self.num_cp = num_cp

        if transformer_encoder:
            self._encoder = TransformerEncoder(num_layers=num_transformer_layers, feature_extractor=feature_extractor())
        else:
            self._encoder = feature_extractor()
        self.d_model = self._encoder.d_model
        self._decoder = TransformerDecoder(self.d_model, image_size, num_layers=num_transformer_layers)

        self._out_probabilites = nn.Linear(self.d_model, 1+image_size*image_size)

    def forward(self, image_input):
        batch_size = image_input.shape[0]
        special_token = self.image_size*self.image_size
        memory = self._encoder(image_input)

        control_points = torch.empty((self.num_cp+1, batch_size), dtype=torch.long, device=image_input.device)
        control_points[0] = special_token

        # Generamos self.num_cp puntos de control (algunos de los cuales pueden ser padding)
        # En concreto, seran padding aquellos predichos despes del EOS token (y el EOS token tambien)
        for n in range(1, self.num_cp+1):
            # Ejecutamos el decoder para obtener un nuevo punto de control, o el EOS
            output = self._decoder(control_points[:n], memory)
            output = output[-1]

            probabilities = self._out_probabilites(output).view(batch_size, -1)
            control_points[n] = torch.argmax(probabilities, dim=1)

        control_points = control_points[1:]

        # Convertimos los CP a formato (x,y) y generamos un tensor "num_cps" que nos dice cuantos cp forman cada
        # curva de bezier del batch
        out_control_points = torch.zeros((len(control_points), batch_size, 2), device=image_input.device)
        num_cps = self.num_cp*torch.ones(batch_size, dtype=torch.long, device=image_input.device)
        not_ended = {i for i in range(batch_size)}
        for i, batch_cp in enumerate(control_points):
            for batch_element in not_ended:
                cp = batch_cp[batch_element]
                if cp == special_token:
                    num_cps[batch_element] = i
                    not_ended.discard(batch_element)
                else:
                    out_control_points[i, batch_element, 0] = cp // 64
                    out_control_points[i, batch_element, 1] = cp % 64

        return out_control_points, num_cps