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


    def forward(self, tgt, memory):
        # tgt.shape= (seq_len, batch_size, 1)
        x = self._embedder(tgt)
        # x.shape = (seq_len, batch_size, d_model)
        x = self._positional_encoder(x)
        return self._decoder(x, memory, tgt_mask=self._generate_tgt_mask(tgt.shape[0], device=x.device))

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

        self._out_probabilites = nn.Linear(self.d_model, 1 + image_size*image_size)

    def forward(self, image_inputs, tgt_seq):
        # Aplicamos el encoder
        memory = self._encoder(image_inputs)

        #Shifteamos en 1 la tgt_seq y le añadimos un BOS token
        input_tgt_seq = torch.empty_like(tgt_seq)
        input_tgt_seq[0, :] = self.image_size*self.image_size
        input_tgt_seq[1:] = tgt_seq[:-1]

        #Aplicamos el decoder
        output = self._decoder(input_tgt_seq, memory)
        #output.shape = (tgt_seq_len, batch_size, d_model)

        # Devolvemos las probabilidades sin pasar por la softmax, porque la cross-entropy loss ya la aplica automáticamente
        return self._out_probabilites(output) #return.shape = (tgt_seq_len, batch_size, num_probabilites)

    def predict(self, image_input):
        memory = self._encoder(image_input)

        control_points = []
        actual = self.image_size*self.image_size

        # Mientras el ganador no sea el que indica que debemos parar la ejecucón
        n = 0
        while (actual != self.image_size*self.image_size and n < self.num_cp + 1) or n == 0:
            n += 1
            control_points.append(actual)

            # Ejecutamos el decoder para obtener un nuevo punto de control, o la orden de parada
            output = self._decoder(torch.tensor(control_points, dtype=torch.long, device=memory.device).view(-1, 1),
                                   memory)
            output = output[-1]

            probabilities = self._out_probabilites(output).view(-1)
            actual = torch.argmax(probabilities)

        control_points = control_points[1:]
        out_control_points = torch.empty((len(control_points), 2), device=image_input.device)
        for i, cp in enumerate(control_points):
            out_control_points[i] = torch.tensor([cp//self.image_size, cp%self.image_size])

        return out_control_points

