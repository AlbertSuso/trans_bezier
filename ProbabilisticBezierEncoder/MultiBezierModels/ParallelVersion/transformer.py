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
    def __init__(self, d_model, num_beziers=2, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.num_beziers = num_beziers

        # Instanciamos el embedder
        self._embedder = nn.Embedding(num_beziers, d_model)

        # decoder_layer with d_model, nhead, dim_feedforward (implementado tal cual en el paper)
        decoder_layer = nn.TransformerDecoderLayer(d_model, 8, 4*d_model)
        self._decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)



    def _generate_tgt_mask(self, tgt_seq_len, device='cuda'):
        # La primera fila se enmascara totalmente, y la última posición de la última fila también se enmascara
        mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=device)).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, memory):
        batch_size = memory.shape[1]
        embeddings = self._embedder(torch.arange(0, self.num_beziers, dtype=torch.long, device=memory.device)).unsqueeze(1)
        embeddings = embeddings.repeat(1, batch_size, 1)
        # x.shape = (seq_len=num_beziers, batch_size, d_model)
        out = self._decoder(embeddings, memory) #, tgt_mask=self._generate_tgt_mask(embeddings.shape[0], device=embeddings.device)
        return out


class Transformer(nn.Module):
    def __init__(self, image_size, feature_extractor=ResNet18, num_transformer_layers=6, num_cp=3, max_beziers=2):
        super().__init__()
        self.image_size = image_size
        self.num_cp = num_cp
        self.max_beziers = max_beziers

        self._encoder = TransformerEncoder(num_layers=num_transformer_layers, feature_extractor=feature_extractor())
        self.d_model = self._encoder.d_model
        self._decoder = TransformerDecoder(self.d_model, num_beziers=max_beziers, num_layers=num_transformer_layers)
        self._out_cp = nn.Linear(self.d_model, 2*num_cp)


    def forward(self, image_input):
        batch_size = image_input.shape[0]

        memory = self._encoder(image_input)
        out = self._decoder(memory)
        control_points = torch.sigmoid(self._out_cp(out))

        # Una vez predichos todos los puntos de control, los pasamos al dominio (0, im_size-0.5)x(0, im_size-0.5)
        control_points = (self.image_size-0.5)*control_points

        return control_points.view(self.max_beziers, batch_size, self.num_cp, 2).permute(0, 2, 1, 3)
        # control_points.shape=(num_beziers, batch_size, num_cp, 2)
