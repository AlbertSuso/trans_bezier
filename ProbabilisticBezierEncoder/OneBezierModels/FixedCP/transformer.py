import torch
import torch.nn as nn

from Utils.feature_extractor import ResNet18
from Utils.positional_encoder import PositionalEncoder

"""
Very simplified model that always produces a fixed number of control points. In consequence, it doesn't generate BOS
nor EOS flags. It always receives one BOS flag as input in the first position of the sequence.
"""


class Embedder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self._d_model = d_model
        # Embedding for the BOS (begin of sequence)
        self._bos_embedder = nn.Embedding(1, d_model)
        # Embedding for the control points
        self._cp_embedder = nn.Linear(2, d_model)


    def forward(self, tgt_seq):
        """
        tgt_seq.shape = (num_cp, batch_size, 2)      If the true sequence has 5 CP and 1 BOS token, then num_cp=5
        """
        out = torch.empty((1+len(tgt_seq), tgt_seq.shape[1], self._d_model), dtype=torch.float32, device=tgt_seq.device)
        out[0] = self._bos_embedder(torch.zeros(1, dtype=torch.long, device=tgt_seq.device))
        out[1:] = self._cp_embedder(tgt_seq)
        return out


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
    def __init__(self, d_model, num_layers=6):
        super().__init__()
        self.d_model = d_model

        # Instanciamos el embedder
        self._embedder = Embedder(d_model)

        # Positional Encoder
        self._positional_encoder = PositionalEncoder(d_model)

        # decoder_layer with d_model, nhead, dim_feedforward (implementado tal cual en el paper)
        decoder_layer = nn.TransformerDecoderLayer(d_model, 8, 4*d_model)
        self._decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)


    def _generate_tgt_mask(self, tgt_seq_len, device='cuda'):
        # La primera fila se enmascara totalmente, y la última posición de la última fila también se enmascara
        mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=device)).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, tgt, memory):
        # tgt.shape= (seq_len, batch_size, 1)
        x = self._embedder(tgt)
        # x.shape = (seq_len, batch_size, d_model)
        x = self._positional_encoder(x)
        out = self._decoder(x, memory, tgt_mask=self._generate_tgt_mask(x.shape[0], device=x.device))
        return out


class Transformer(nn.Module):
    def __init__(self, image_size, feature_extractor=ResNet18, num_transformer_layers=6, num_cp=3, transformer_encoder=True):
        super().__init__()
        self.image_size = image_size
        self.num_cp = num_cp

        self._encoder = TransformerEncoder(num_layers=num_transformer_layers, feature_extractor=feature_extractor())
        self.d_model = self._encoder.d_model
        self._decoder = TransformerDecoder(self.d_model, num_layers=num_transformer_layers)

        self._out_cp = nn.Linear(self.d_model, 2)
        #self._out_variance = nn.Linear(self.d_model, 1)


    def forward(self, image_input):
        batch_size = image_input.shape[0]
        memory = self._encoder(image_input)

        control_points = torch.zeros((0, batch_size, 2), dtype=torch.float32, device=image_input.device)

        # Generamos self.num_cp puntos de control (ninguno puede ser padding)
        for n in range(self.num_cp):
            # Ejecutamos el decoder para obtener un nuevo punto de control
            output = self._decoder(control_points, memory)
            last = output[-1]

            cp = torch.sigmoid(self._out_cp(last)).view(1, batch_size, 2)
            control_points = torch.cat((control_points, cp), dim=0)

        # Una vez predichos todos los puntos de control, los pasamos al dominio (0, im_size-0.5)x(0, im_size-0.5)
        control_points *= self.image_size-0.5

        # Calculamos el tensor num_cps (POR IMPLEMENTAR PARA MultiCP !!!!!!!!!!!!!!!!)
        num_cps = self.num_cp*torch.ones(batch_size, dtype=torch.long, device=image_input.device)

        # En caso de ser necesario, predecimos la variancia de los CP de este lote y la devolvemos
        # return control_points, num_cps, 0.1+torch.relu(self._out_variance(output)).unsqueeze(-1)
        return control_points, num_cps

