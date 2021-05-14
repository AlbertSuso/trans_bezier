import torch
import cv2
import torch.nn as nn
import numpy as np

from Utils.feature_extractor import ResNet18
from ProbabilisticBezierEncoder.MultiBezierModels.FixedCP.dataset_generation import bezier
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
        # control_points.shape=(num_beziers, num_cp, batch_size, 2)


    def predict(self, image_input):
        """
        image_input.shape = (bs, 1, H, W)
        """
        # Segmentamos las imagenes, poniendo cada componente conexa en una imagen individual
        im_size = image_input.shape[-1]
        batch_size = image_input.shape[0]
        segmented_images = torch.zeros((0, 1, im_size, im_size), dtype=image_input.dtype, device=image_input.device)
        num_components = torch.zeros(batch_size, dtype=torch.long, device=image_input.device)
        for n, im in enumerate(image_input):
            num_labels, labels_im = cv2.connectedComponents(im[0].numpy().astype(np.uint8))
            num_components[n] = num_labels-1
            new_segmented_images = torch.zeros((num_labels-1, 1, im_size, im_size), dtype=image_input.dtype, device=image_input.device)
            for i in range(1, num_labels):
                new_segmented_images[i-1, 0][labels_im == i] = 1
            segmented_images = torch.cat((segmented_images, new_segmented_images), dim=0)

        # Aplicamos la red para obtener los puntos de control que generan cada componente conexa
        control_points = self(segmented_images.cuda())

        # Renderizamos cada componenete conexa
        connected_components = torch.zeros_like(segmented_images)
        num_cps = self.num_cp * torch.ones(control_points.shape[2], dtype=torch.long, device=control_points.device)
        for bezier_cp in control_points:
            im_seq = bezier(bezier_cp, num_cps,
                                      torch.linspace(0, 1, 150, device=num_cps.device).unsqueeze(0),
                                      device=num_cps.device)
            im_seq = torch.round(im_seq).long()
            for j in range(connected_components.shape[0]):
                connected_components[j, 0, im_seq[j, :, 0], im_seq[j, :, 1]] = 1

        # Ensamblamos las componentes conexas que pertenecen a la misma imagen
        predicted_images = torch.zeros_like(image_input)
        last_nonprocessed = 0
        for i in range(batch_size):
            n = num_components[i]
            predicted_images[i], _ = torch.max(connected_components[last_nonprocessed:last_nonprocessed+n], dim=0)
            last_nonprocessed += n

        return predicted_images
