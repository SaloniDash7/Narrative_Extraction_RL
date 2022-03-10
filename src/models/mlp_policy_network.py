import torch.nn as nn
import torch.nn.functional as F
from src.models.utils import clones, create_padding_mask


class CoordinateWiseMLPPolicy(nn.Module):
    def __init__(self, dim_input, dim_hidden=128, num_layers=4):
        super(CoordinateWiseMLPPolicy, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        narr_encoder_layers, query_encoder_layers = self.create_encoder_layers()

        self.narr_encoder = nn.Sequential(*narr_encoder_layers)
        self.query_encoder = nn.Sequential(*query_encoder_layers)

    def forward(self, narrative_embeds, query_embeds, num_narratives):
        """
        Inputs:
            - narrative_embeds: Shape -> [batch_size, max_num_narratives, D]
            - query_embeddings: Shape -> [batch_size, num_queries, D]
            - num_narratives: Shape -> [batch_size,]
        """
        narrative_encodings = self.narr_encoder(narrative_embeds)
        query_encodings = self.query_encoder(query_embeds)

        narrative_mask = create_padding_mask(
            num_narratives, narrative_embeds.size(1)
        ).unsqueeze(-1)
        agg_narrative_encoding = (narrative_encodings * narrative_mask).sum(
            1
        ) / num_narratives.unsqueeze(-1)

        outputs = agg_narrative_encoding.unsqueeze(1) @ query_encodings.transpose(
            -1, -2
        )
        outputs = outputs.squeeze(1)
        outputs = F.softmax(outputs, dim=-1)
        return outputs

    def create_encoder_layers(self):
        narr_encoder_layers = []
        query_encoder_layers = []
        in_features = self.dim_input
        for _ in range(self.num_layers - 1):
            encoder_layer = nn.Sequential(
                nn.Linear(in_features, self.dim_hidden), nn.ReLU()
            )
            narr_encoder_layer, query_encoder_layer = clones(encoder_layer, 2)
            narr_encoder_layers.append(narr_encoder_layer)
            query_encoder_layers.append(query_encoder_layer)
            in_features = self.dim_hidden
        narr_encoder_layer, query_encoder_layer = clones(
            nn.Linear(self.dim_input, self.dim_hidden), 2
        )
        narr_encoder_layers.append(narr_encoder_layer)
        query_encoder_layers.append(query_encoder_layer)

        return narr_encoder_layers, query_encoder_layers
