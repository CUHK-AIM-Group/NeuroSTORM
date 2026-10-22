import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from .ptdec import DEC
from typing import List
from .components import InterpretableTransformerEncoder


class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True, freeze_center=False, project_assignment=True, nHead=4, local_transformer=False):
        super().__init__()
        self.transformer = InterpretableTransformerEncoder(d_model=input_feature_size, nhead=nHead,
                                                           dim_feedforward=hidden_size,
                                                           batch_first=True)

        self.local_transformer = local_transformer
        if local_transformer:
            self.pooling = False
        else:
            self.pooling = pooling
        if self.pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size *
                          input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

        if local_transformer:
            self.class_token = nn.ParameterList()
            self.class_token.append(nn.Parameter(torch.Tensor(1,input_feature_size), requires_grad = True))
            self.class_token.append(nn.Parameter(torch.Tensor(1,input_feature_size), requires_grad = True))
            self.class_token.append(nn.Parameter(torch.Tensor(1,input_feature_size), requires_grad = True))
            self.class_token.append(nn.Parameter(torch.Tensor(1,input_feature_size), requires_grad = True))
            self.class_token.append(nn.Parameter(torch.Tensor(1,input_feature_size), requires_grad = True))
            self.class_token.append(nn.Parameter(torch.Tensor(1,input_feature_size), requires_grad = True))
            self.class_token.append(nn.Parameter(torch.Tensor(1,input_feature_size), requires_grad = True))
            self.class_token.append(nn.Parameter(torch.Tensor(1,input_feature_size), requires_grad = True))
        self.reset_parameters(local_transformer)

        self.mlp = nn.Sequential(
            nn.Linear(8*input_feature_size, 1024),
            nn.Linear(1024, input_feature_size),
            nn.ReLU()
        )

    def reset_parameters(self, local_transformer=False):
        if local_transformer:
            for i in range(len(self.class_token)):
                self.class_token[i] = nn.init.xavier_normal_(self.class_token[i])


    def is_pooling_enabled(self):
        return self.pooling

    def forward(self,
            x: torch.tensor, cluster_num=-1):
        bz, node_num, dim = x.shape
        if self.local_transformer:
            class_token = self.class_token[cluster_num]
            class_token = class_token.repeat(bz,1,1)
            x = torch.cat((class_token, x), dim=1)
        x = self.transformer(x)
        if self.local_transformer:
            cls_token = x[:, 0, :]
            x = x[:, 1:, :]
            return x, None, cls_token.reshape(x.shape[0], 1, -1)
        else:
            cls_token = x[:, 0, :]
            x = x[:, 1:, :]
            if self.pooling:
                x, assignment = self.dec(x)
                return x, assignment, cls_token.reshape(x.shape[0], 1, -1)
            else:
                return x, None, cls_token.reshape(x.shape[0], 1, -1)

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)
