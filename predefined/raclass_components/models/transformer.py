from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import math
import ml_collections
import copy
from models.utils.attention import Mlp, Attention, Block
from models.utils.conv_modules import conv_block_group


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.n_classes = 2
    config.activation = 'softmax'
    return config


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Embeddings(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, group_num=2):  # 6 is 3 TRA + 3 COR
        super(Encoder, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.Conv1 = conv_block_group(ch_in=img_ch, ch_out=32*group_num, group_num=group_num)
        self.Conv2 = conv_block_group(ch_in=32*group_num, ch_out=64*group_num, group_num=group_num)
        self.Conv3 = conv_block_group(ch_in=64*group_num, ch_out=128*group_num, group_num=group_num)
        self.Conv4 = conv_block_group(ch_in=128*group_num, ch_out=256*group_num, group_num=group_num)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        # encoding path
        x = self.Conv1(x)

        x = self.Maxpool(x)
        x = self.Conv2(x)

        x = self.Maxpool(x)
        x = self.Conv3(x)

        x = self.Maxpool(x)
        x = self.Conv4(x)
        x = self.avgpool(x)

        return x


class Transformer(nn.Module):
    def __init__(self, config, vis, img_ch=5, output_ch=1, group_num=2):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_ch, output_ch, group_num)  # TODO embedding using the CNN -- 
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features 
        # if x.size()[1] == 1:
        #    x = x.repeat(1,3,1,1)
        # embedding -- input images [batch_size, channel(20), 512, 512] and Conv output [batch_size, hidden_size, 7, 7]
        # [batch, 20, 512, 512]  -- [batch, 256, 7, 7]
        # Then switch the patch to second and use feature channel as the target
        # x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2)) - [batch, 256, 7, 7]
        # x = x.flatten(2) - [batch, 256, 49]
        # x = x.transpose(-1, -2) - [batch, 49, 256]
        # embeddings = x + self.position_embeddings  - self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        # embeddings = self.dropout(embeddings) -- output - [batch, 49, 256]


        # encoder -- input [batch, n_patch, hidden_size] and output [batch, n_patch, hidden_size]
        # return the size of [batch_size, n_patch, hidden_size]


# TODO need to fit the new output of transformer
class Classifier(nn.Module):
    def __init__(self, num_classes=2, group_num=2):
        super(Classifier, self).__init__()
        self.classifier_fc = nn.Sequential(
            nn.Linear(256*group_num * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = self.classifier_fc(x)
        x = self.sigmoid(x)
        # x = self.softmax(x)
        return x


class ConviT(nn.Module):
    def __init__(self, config, vis=False, img_ch=5, output_ch=1, group_num=2):
        super(ConviT, self).__init__()
        self.transformer = Transformer(config, vis, img_ch, output_ch, group_num)
        self.classifier = Classifier(num_classes=2, group_num=2)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.transformer(x)
        x = self.classifier(x)
        x = self.sigmoid(x)
        # x = self.softmax(x)
        return x
