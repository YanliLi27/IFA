import torch.nn as nn
from torch.utils.data import DataLoader


def model_out_reader(model:nn.Module, dataset:DataLoader):
    iterator = iter(dataset)
    x, _ = next(iterator)
    out = model(x)
    return out.shape[1]  # num_classes:int