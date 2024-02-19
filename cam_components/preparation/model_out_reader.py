import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def model_out_reader(model:nn.Module, dataset:DataLoader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    iterator = iter(dataset)
    x, _ = next(iterator)
    model = model.to(device)
    out = model(x.to(device))
    return out.shape[1]  # num_classes:int