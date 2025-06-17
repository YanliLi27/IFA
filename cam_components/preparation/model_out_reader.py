import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Union
import numpy as np


def model_out_reader(model:nn.Module, dataset:Union[DataLoader, np.ndarray], device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    iterator = iter(dataset)
    x, _ = next(iterator)
    model = model.to(device)
    out = model(x.to(device))
    return out.shape[1]  # num_classes:int