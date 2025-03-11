import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

torch.manual_seed(1337)
