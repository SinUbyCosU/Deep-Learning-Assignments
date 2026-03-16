#Assignment 2
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset , Dataset
from torchvision import corrupt
from sklearn.model_selection import train_test_split

class CorruptDataset(Dataset):
    def __init__(self,base_dataset,corruption_name=None,severity=2):
        self.base_dataset=base_dataset
        self.corruption_name=corruption_name
        self.severity=severity
        