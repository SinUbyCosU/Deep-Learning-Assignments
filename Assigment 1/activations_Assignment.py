import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

# Define transformations for the training and test sets
device=torch.device('cuda' if torch.cuda.is_available() else'cpu')