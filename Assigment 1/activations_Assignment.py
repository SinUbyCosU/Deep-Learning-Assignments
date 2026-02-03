import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

# Define transformations for the training and test sets
device=torch.device('cuda' if torch.cuda.is_available() else'cpu')
batch_size=64
learning_rate=0.001
epochs=10
hidden_state=128
num_classes=10

transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))])