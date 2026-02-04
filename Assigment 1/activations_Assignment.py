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
# Load the MNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform)   

#train loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
#Modular mlp architecture 
class MLP(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,activation_fn):
        super(MLP,self).__init__()
        self.flatten=nn.Flatten()
        self.network=nn.Sequential(
            nn.Linear(input_size,hidden_size),
            activation_fn,
            nn.Linear(hidden_size,output_size),
            activation_fn,
            nn.Linear(output_size,output_size)
        )
    def forward(self,x):
        x=self.flatten(x)
        out=self.network(x)
        return out  