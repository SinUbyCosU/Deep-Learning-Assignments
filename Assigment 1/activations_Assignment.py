import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import json
import pandas as pd
from datetime import datetime

# Define transformations for the training and test sets
device=torch.device('cuda' if torch.cuda.is_available() else'cpu')
batch_size=64
learning_rate=0.001
epochs=10
hidden_state=128
num_classes=10

transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))])
# Load the FashionMNIST dataset
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

# 4. Training and Validation Logic
def train_and_validate(name, activation_fn):
    print(f"\nTraining with {name} activation fn")
    model = MLP(input_size=784, hidden_size=hidden_state, output_size=num_classes, activation_fn=activation_fn).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    stats = {'train_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # Training Phase
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation Phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        stats['train_loss'].append(avg_loss)
        stats['val_acc'].append(acc)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Val Acc = {acc:.2f}%")
    
    # Store final metrics
    stats['final_loss'] = stats['train_loss'][-1]
    stats['final_acc'] = stats['val_acc'][-1]
    stats['best_acc'] = max(stats['val_acc'])
    stats['best_epoch'] = stats['val_acc'].index(stats['best_acc']) + 1
    
    return stats

# 5. Execute Experiments
activations = {
    'Sigmoid': nn.Sigmoid(),  # Squashes values to (0, 1), prone to vanishing gradients
    'Tanh': nn.Tanh(),  # Squashes values to (-1, 1), zero-centered output
    'ReLU': nn.ReLU(),  # Most popular, outputs max(0, x), can suffer from dying ReLU
    'Leaky ReLU': nn.LeakyReLU(0.01),  # Allows small negative values, prevents dying ReLU
    'ELU': nn.ELU(),  # Exponential for negatives, smoother than ReLU, closer to zero mean
    'PReLU': nn.PReLU(),  # Learnable parameter for negative slope, adaptive version of Leaky ReLU
    'SELU': nn.SELU(),  # Self-normalizing, maintains mean and variance through layers
    'GELU': nn.GELU(),  # Gaussian Error Linear Unit, used in transformers like BERT and GPT
    'SiLU': nn.SiLU(),  # Sigmoid Linear Unit (Swish), smooth and non-monotonic
    'Mish': nn.Mish(),  # Smooth non-monotonic, similar to Swish with better performance
    'Softplus': nn.Softplus(),  # Smooth approximation of ReLU, always differentiable
    'Hardswish': nn.Hardswish()  # Efficient approximation of Swish, faster computation
}

results = {name: train_and_validate(name, fn) for name, fn in activations.items()}

# Save results to JSON file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f'activation_results_{timestamp}.json', 'w') as f:
    json_results = {}
    for name, stats in results.items():
        json_results[name] = {
            'train_loss': stats['train_loss'],
            'val_acc': stats['val_acc'],
            'final_loss': stats['final_loss'],
            'final_acc': stats['final_acc'],
            'best_acc': stats['best_acc'],
            'best_epoch': stats['best_epoch']
        }
    json.dump(json_results, f, indent=4)

print(f"\nResults:activation_results_{timestamp}.json")

# Create comparison summary
comparison_data = []
for name, stats in results.items():
    comparison_data.append({
        'Activation': name,
        'Final Loss': f"{stats['final_loss']:.4f}",
        'Final Accuracy': f"{stats['final_acc']:.2f}%",
        'Best Accuracy': f"{stats['best_acc']:.2f}%",
        'Best Epoch': stats['best_epoch']
    })

df = pd.DataFrame(comparison_data)
df = df.sort_values('Best Accuracy', ascending=False, key=lambda x: x.str.rstrip('%').astype(float))
df.to_csv(f'activation_comparison_{timestamp}.csv', index=False)


print(df.to_string(index=False))
print(f"\nComparison table saved to: activation_comparison_{timestamp}.csv\n")

# 6. Visualization
plt.figure(figsize=(14, 6))

# Plot Training Loss
plt.subplot(1, 2, 1)
for name, data in results.items():
    plt.plot(range(1, epochs+1), data['train_loss'], label=name, marker='o')
plt.title('Training Loss Comparison')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Validation Accuracy
plt.subplot(1, 2, 2)
for name, data in results.items():
    plt.plot(range(1, epochs+1), data['val_acc'], label=name, marker='s')
plt.title('Validation Accuracy Comparison')
plt.xlabel('epochs')
plt.ylabel('acc)')
plt.legend()
plt.tight_layout()
plt.show()