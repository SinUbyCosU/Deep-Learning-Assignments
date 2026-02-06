import json
import matplotlib.pyplot as plt

# --- Configuration ---
RESULTS_FILE = 'activation_results_20260205_022245.json'
EPOCHS = 10

# --- Load Data ---
try:
    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)
    print(f"Successfully loaded results from {RESULTS_FILE}")
except FileNotFoundError:
    print(f"Error: The file {RESULTS_FILE} was not found.")
    print("Please make sure the results file exists in the same directory.")
    exit()
except json.JSONDecodeError:
    print(f"Error: The file {RESULTS_FILE} is not a valid JSON file.")
    exit()

# --- Visualization ---
print("Generating plots...")

plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# 1. Plot Training Loss
for name, data in results.items():
    if 'train_loss' in data and len(data['train_loss']) == EPOCHS:
        ax1.plot(range(1, EPOCHS + 1), data['train_loss'], marker='o', linestyle='-', label=name)
ax1.set_title('Training Loss Comparison', fontsize=16, fontweight='bold')
ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True)

# 2. Plot Validation Accuracy
for name, data in results.items():
    if 'val_acc' in data and len(data['val_acc']) == EPOCHS:
        ax2.plot(range(1, EPOCHS + 1), data['val_acc'], marker='s', linestyle='-', label=name)
ax2.set_title('Validation Accuracy Comparison', fontsize=16, fontweight='bold')
ax2.set_xlabel('Epochs', fontsize=12)
ax2.set_ylabel('Accuracy (\%)', fontsize=12)
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True)

# --- Display Plot ---
plt.tight_layout(pad=3.0)
print("Displaying graphs. Close the plot window to exit the script.")
plt.show()


