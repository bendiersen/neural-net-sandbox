import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from world import GridWorld, Action
from simple_policy import simple_policy

# Device configuration: GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Generate dataset
def generate_data(samples=10000):
    X, y = [], []
    world = GridWorld(visualize=False)
    for _ in range(samples):
        world.reset()
        state = world.get_state()
        action = simple_policy(state)
        X.append(state)
        y.append(action.value)  # integer label
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

print("Generating data...")
X, y = generate_data(200)

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).to(device)
y_tensor = torch.from_numpy(y).to(device)

# Create DataLoader for batching
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the model
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, num_actions):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
    
    def forward(self, x):
        return self.net(x)

model = PolicyNetwork(input_size=7, num_actions=len(Action)).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

# Save the model
torch.save(model.state_dict(), "pickup_model.pth")
print("Model saved as pickup_model.pth")

