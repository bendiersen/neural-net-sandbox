#!/usr/bin/env python3
import os
import time
import torch
import torch.nn as nn
import numpy as np
import imageio
from world import GridWorld, Action

# -----------------------------
# Create output folder
# -----------------------------
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Define the same model architecture as used in training
# -----------------------------
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

# -----------------------------
# Load the trained model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PolicyNetwork(input_size=7, num_actions=len(Action)).to(device)
model.load_state_dict(torch.load("pickup_model.pth", map_location=device))
model.eval()  # set to evaluation mode

# -----------------------------
# Function to render and save frames for GIF
# -----------------------------
def render_to_frame(world):
    grid = np.zeros((world.size, world.size, 3), dtype=np.uint8)  # 0-255
    grid[world.star_pos[1], world.star_pos[0]] = [255, 255, 0]  # star yellow
    grid[world.drop_pos[1], world.drop_pos[0]] = [0, 0, 255]   # drop blue
    color = [255, 0, 0] if world.has_star == 0 else [0, 255, 0] # agent red/green
    grid[world.agent_pos[1], world.agent_pos[0]] = color

    # Upscale for visibility
    scale = 40
    image = np.kron(grid, np.ones((scale, scale, 1), dtype=np.uint8))
    return image

# -----------------------------
# Run a single simulation and save GIF + actions
# -----------------------------
def run_single_sim(world, sim_index):
    world.reset()
    state = world.get_state()
    done = False
    total_reward = 0
    frames = []
    actions_taken = []

    while not done:
        # Predict action
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(state_tensor)
            action_idx = torch.argmax(logits, dim=1).item()
        action = Action(action_idx)

        actions_taken.append(action.name)
        state, reward, done = world.step(action)
        total_reward += reward

        # Capture frame
        frame = render_to_frame(world)
        frames.append(frame)

    # Save GIF
    gif_filename = os.path.join(OUTPUT_DIR, f"simulation_{sim_index+1}.gif")
    imageio.mimsave(gif_filename, frames, duration=0.2)

    # Save actions to TXT
    txt_filename = os.path.join(OUTPUT_DIR, f"simulation_{sim_index+1}_actions.txt")
    with open(txt_filename, "w") as f:
        for a in actions_taken:
            f.write(a + "\n")

    print(f"Episode complete. Total reward: {total_reward:.2f}, GIF saved as {gif_filename}, actions saved as {txt_filename}")

# -----------------------------
# Main loop
# -----------------------------
def main():
    world = GridWorld(size=10, visualize=False)

    for i in range(5):  # run 5 simulations
        print(f"\n--- Simulation {i+1} ---")
        run_single_sim(world, i)
        time.sleep(0.5)

    print(f"\nAll simulations complete. GIFs and action files saved in '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()
