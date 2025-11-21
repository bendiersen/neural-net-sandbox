#!/usr/bin/env python3
import os
import time
import numpy as np
import imageio
from world import GridWorld, Action
from simple_policy import simple_policy

# -----------------------------
# Create output folder
# -----------------------------
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Function to render world state to image for GIF
# -----------------------------
def render_to_frame(world):
    """
    Convert the GridWorld state to an RGB image for GIF saving.
    """
    grid = np.zeros((world.size, world.size, 3), dtype=np.uint8)
    # star = yellow
    grid[world.star_pos[1], world.star_pos[0]] = [255, 255, 0]
    # drop = blue
    grid[world.drop_pos[1], world.drop_pos[0]] = [0, 0, 255]
    # agent = red or green if carrying star
    color = [255, 0, 0] if world.has_star == 0 else [0, 255, 0]
    grid[world.agent_pos[1], world.agent_pos[0]] = color

    # Upscale for better visibility
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
        action = simple_policy(state)
        actions_taken.append(action.name)
        state, reward, done = world.step(action)
        total_reward += reward
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

    for i in range(10):
        print(f"\n--- Simulation {i+1} ---")
        run_single_sim(world, i)
        time.sleep(0.5)

    print(f"\nAll simulations complete. GIFs and action files saved in '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()

