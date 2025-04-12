import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_and_save_gridworld(gridworld, filename):
    """Visualizes a gridworld and saves it as an image file."""
    plt.figure(figsize=(8, 8))
    plt.imshow(gridworld, cmap='binary')  # 'binary' colormap for blocked (1) and unblocked (0) cells
    plt.title(f"Gridworld Visualization: {filename}")
    plt.colorbar(label='Cell Status (0=Unblocked, 1=Blocked)')
    plt.savefig(filename, bbox_inches='tight')  # Save the figure as an image file
    plt.close()  # Close the figure to free memory

def load_and_visualize_all_gridworlds(gridworld_folder='gridworlds', output_folder='gridworld_visualizations'):
    """Loads all gridworlds from a folder, visualizes, and saves them as images."""
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all gridworld files in the specified folder
    gridworld_files = sorted([f for f in os.listdir(gridworld_folder) if f.endswith('.txt')])

    for file in gridworld_files:
        # Load each gridworld
        filepath = os.path.join(gridworld_folder, file)
        gridworld = np.loadtxt(filepath, dtype=int)

        # Generate a corresponding image filename
        image_filename = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.png")

        # Visualize and save the gridworld as an image
        visualize_and_save_gridworld(gridworld, image_filename)
        print(f"Saved visualization: {image_filename}")

# Call the function to visualize and save all gridworlds
load_and_visualize_all_gridworlds()
