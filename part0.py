import random
import matplotlib.pyplot as plt
import numpy as np
import os

class GridWorld:
    def __init__(self, size=101):
        self.size = size
        self.grid = np.ones((size, size))  # Initialize grid with 1s (blocked cells)
        self.visited = np.zeros((size, size))  # Track visited cells

    def is_within_bounds(self, x, y):
        """Check if a cell is within the grid boundaries."""
        return 0 <= x < self.size and 0 <= y < self.size

    def generate_gridworld(self):
        """Generate a gridworld using a DFS approach with random tie-breaking."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # North, South, West, East
        random.shuffle(directions)  # Randomize initial direction order

        # Start from a random cell and mark it as visited and unblocked
        start_x, start_y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
        self.grid[start_x, start_y] = 0  # Unblocked
        self.visited[start_x, start_y] = 1

        stack = [(start_x, start_y)]

        while stack:
            current = stack[-1]
            x, y = current

            # Get all unvisited neighbors
            neighbors = []
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if self.is_within_bounds(nx, ny) and self.visited[nx, ny] == 0:
                    neighbors.append((nx, ny))

            if neighbors:
                # Choose a random neighbor
                next_cell = random.choice(neighbors)
                nx, ny = next_cell

                # Mark the neighbor as visited
                self.visited[nx, ny] = 1

                # With 30% probability, mark as blocked; otherwise, mark as unblocked
                if random.random() < 0.3:
                    self.grid[nx, ny] = 1  # Blocked
                else:
                    self.grid[nx, ny] = 0  # Unblocked

                # Add to stack for DFS
                stack.append(next_cell)
            else:
                # Dead-end, backtrack
                stack.pop()

    def save_gridworld(self, filename):
        """Save the gridworld to a text file."""
        np.savetxt(filename, self.grid, fmt='%d')

    def visualize_gridworld(self):
        """Visualize the generated gridworld using matplotlib."""
        plt.imshow(self.grid, cmap='binary')  # 'binary' colormap for blocked (1) and unblocked (0) cells
        plt.title("Generated Gridworld")
        plt.show()

def generate_and_save_gridworlds(num_worlds=50, size=101, folder='gridworlds'):
    """Generate and save multiple gridworlds."""
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(num_worlds):
        gw = GridWorld(size=size)
        gw.generate_gridworld()  # Generate a gridworld
        filename = f'{folder}/gridworld_{i}.txt'
        gw.save_gridworld(filename)  # Save to a text file
        print(f"Saved: {filename}")

# Generate and save 50 gridworlds of size 101x101
generate_and_save_gridworlds()
