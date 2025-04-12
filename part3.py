import heapq
import numpy as np
import os
import time
import pandas as pd

class RepeatedAStar:
    def __init__(self, grid, start, goal, mode="forward"):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.size = len(grid)
        self.mode = mode  # Can be "forward" or "backward"
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

    def is_within_bounds(self, x, y):
        """Check if a cell is within the grid boundaries."""
        return 0 <= x < self.size and 0 <= y < self.size

    def heuristic(self, cell):
        """Manhattan distance heuristic."""
        return abs(cell[0] - self.goal[0]) + abs(cell[1] - self.goal[1])

    def get_neighbors(self, cell):
        """Get neighbors of a cell."""
        neighbors = []
        for dx, dy in self.directions:
            nx, ny = cell[0] + dx, cell[1] + dy
            if self.is_within_bounds(nx, ny) and self.grid[nx, ny] == 0:
                neighbors.append((nx, ny))
        return neighbors

    def run(self):
        """Run Repeated A* with either forward or backward search."""
        # Set start and goal positions based on the mode
        if self.mode == "forward":
            start, goal = self.start, self.goal
        elif self.mode == "backward":
            start, goal = self.goal, self.start

        open_list = []
        closed_list = set()
        g_values = {start: 0}
        parents = {}

        # Priority Queue: (priority, g_value, cell)
        # Priority = f(cell) = g(cell) + h(cell)
        heapq.heappush(open_list, (self.heuristic(start), g_values[start], start))

        expanded_nodes = 0  # Track number of expanded nodes

        while open_list:
            # Get the cell with the lowest f-value, break ties based on g-value
            _, g, current = heapq.heappop(open_list)
            
            if current in closed_list:
                continue

            closed_list.add(current)
            expanded_nodes += 1  # Increment expanded nodes count

            # If we reach the goal, return the path and number of expanded nodes
            if current == goal:
                return self.reconstruct_path(parents, start, goal), expanded_nodes

            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                tentative_g = g + 1  # Cost of moving to a neighbor

                if neighbor in closed_list:
                    continue

                if neighbor not in g_values or tentative_g < g_values[neighbor]:
                    g_values[neighbor] = tentative_g
                    f_value = tentative_g + self.heuristic(neighbor)

                    heapq.heappush(open_list, (f_value, tentative_g, neighbor))
                    parents[neighbor] = current

        return None, expanded_nodes  # If no path is found, return None and number of expanded nodes

    def reconstruct_path(self, parents, start, goal):
        """Reconstruct the path from start to goal."""
        path = []
        current = goal
        while current in parents:
            path.append(current)
            current = parents[current]
        path.append(start)  # Add the start to the path
        path.reverse()
        return path


def compare_forward_and_backward_a_star(gridworld_folder='gridworlds'):
    """Compare Repeated Forward A* and Repeated Backward A* across all gridworlds."""
    # Get all gridworld files in the specified folder
    gridworld_files = sorted([f for f in os.listdir(gridworld_folder) if f.endswith('.txt')])
    
    results = []  # Store results for each gridworld
    
    for file in gridworld_files:
        # Load each gridworld
        filepath = os.path.join(gridworld_folder, file)
        gridworld = np.loadtxt(filepath, dtype=int)
        start, goal = (0, 0), (len(gridworld) - 1, len(gridworld) - 1)  # Example start and goal positions

        print(f"Running Repeated Forward and Backward A* on {file}...")

        # Run Repeated Forward A*
        rf_a_star = RepeatedAStar(gridworld, start, goal, mode="forward")
        start_time_forward = time.time()
        _, expanded_forward = rf_a_star.run()
        runtime_forward = time.time() - start_time_forward

        # Run Repeated Backward A*
        rb_a_star = RepeatedAStar(gridworld, start, goal, mode="backward")
        start_time_backward = time.time()
        _, expanded_backward = rb_a_star.run()
        runtime_backward = time.time() - start_time_backward

        # Store the results for this gridworld
        results.append({
            "Gridworld": file,
            "Expanded Nodes (Forward)": expanded_forward,
            "Runtime (Forward)": runtime_forward,
            "Expanded Nodes (Backward)": expanded_backward,
            "Runtime (Backward)": runtime_backward,
        })

        print(f"Completed {file}. Results saved.")

    # Create DataFrame to store and display the results
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file
    results_filename = "forward_vs_backward_astar_results.csv"
    results_df.to_csv(results_filename, index=False)
    print(f"\nResults have been saved to {results_filename}.\n")

    # Display results in a structured format using pandas
    print("\nFinal Results for Repeated Forward vs Backward A* Strategies:\n")
    print(results_df)

    return results_df


# Call the function to compare all 50 gridworlds and save results
results_df = compare_forward_and_backward_a_star()
