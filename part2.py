import heapq
import numpy as np
import os
import time
import pandas as pd

class RepeatedForwardAStar:
    def __init__(self, grid, start, goal, break_ties="smaller_g"):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.size = len(grid)
        self.break_ties = break_ties
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
        """Run Repeated Forward A* with specified tie-breaking strategy."""
        open_list = []
        closed_list = set()
        g_values = {self.start: 0}
        parents = {}

        # Priority Queue: (priority, g_value, cell)
        # Priority = f(cell) = g(cell) + h(cell)
        heapq.heappush(open_list, (self.heuristic(self.start), g_values[self.start], self.start))

        expanded_nodes = 0  # Track number of expanded nodes

        while open_list:
            # Get the cell with the lowest f-value, break ties based on g-value
            _, g, current = heapq.heappop(open_list)
            
            if current in closed_list:
                continue

            closed_list.add(current)
            expanded_nodes += 1  # Increment expanded nodes count

            # If we reach the goal, return the path and number of expanded nodes
            if current == self.goal:
                return self.reconstruct_path(parents), expanded_nodes

            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                tentative_g = g + 1  # Cost of moving to a neighbor

                if neighbor in closed_list:
                    continue

                if neighbor not in g_values or tentative_g < g_values[neighbor]:
                    g_values[neighbor] = tentative_g
                    f_value = tentative_g + self.heuristic(neighbor)
                    
                    # Priority adjustment for tie-breaking
                    if self.break_ties == "smaller_g":
                        priority = f_value
                    else:
                        priority = f_value - tentative_g  # Larger g-value first

                    heapq.heappush(open_list, (priority, tentative_g, neighbor))
                    parents[neighbor] = current

        return None, expanded_nodes  # If no path is found, return None and number of expanded nodes

    def reconstruct_path(self, parents):
        """Reconstruct the path from start to goal."""
        path = []
        current = self.goal
        while current in parents:
            path.append(current)
            current = parents[current]
        path.reverse()
        return path


def compare_tie_breaking_strategies_on_all_gridworlds(gridworld_folder='gridworlds'):
    """Compare Repeated Forward A* with different tie-breaking strategies across all gridworlds."""
    # Get all gridworld files in the specified folder
    gridworld_files = sorted([f for f in os.listdir(gridworld_folder) if f.endswith('.txt')])
    
    results = []  # Store results for each gridworld
    
    for file in gridworld_files:
        # Load each gridworld
        filepath = os.path.join(gridworld_folder, file)
        gridworld = np.loadtxt(filepath, dtype=int)
        start, goal = (0, 0), (len(gridworld) - 1, len(gridworld) - 1)  # Example start and goal positions

        print(f"Running Repeated Forward A* on {file}...")

        # Run Repeated Forward A* with tie-breaking in favor of smaller g-values
        rf_a_star_smaller_g = RepeatedForwardAStar(gridworld, start, goal, break_ties="smaller_g")
        start_time_smaller_g = time.time()
        _, expanded_smaller_g = rf_a_star_smaller_g.run()
        runtime_smaller_g = time.time() - start_time_smaller_g

        # Run Repeated Forward A* with tie-breaking in favor of larger g-values
        rf_a_star_larger_g = RepeatedForwardAStar(gridworld, start, goal, break_ties="larger_g")
        start_time_larger_g = time.time()
        _, expanded_larger_g = rf_a_star_larger_g.run()
        runtime_larger_g = time.time() - start_time_larger_g

        # Store the results for this gridworld
        results.append({
            "Gridworld": file,
            "Expanded Nodes (Smaller g)": expanded_smaller_g,
            "Runtime (Smaller g)": runtime_smaller_g,
            "Expanded Nodes (Larger g)": expanded_larger_g,
            "Runtime (Larger g)": runtime_larger_g,
        })

        print(f"Completed {file}. Results saved.")

    # Create DataFrame to store and display the results
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file
    results_filename = "repeated_forward_astar_results.csv"
    results_df.to_csv(results_filename, index=False)
    print(f"\nResults have been saved to {results_filename}.\n")

    # Display results in a structured format using pandas
    print("\nFinal Results for Repeated Forward A* Tie-breaking Strategies:\n")
    print(results_df)

    return results_df


# Call the function to compare all 50 gridworlds and save results
results_df = compare_tie_breaking_strategies_on_all_gridworlds()
