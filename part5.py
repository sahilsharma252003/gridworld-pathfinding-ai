import heapq
import numpy as np
import os
import time
import pandas as pd
import random

class RepeatedForwardAStar:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.size = len(grid)
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
        """Run Repeated Forward A*."""
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

                    # Tie-breaking strategy: Prefer cells with larger g-values, and break remaining ties randomly
                    priority = (f_value, -tentative_g, random.random())  # Negative g for larger g-values preference

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


class AdaptiveAStar(RepeatedForwardAStar):
    def __init__(self, grid, start, goal):
        super().__init__(grid, start, goal)
        self.h_values = {self.start: self.heuristic(self.start)}

    def run(self):
        """Run Adaptive A*."""
        open_list = []
        closed_list = set()
        g_values = {self.start: 0}
        self.h_values = {self.start: self.heuristic(self.start)}  # Initialize h-values with the heuristic value
        parents = {}

        # Priority Queue: (priority, g_value, cell)
        # Priority = f(cell) = g(cell) + h(cell)
        heapq.heappush(open_list, (self.h_values[self.start], g_values[self.start], self.start))

        expanded_nodes = 0  # Track number of expanded nodes

        while open_list:
            # Get the cell with the lowest f-value, break ties based on g-value
            _, g, current = heapq.heappop(open_list)
            
            if current in closed_list:
                continue

            closed_list.add(current)
            expanded_nodes += 1  # Increment expanded nodes count

            # If we reach the goal, update h-values of expanded cells and return path and expanded nodes
            if current == self.goal:
                self.update_h_values(closed_list, g_values)
                return self.reconstruct_path(parents), expanded_nodes

            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                tentative_g = g + 1  # Cost of moving to a neighbor

                if neighbor in closed_list:
                    continue

                if neighbor not in g_values or tentative_g < g_values[neighbor]:
                    g_values[neighbor] = tentative_g
                    if neighbor not in self.h_values:
                        self.h_values[neighbor] = self.heuristic(neighbor)

                    f_value = tentative_g + self.h_values[neighbor]

                    # Tie-breaking strategy: Prefer cells with larger g-values, and break remaining ties randomly
                    priority = (f_value, -tentative_g, random.random())  # Negative g for larger g-values preference

                    heapq.heappush(open_list, (priority, tentative_g, neighbor))
                    parents[neighbor] = current

        return None, expanded_nodes  # If no path is found, return None and number of expanded nodes

    def update_h_values(self, closed_list, g_values):
        """Update h-values for cells expanded during the search."""
        for cell in closed_list:
            if cell in g_values:
                self.h_values[cell] = g_values[self.goal] - g_values[cell]  # h(s) = g(goal) - g(s)


def compare_forward_and_adaptive_a_star(gridworld_folder='gridworlds'):
    """Compare Repeated Forward A* and Adaptive A* across all gridworlds."""
    # Get all gridworld files in the specified folder
    gridworld_files = sorted([f for f in os.listdir(gridworld_folder) if f.endswith('.txt')])
    
    results = []  # Store results for each gridworld
    
    for file in gridworld_files:
        # Load each gridworld
        filepath = os.path.join(gridworld_folder, file)
        gridworld = np.loadtxt(filepath, dtype=int)
        start, goal = (0, 0), (len(gridworld) - 1, len(gridworld) - 1)  # Example start and goal positions

        print(f"Running Repeated Forward A* and Adaptive A* on {file}...")

        # Run Repeated Forward A*
        rf_a_star = RepeatedForwardAStar(gridworld, start, goal)
        start_time_forward = time.time()
        _, expanded_forward = rf_a_star.run()
        runtime_forward = time.time() - start_time_forward

        # Run Adaptive A*
        adaptive_a_star = AdaptiveAStar(gridworld, start, goal)
        start_time_adaptive = time.time()
        _, expanded_adaptive = adaptive_a_star.run()
        runtime_adaptive = time.time() - start_time_adaptive

        # Store the results for this gridworld
        results.append({
            "Gridworld": file,
            "Expanded Nodes (Repeated Forward)": expanded_forward,
            "Runtime (Repeated Forward)": runtime_forward,
            "Expanded Nodes (Adaptive A*)": expanded_adaptive,
            "Runtime (Adaptive A*)": runtime_adaptive,
        })

        print(f"Completed {file}. Results saved.")

    # Create DataFrame to store and display the results
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file
    results_filename = "forward_vs_adaptive_astar_results.csv"
    results_df.to_csv(results_filename, index=False)
    print(f"\nResults have been saved to {results_filename}.\n")

    # Display results in a structured format using pandas
    print("\nFinal Results for Repeated Forward vs Adaptive A* Strategies:\n")
    print(results_df)

    return results_df


# Call the function to compare all 50 gridworlds and save results
results_df = compare_forward_and_adaptive_a_star()
