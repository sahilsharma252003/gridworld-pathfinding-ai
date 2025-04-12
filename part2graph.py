import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file for Part 5 results (Replace the path with your actual CSV file path)
df_part5 = pd.read_csv("forward_vs_adaptive_astar_results.csv")  # Replace with your file name, e.g., "forward_vs_adaptive_astar_results.csv"

# Calculate the average values for expanded nodes and runtime
avg_expanded_nodes_forward = df_part5["Expanded Nodes (Repeated Forward)"].mean()
avg_expanded_nodes_adaptive = df_part5["Expanded Nodes (Adaptive A*)"].mean()
avg_runtime_forward = df_part5["Runtime (Repeated Forward)"].mean()
avg_runtime_adaptive = df_part5["Runtime (Adaptive A*)"].mean()

# Create a DataFrame to hold the summary results
data_part5 = {
    "Algorithm": ["Repeated Forward A*", "Adaptive A*"],
    "Average Expanded Nodes": [avg_expanded_nodes_forward, avg_expanded_nodes_adaptive],
    "Average Runtime (seconds)": [avg_runtime_forward, avg_runtime_adaptive]
}

df_summary_part5 = pd.DataFrame(data_part5)

# Plot 1: Average Number of Expanded Nodes for Repeated Forward A* vs Adaptive A*
plt.figure(figsize=(10, 6))
plt.bar(df_summary_part5["Algorithm"], df_summary_part5["Average Expanded Nodes"], color=['skyblue', 'orange'])
plt.title("Comparison of Average Number of Expanded Nodes: Repeated Forward A* vs Adaptive A*", fontsize=16)
plt.xlabel("Algorithm", fontsize=14)
plt.ylabel("Average Expanded Nodes", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for index, value in enumerate(df_summary_part5["Average Expanded Nodes"]):
    plt.text(index, value + 50, f'{value:.2f}', ha='center', va='bottom', fontsize=12)  # Display value on top of bars
plt.show()

# Plot 2: Average Runtime for Repeated Forward A* vs Adaptive A*
plt.figure(figsize=(10, 6))
plt.bar(df_summary_part5["Algorithm"], df_summary_part5["Average Runtime (seconds)"], color=['lightgreen', 'lightcoral'])
plt.title("Comparison of Average Runtime: Repeated Forward A* vs Adaptive A*", fontsize=16)
plt.xlabel("Algorithm", fontsize=14)
plt.ylabel("Average Runtime (seconds)", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for index, value in enumerate(df_summary_part5["Average Runtime (seconds)"]):
    plt.text(index, value + 0.0001, f'{value:.6f}', ha='center', va='bottom', fontsize=12)  # Display value on top of bars
plt.show()
