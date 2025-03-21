"""
visualization.py

This module provides visualization utilities for the Traveling Salesman Problem (TSP).
It contains functions to:
- Ensure directories exist before saving files.
- Visualize the best TSP route with arrows between cities.
- Visualize the distance matrix between all cities as a heatmap.
- Visualize the progression of the best cost over generations during optimization.

Each function includes detailed inline comments to explain the purpose of each step and how data flows through the code.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

def ensure_dir(save_path):
    """
    Ensures that the directory for the given save_path exists.

    Parameters:
    - save_path (str): The file path where a file will be saved.

    Explanation:
    - Extracts the directory from the save_path.
    - Checks if the directory exists; if not, creates the directory (and any intermediate directories).
    """
    # Extract the directory path from the provided file path.
    directory = os.path.dirname(save_path)
    # Check if a directory path exists and whether it is not already present.
    if directory and not os.path.exists(directory):
        # Create the directory including any intermediate directories.
        os.makedirs(directory)

def visualize_best_route_with_arrows(tsp_problem, route, save_path="../graphs/best_route_arrows.png", show=False):
    """
    Visualizes the best TSP route by drawing arrows between cities in order.

    Parameters:
    - tsp_problem: An instance containing TSP problem details (cities, etc.).
    - route (list): List of city indices representing the TSP tour order.
    - save_path (str): Path to save the generated plot image.
    - show (bool): If True, displays the plot.

    Explanation:
    - Calls ensure_dir() to verify that the directory for saving the plot exists.
    - Iterates over the route, drawing arrows from each city to the next.
    - Uses modulo arithmetic to wrap around from the last city back to the first.
    - Plots each city and annotates them with their index.
    - Saves the plot to the given path and optionally displays it.
    """
    # Ensure that the directory to save the image exists.
    ensure_dir(save_path)
    # Retrieve the list of city coordinates from the tsp_problem instance.
    cities = tsp_problem.cities
    # Create a new figure with a specified size.
    plt.figure(figsize=(8, 6))
    # Mark the starting city with a text annotation.
    plt.text(cities[0][0], cities[0][1], "Start city", fontsize=9, ha='left')
    # Loop through each index in the route to draw arrows.
    for i in range(len(route)):
        # Retrieve the starting city coordinate for the current leg.
        start_city = cities[route[i]]
        # Retrieve the ending city coordinate for the next leg; wrap-around using modulo.
        end_city = cities[route[(i + 1) % len(route)]]
        # Calculate the differences in x and y coordinates for the arrow vector.
        dx = end_city[0] - start_city[0]
        dy = end_city[1] - start_city[1]
        # Draw an arrow from the start city to the end city.
        plt.arrow(start_city[0], start_city[1], dx, dy,
                  length_includes_head=True, head_width=2, head_length=2,
                  fc='skyblue', ec='dodgerblue')
    # Unzip the list of city coordinates into separate lists for x and y values.
    xs, ys = zip(*cities)
    # Plot the cities as scatter points.
    plt.scatter(xs, ys, color='dodgerblue')
    # Annotate each city with its corresponding index.
    for idx, (x, y) in enumerate(cities):
        plt.text(x, y, str(idx), fontsize=9, ha='right')
    # Set the title and labels for the plot.
    plt.title("Best TSP Route")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    # Add a grid to the plot with a specified color.
    plt.grid(c='paleturquoise')
    # Save the plot to the specified path.
    plt.savefig(save_path)
    # Optionally display the plot if show is True.
    if show:
        plt.show()

def visualize_distance_matrix(tsp_problem, save_path="../graphs/distance_matrix.png", show=False):
    """
    Visualizes the distance matrix as a heatmap.

    Parameters:
    - tsp_problem: An instance containing TSP problem details (cities, etc.).
    - save_path (str): Path to save the generated heatmap image.
    - show (bool): If True, displays the plot.

    Explanation:
    - Ensures the save directory exists.
    - Computes the distance between every pair of cities using Euclidean distance.
    - Creates a heatmap of the distance matrix.
    - Annotates each cell with the computed distance, formatted to one decimal place.
    - Saves the heatmap image and optionally displays it.
    """
    # Ensure that the directory to save the image exists.
    ensure_dir(save_path)
    # Retrieve the list of city coordinates.
    cities = tsp_problem.cities
    # Determine the number of cities.
    num_cities = len(cities)
    # Initialize a matrix of zeros with shape (num_cities x num_cities).
    matrix = np.zeros((num_cities, num_cities))
    # Iterate over each pair of cities to compute their Euclidean distance.
    for i in range(num_cities):
        for j in range(num_cities):
            # Calculate differences in x and y coordinates.
            dx = cities[i][0] - cities[j][0]
            dy = cities[i][1] - cities[j][1]
            # Compute the Euclidean distance and store it in the matrix.
            matrix[i, j] = np.sqrt(dx ** 2 + dy ** 2)
    # Create a new figure for the heatmap.
    plt.figure(figsize=(8, 6))
    # Display the distance matrix as an image with a blue color map.
    plt.imshow(matrix, cmap='Blues')
    # Annotate each cell with its distance value.
    for i in range(num_cities):
        for j in range(num_cities):
            plt.text(j, i, f"{matrix[i, j]:.1f}", ha='center', va='center', color='black', fontsize=8)
    # Set the title and axis labels.
    plt.title("Distance Matrix Heatmap")
    plt.xlabel("City Index")
    plt.ylabel("City Index")
    # Add a colorbar to the side of the heatmap to indicate the distance scale.
    plt.colorbar(label='Distance')
    # Save the heatmap to the specified file path.
    plt.savefig(save_path)
    # Optionally display the plot if show is True.
    if show:
        plt.show()

def visualize_generations(generation_history, save_path="../graphs/generations.png", show=False):
    """
    Visualizes the progress of the best cost over generations.

    Parameters:
    - generation_history (list): List containing the best cost for each generation.
    - save_path (str): File path to save the generated plot.
    - show (bool): If True, displays the plot.

    Explanation:
    - Ensures the save directory exists.
    - Plots the best cost per generation as a line chart with markers.
    - Configures the chart with title, labels, and grid.
    - Saves the plot to the specified file path and optionally displays it.
    """
    # Ensure that the directory for saving the plot exists.
    ensure_dir(save_path)
    # Create a new figure with a specified size.
    plt.figure(figsize=(12, 6))
    # Generate a sequence for the x-axis representing generation numbers.
    generations = range(1, len(generation_history) + 1)
    # Plot the generation history as a line plot with circular markers.
    plt.plot(generations, generation_history, marker='o', linestyle='-', color='steelblue')
    # Set the title and axis labels for clarity.
    plt.title("Best Cost Progress Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Best Cost")
    # Add a grid to the plot for easier visualization of values.
    plt.grid(c='paleturquoise')
    # Save the generated plot to the specified path.
    plt.savefig(save_path)
    # Optionally display the plot if show is True.
    if show:
        plt.show()