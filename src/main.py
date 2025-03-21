"""
main.py

This module serves as the entry point for the Genetic Algorithm solution to the Traveling Salesman Problem (TSP).
It integrates the TSP problem setup, the genetic algorithm for route optimization, and visualization of the results.

Main functionalities:
- Parse command-line arguments to set the number of cities and generations.
- Initialize the TSP problem instance.
- Create and run the Genetic Algorithm to optimize the route.
- Print out the best route and its cost.
- Generate visualizations for the best route, distance matrix, and cost progress over generations.

Usage:
    python main.py --cities <number_of_cities> --generations <number_of_generations>
"""

import argparse
# Import the GeneticAlgorithm class from the genetic_algorithm module.
from genetic_algorithm import GeneticAlgorithm
# Import the TSPProblem class from the tsp_problem module.
from tsp_problem import TSPProblem
# Import visualization functions for plotting results.
from visualization import (
    visualize_best_route_with_arrows,
    visualize_distance_matrix,
    visualize_generations,
)

def main():
    """
    Main function to run the Genetic Algorithm for the TSP problem and visualize results.

    Steps:
    1. Parse command-line arguments for the number of cities and generations.
    2. Initialize the TSP problem with the specified number of cities.
    3. Create a GeneticAlgorithm instance with the TSP problem and the number of generations.
    4. Run the Genetic Algorithm to obtain the best route, its cost, and the cost progression history.
    5. Output the best route and cost to the console.
    6. Visualize the best route, the distance matrix, and the generation progress using dedicated functions.
    """
    # Create an ArgumentParser instance to parse command-line arguments.
    parser = argparse.ArgumentParser(
        description="Genetic Algorithm for the Traveling Salesman Problem"
    )
    # Add an argument for specifying the number of cities (default is 5).
    parser.add_argument("--cities", type=int, default=10, help="Number of cities in the TSP problem")
    # Add an argument for specifying the number of generations (default is 1).
    parser.add_argument("--generations", type=int, default=1000, help="Number of generations to run")
    # Parse the provided arguments.
    args = parser.parse_args()

    # Initialize the TSP problem with the number of cities provided by the user.
    tsp = TSPProblem(num_cities=args.cities)

    # Create the GeneticAlgorithm instance by providing the TSP problem and generations parameter.
    ga = GeneticAlgorithm(tsp_problem=tsp, generations=args.generations)

    # Run the Genetic Algorithm:
    # - 'show=True' enables detailed debugging output during the run.
    # - Returns the best route found, its total cost, and the history of the best cost per generation.
    best_route, best_cost, best_cost_history = ga.run(show=False)

    # Output the best route and its cost to the console.
    print("Best route:", best_route)
    print("Best cost:", best_cost)

    # Visualize the results using the provided visualization functions:
    # Plot the best route with arrows indicating the sequence of cities.
    visualize_best_route_with_arrows(tsp, best_route)
    # Visualize the distance matrix between all cities as a heatmap.
    visualize_distance_matrix(tsp)
    # Visualize the evolution of the best cost over the generations.
    visualize_generations(best_cost_history)

# Ensure that main() is called when the script is executed directly.
if __name__ == "__main__":
    main()
