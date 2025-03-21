# Genetic Algorithm for the Traveling Salesman Problem (TSP)

This repository implements a Genetic Algorithm to solve the Traveling Salesman Problem (TSP) and includes several visualization utilities to display the results. The project is designed for educational purposes and demonstrates how evolutionary algorithms can be applied to combinatorial optimization problems.

## Overview

The project consists of the following main components:

- **genetic_algorithm.py:** Contains the `GeneticAlgorithm` class that implements the genetic algorithm. It includes methods for creating random routes, performing selection, crossover, and mutation, and running the evolutionary process over multiple generations.
- **tsp_problem.py:** Defines the `TSPProblem` class which generates a TSP instance with a specified number of cities with random coordinates. It also calculates the total distance of a given route using the Euclidean distance.
- **visualization.py:** Provides functions to visualize the best route (with arrows), the distance matrix (as a heatmap), and the progression of the best cost over generations.
- **main.py:** Serves as the entry point of the application. It parses command-line arguments, initializes the TSP problem and genetic algorithm, runs the algorithm, and calls the visualization functions to display results.

## Features

- **Random TSP Generation:** Create TSP instances with customizable number of cities and coordinate bounds.
- **Genetic Algorithm:** Utilize selection, crossover, and mutation to evolve solutions over generations.
- **Visualization Tools:** Visualize the best route, distance matrix heatmap, and generation-wise progress.
- **Command-Line Interface:** Easily run experiments with adjustable parameters via command-line arguments.

## Dependencies

Make sure you have the following Python packages installed:

- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)

You can install these dependencies using pip:

```bash
pip install numpy matplotlib
```

## Usage

Run the main application using the command line. You can customize the number of cities and generations:

```bash
python main.py --cities 20 --generations 100
```

- `--cities`: Specifies the number of cities in the TSP problem (default is 5).
- `--generations`: Specifies the number of generations for the genetic algorithm (default is 1).

When executed, the program will:
1. Generate a TSP instance with the given number of cities.
2. Run the genetic algorithm to find the best route.
3. Print the best route and its cost to the console.
4. Generate visualizations:
   - Best route plotted with arrows.
   - Distance matrix heatmap.
   - Generation progress plot showing best cost over time.

## Repository Structure

```
.
├── src
│   ├── genetic_algorithm.py    # Genetic algorithm implementation for TSP.
│   ├── tsp_problem.py          # Generates a TSP problem instance and calculates route costs.
│   ├── visualization.py        # Visualization utilities for TSP results.
│   └── main.py                 # Entry point for running the genetic algorithm and visualizations.
├── graphs                      # Folder where all generated plots are saved.
│   ├── best_route_arrows.png
│   ├── distance_matrix.png
│   └── generations.png
└── README.md                   # This file.
```

## Customization

- **Algorithm Parameters:** You can change the population size, number of generations, or mutation rate by modifying the `GeneticAlgorithm` class in `genetic_algorithm.py`.
- **TSP Instance:** Adjust the number of cities or the coordinate boundaries in `TSPProblem` to simulate different scenarios.
- **Visualization:** Modify visualization settings in `visualization.py` (such as figure sizes, colors, and annotations) to tailor the plots to your needs.
