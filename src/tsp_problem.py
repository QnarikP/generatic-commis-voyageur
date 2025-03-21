"""
tsp_problem.py

This module defines the TSPProblem class for generating and evaluating a Traveling Salesman Problem (TSP) instance.
The TSPProblem class is responsible for:
- Generating a specified number of cities with random (x, y) coordinates.
- Calculating the total distance (cost) of a given tour (route) through these cities.
- Computing the Euclidean distance between two cities.

Key Components:
- __init__: Initializes the problem with parameters for the number of cities and the bounds for their coordinates.
- generate_cities: Creates random city coordinates based on given dimensions.
- calculate_cost: Computes the total route distance by summing the Euclidean distances between consecutive cities.
- distance: A static method that calculates the Euclidean distance between two cities.

This file is designed to be used as a module by the genetic algorithm or any other route optimization algorithm.
"""

import numpy as np


class TSPProblem:
    def __init__(self, num_cities=20, width=100, height=100):
        """
        Initializes the TSP problem by generating random city coordinates.

        Parameters:
        - num_cities (int): The number of cities to generate.
        - width (int or float): The maximum x-coordinate value (defines the range for city positions).
        - height (int or float): The maximum y-coordinate value (defines the range for city positions).

        Explanation:
        - Sets the number of cities and the dimensions (width and height) for the coordinate space.
        - Calls generate_cities() to create the list of city coordinates.
        """
        # Store the total number of cities.
        self.num_cities = num_cities
        # Define the maximum x-coordinate value.
        self.width = width
        # Define the maximum y-coordinate value.
        self.height = height
        # Generate and store the list of cities with random coordinates.
        self.cities = self.generate_cities()

    def generate_cities(self):
        """
        Generates a list of cities with random (x, y) coordinates.

        Returns:
        - List of tuples, each representing the (x, y) coordinate of a city.

        Explanation:
        - For each city, a random x value is generated between 0 and width.
        - Similarly, a random y value is generated between 0 and height.
        - The coordinates are stored as a tuple and added to a list.
        """
        # Use a list comprehension to generate the specified number of cities.
        return [
            (
                # Generate a random x-coordinate within the range [0, width].
                np.random.uniform(0, self.width),
                # Generate a random y-coordinate within the range [0, height].
                np.random.uniform(0, self.height)
            )
            for _ in range(self.num_cities)  # Iterate for the number of cities.
        ]

    def calculate_cost(self, route):
        """
        Calculates the total distance (cost) of the given route.

        Parameters:
        - route: List of city indices representing the order in which the cities are visited.

        Returns:
        - Total distance (float) of the tour.

        Explanation:
        - Iterates over each pair of consecutive cities in the route.
        - Uses the distance method to compute the Euclidean distance between each pair.
        - The tour is circular, meaning that after the last city, the route returns to the first city.
        - Sums all individual distances to determine the overall cost.
        """
        # Initialize the total cost of the tour to zero.
        total_cost = 0
        # Loop over each index in the route.
        for i in range(len(route)):
            # Retrieve the current city using the route index.
            city_a = self.cities[route[i]]
            # Retrieve the next city; use modulo to wrap around to the first city after the last.
            city_b = self.cities[route[(i + 1) % len(route)]]
            # Compute the distance between the current city and the next city and add it to total_cost.
            total_cost += self.distance(city_a, city_b)
        # Return the accumulated total cost.
        return total_cost

    @staticmethod
    def distance(city_a, city_b):
        """
        Computes the Euclidean distance between two cities.

        Parameters:
        - city_a: Tuple (x, y) representing the coordinates of the first city.
        - city_b: Tuple (x, y) representing the coordinates of the second city.

        Returns:
        - Euclidean distance (float) between city_a and city_b.

        Explanation:
        - Uses the formula: sqrt((x2 - x1)^2 + (y2 - y1)^2) to calculate the distance.
        """
        # Calculate the squared difference in x-coordinates and y-coordinates.
        return ((city_a[0] - city_b[0]) ** 2 + (city_a[1] - city_b[1]) ** 2) ** 0.5