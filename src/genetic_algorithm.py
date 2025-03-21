"""
genetic_algorithm.py

This module implements a Genetic Algorithm for solving a Traveling Salesman Problem (TSP).
The algorithm evolves a population of candidate routes through selection, crossover, and mutation
to gradually find a route with minimal total travel cost.

Key Components:
- GeneticAlgorithm class: Contains methods to create routes, generate initial populations,
  calculate fitness, perform selection (roulette wheel), crossover, and mutation operations.
- Each function is documented to explain what it does, how inputs are processed, and how outputs are computed.
- Detailed inline comments provide step-by-step explanations for all calculations and data flows.

Assumptions:
- The TSP problem instance (passed as tsp_problem) must have:
  - A list of cities in tsp_problem.cities.
  - A method calculate_cost(route) that computes the total distance or cost for a given route.
- The algorithm uses the inverse of cost as the fitness function.
- For mutation, a swap mutation strategy is employed, where genes (cities) may be swapped based on a given mutation rate.
"""

import random


class GeneticAlgorithm:
    def __init__(self, tsp_problem, population_size=10, generations=10, mutation_rate=0.2):
        """
        Initialize the GeneticAlgorithm instance.

        Parameters:
        - tsp_problem: An instance representing the TSP problem, containing cities and a cost function.
        - population_size (int): The number of candidate solutions (routes) in each generation.
        - generations (int): The number of iterations the algorithm will run.
        - mutation_rate (float): The probability for each gene (city) to be mutated during mutation.
        """
        # Save the TSP problem instance which provides the cities and cost calculation.
        self.tsp_problem = tsp_problem
        # Number of candidate solutions maintained in the population.
        self.population_size = population_size
        # Number of iterations (generations) to run the evolution process.
        self.generations = generations
        # Mutation probability for altering genes (swapping cities).
        self.mutation_rate = mutation_rate

    def create_route(self):
        """
        Create a random route (permutation of city indices) for the TSP.

        Returns:
        - A list of integers representing a random permutation of the cities.

        Explanation:
        - Generates a list of indices corresponding to the cities in the TSP.
        - Shuffles the list randomly to produce a potential route.
        """
        # Generate a list from 0 to number of cities - 1.
        route = list(range(len(self.tsp_problem.cities)))
        # Randomly shuffle the route to create a random permutation.
        random.shuffle(route)
        return route

    def initial_population(self):
        """
        Generate the initial population of routes.

        Returns:
        - A list of routes, where each route is a permutation of city indices.

        Explanation:
        - Calls create_route() repeatedly to build a list of random candidate solutions.
        """
        # Use list comprehension to create a population of given size.
        return [self.create_route() for _ in range(self.population_size)]

    def route_cost(self, route):
        """
        Calculate the total cost (distance) of a given route.

        Parameters:
        - route: A list representing the order in which cities are visited.

        Returns:
        - The cost calculated by the tsp_problem instance.

        Explanation:
        - Delegates the calculation of route cost to the tsp_problem instance.
        """
        return self.tsp_problem.calculate_cost(route)

    def fitness(self, route):
        """
        Compute the fitness value of a given route.

        Parameters:
        - route: A candidate solution route (list of city indices).

        Returns:
        - A fitness value where higher fitness corresponds to a lower cost.

        Explanation:
        - The fitness is calculated as the inverse of the route cost.
        - If cost is zero (to avoid division by zero), returns infinity.
        """
        # Compute cost of the route.
        cost = self.route_cost(route)
        # Return inverse cost as fitness, ensuring no division by zero.
        return 1 / cost if cost > 0 else float('inf')

    def selection(self, population, fitnesses):
        """
        Select a route from the population using roulette wheel selection.

        Parameters:
        - population: A list of routes.
        - fitnesses: A list of fitness values corresponding to each route.

        Returns:
        - A selected route based on its fitness probability.

        Explanation:
        - Total fitness is computed by summing all fitness values.
        - A random pick is made between 0 and the total fitness.
        - Iterate over population and accumulate fitness until the running sum exceeds the pick.
        - The route at that point is selected.
        """
        # Sum all fitness values to determine the roulette wheel's range.
        total_fitness = sum(fitnesses)
        # Pick a random threshold between 0 and the total fitness.
        pick = random.uniform(0, total_fitness)
        current = 0
        # Iterate through each route and its fitness.
        for route, fit in zip(population, fitnesses):
            current += fit  # Accumulate the fitness value.
            # When the accumulated fitness exceeds the threshold, select the route.
            if current > pick:
                return route
        # If no route is selected due to rounding, return the last route.
        return population[-1]

    def crossover(self, parent1, parent2, print_parents=False):
        """
        Perform crossover between two parent routes to produce a child route.

        Parameters:
        - parent1: The first parent route (list of city indices).
        - parent2: The second parent route.
        - print_parents (bool): If True, print intermediate child information for debugging.

        Returns:
        - A child route resulting from combining segments of both parents.

        Explanation:
        - A random subsequence is chosen from parent1.
        - The corresponding positions in the child are filled with the chosen subsequence.
        - The remaining positions are filled with cities from parent2 in the order they appear,
          skipping any cities already present in the child.
        """
        # Determine the size of the routes (assumed equal for both parents).
        size = len(parent1)
        # Randomly choose two distinct indices and sort them to define a subsequence.
        start, end = sorted(random.sample(range(size), 2))
        # Initialize the child route with placeholders (None).
        child = [None] * size
        # Copy the subsequence from parent1 into the child.
        child[start:end] = parent1[start:end]
        if print_parents:
            print(f"Child from Parent 1: {child}")
        # Start filling missing positions from parent2, maintaining order.
        current_idx = end
        for gene in parent2:
            # If the gene (city index) is not already in the child, place it in the child.
            if gene not in child:
                # If we've reached the end of the child, wrap around to the beginning.
                if current_idx >= size:
                    current_idx = 0
                child[current_idx] = gene
                current_idx += 1
        if print_parents:
            print(f"Child with Parent 2: {child}")
        return child

    def mutate(self, route, print_mutation=False):
        """
        Apply mutation to a route using the swap mutation strategy.

        Parameters:
        - route: The route to mutate (list of city indices).
        - print_mutation (bool): If True, print the route before and after mutation for debugging.

        Returns:
        - The mutated route after performing swap mutations.

        Explanation:
        - For each gene (city index) in the route, decide randomly (based on mutation_rate)
          whether to perform a swap with another randomly chosen gene.
        - This introduces small random changes to maintain diversity in the population.
        """
        # Create a copy of the route to avoid modifying the original.
        mutated = route.copy()
        if print_mutation:
            print(f"Original: {mutated}")
        # Iterate over each position in the route.
        for i in range(len(mutated)):
            # Decide if mutation should occur based on the mutation rate.
            if random.random() < self.mutation_rate:
                # Choose a random position in the route to swap with.
                j = random.randint(0, len(mutated) - 1)
                # Swap the cities at positions i and j.
                mutated[i], mutated[j] = mutated[j], mutated[i]
        if print_mutation:
            print(f"Mutation: {mutated}")
        return mutated

    def run(self, show=False):
        """
        Run the Genetic Algorithm to optimize the TSP route.

        Parameters:
        - show (bool): If True, print additional debug information for each generation.

        Returns:
        - best_route: The best route found after all generations.
        - best_cost: The cost of the best route.
        - best_cost_history: A list tracking the best cost at each generation.

        Explanation:
        - Initializes the population and iterates over a number of generations.
        - In each generation, the fitness of each route is calculated.
        - A new population is generated by selecting parents, performing crossover, and applying mutation.
        - The best route and cost are updated based on the routes in the new population.
        - Debug output can be enabled with the show flag to trace population changes.
        """
        # Generate the initial population of candidate routes.
        population = self.initial_population()
        # Initialize best_route and best_cost to track the best solution found.
        best_route = None
        best_cost = float('inf')
        best_cost_history = []  # Record the best cost for each generation.

        # Iterate through each generation.
        for generation in range(self.generations):
            # Calculate fitness for each route in the current population.
            fitnesses = [self.fitness(route) for route in population]
            new_population = []

            # Create a new population by performing selection, crossover, and mutation.
            for _ in range(self.population_size):
                # Select two parents using roulette wheel selection.
                parent1 = self.selection(population, fitnesses)
                parent2 = self.selection(population, fitnesses)
                # Create a child route by combining parts of both parents.
                child = self.crossover(parent1, parent2, print_parents=show)
                # Apply mutation to the child route.
                child = self.mutate(child, print_mutation=show)
                # Add the new child to the new population.
                new_population.append(child)

            # Replace the old population with the new one.
            population = new_population
            if show:
                print("Population:", population)
            # Evaluate each route in the new population to update the best route found.
            for route in population:
                cost = self.route_cost(route)
                # If a route with a lower cost is found, update best_route and best_cost.
                if cost < best_cost:
                    best_cost = cost
                    best_route = route

            # Record the best cost of the current generation.
            best_cost_history.append(best_cost)
            # Output the best cost for the current generation.
            print(f"Generation {generation + 1}: Best cost = {best_cost}")

        # Return the best found route, its cost, and the history of best costs across generations.
        return best_route, best_cost, best_cost_history