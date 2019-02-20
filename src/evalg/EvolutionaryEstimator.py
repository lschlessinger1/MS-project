import numpy as np

from src.evalg import plotting


class EvolutionaryEstimator:

    def __init__(self, parent_selector, initializer, variator, survivors_selector):
        self.create_population = initializer
        self.select_parents = parent_selector
        self.vary = variator
        self.select_survivors = survivors_selector

    def fit(self, n_individuals, fitness, genotype_size=64, budget=100, display_results=True):
        population = self.create_population(n_individuals, genotype_size=genotype_size)

        # store an extra value because we want to store the max before and after running the GA
        best_so_far = np.zeros(budget + 1)

        for i in range(budget):
            fitness_list = [fitness(parent) for parent in population]
            # save the maximum fitness for each generation
            best_so_far[i] = max(fitness_list)

            parents = self.select_parents(population, fitness_list, n_parents=n_individuals)

            offspring = self.vary(parents, n_offspring=10)

            population = self.select_survivors(offspring, fitness_list, k=10)

        # recompute fitness_list for the last generation
        fitness_list = [fitness(parent) for parent in population]
        best_so_far[-1] = max(fitness_list)
        if display_results:
            plotting.plot_best_so_far(best_so_far)

        return population
