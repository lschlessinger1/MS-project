from typing import List

import numpy as np


# Decorators

def check_gte_two_parents(f):
    def wrapper(self, parents):
        if len(parents) < 2:
            raise ValueError('At least two parents are required.')
        return f(self, parents)

    return wrapper


def check_two_parents(f):
    def wrapper(self, parents):
        if len(parents) != 2:
            raise ValueError('Exactly two parents are required.')
        return f(self, parents)

    return wrapper


class Recombinator:

    @check_gte_two_parents
    def crossover(self, parents: list):
        raise NotImplementedError("crossover must be implemented in a child class")


class OnePointBinaryRecombinator(Recombinator):

    @check_two_parents
    def crossover(self, parents: List[np.array]):
        """One-point crossover."""
        parent_1 = parents[0]
        parent_2 = parents[1]
        gene_size = parent_1.size

        crossover_point = np.random.randint(0, gene_size)
        child_1 = np.hstack((parent_1[:crossover_point], parent_2[crossover_point:]))
        child_2 = np.hstack((parent_2[:crossover_point], parent_1[crossover_point:]))

        return child_1, child_2


class TwoPointBinaryRecombinator(Recombinator):

    @check_two_parents
    def crossover(self, parents: List[np.array]):
        """Two-point crossover."""
        recombinator = NPointBinaryRecombinator(n_points=2)
        return recombinator.crossover(parents)


class NPointBinaryRecombinator(Recombinator):

    def __init__(self, n_points: int):
        if n_points < 1:
            raise ValueError('n_points must be at least 1')
        self.n_points = n_points

    @check_two_parents
    def crossover(self, parents: List[np.array]):
        """n-point crossover"""
        # TODO: use np.where instead
        parent_1 = parents[0]
        parent_2 = parents[1]
        gene_size = parent_1.size

        crossover_points = sorted(np.random.choice(np.arange(0, gene_size), size=self.n_points, replace=False))
        child_1 = parent_1
        child_2 = parent_2

        for i, cx_point in enumerate(crossover_points):
            if i == len(crossover_points) - 1:
                break
            next_cx_point = crossover_points[i + 1]
            if i % 2 == 0:
                child_1[cx_point:next_cx_point] = parent_2[cx_point:next_cx_point]
            else:
                child_2[cx_point:next_cx_point] = parent_1[cx_point:next_cx_point]

        return child_1, child_2

    def __repr__(self):
        return f'{self.__class__.__name__}('f'n={self.n_points!r})'


class UniformBinaryRecombinator(Recombinator):

    @check_two_parents
    def crossover(self, parents: List[np.array]):
        """Uniform crossover."""
        raise NotImplementedError("crossover_uniform not yet implemented")
