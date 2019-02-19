from abc import ABC

import numpy as np


# Decorators

def check_gte_two_parents(decorated):
    def func_wrapper(parents):
        if len(parents) < 2:
            raise ValueError('At least two parents are required.')
        return decorated(parents)

    return func_wrapper


def check_two_parents(decorated):
    def func_wrapper(parents):
        if len(parents) != 2:
            raise ValueError('Exactly two parents are required.')
        return decorated(parents)

    return func_wrapper


class Recombinator:

    @check_gte_two_parents
    def crossover(self, parents):
        raise NotImplementedError("crossover must be implemented in a child class")


class BinaryRecombinator(Recombinator, ABC):

    @check_two_parents
    def crossover(self, parents):
        raise NotImplementedError('crossover must be implemented in a child class')


class OnePointBinaryRecombinator(BinaryRecombinator):

    def crossover(self, parents):
        """One-point crossover."""
        parent_1 = parents[0]
        parent_2 = parents[1]
        gene_size = parent_1.size

        crossover_point = np.random.randint(0, gene_size)
        child_1 = np.hstack((parent_1[:crossover_point], parent_2[crossover_point:]))
        child_2 = np.hstack((parent_2[:crossover_point], parent_1[crossover_point:]))

        return child_1, child_2


class TwoPointBinaryRecombinator(BinaryRecombinator):

    def crossover(self, parents):
        """Two-point crossover."""
        recombinator = NPointBinaryRecombinator(n=2)
        return recombinator.crossover(parents)


class NPointBinaryRecombinator(BinaryRecombinator):

    def __init__(self, n):
        self.n = n

    def crossover(self, parents):
        """n-point crossover"""
        # TODO: use np.where instead
        parent_1 = parents[0]
        parent_2 = parents[1]
        gene_size = parent_1.size

        crossover_points = sorted(np.random.choice(np.arange(0, gene_size), size=self.n, replace=False))
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
        return f'{self.__class__.__name__}('f'n={self.n!r})'


class UniformBinaryRecombinator(BinaryRecombinator):

    def crossover(self, parents):
        """Uniform crossover."""
        raise NotImplementedError("crossover_uniform not yet implemented")
