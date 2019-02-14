from abc import ABC

import numpy as np


class Recombinator:

    def __init__(self, parents):
        if len(parents) < 2:
            raise ValueError('At least two parents are required.')

        self.parents = parents
        self.L = parents[0].size

    def crossover(self):
        raise NotImplementedError("crossover must be implemented in a child class")


class BinaryRecombinator(Recombinator, ABC):

    def __init__(self, parents):
        super().__init__(parents)
        if len(parents) != 2:
            raise ValueError('Exactly two parents are required.')
        self.parent_1 = self.parents[0].copy()
        self.parent_2 = self.parents[1].copy()


class OnePointBinaryRecombinator(BinaryRecombinator):

    def __init__(self, parents):
        super().__init__(parents)

    def crossover(self):
        """One-point crossover."""

        crossover_point = np.random.randint(0, self.L)
        child_1 = np.hstack((self.parent_1[:crossover_point], self.parent_2[crossover_point:]))
        child_2 = np.hstack((self.parent_2[:crossover_point], self.parent_1[crossover_point:]))

        return child_1, child_2


class TwoPointBinaryRecombinator(BinaryRecombinator):

    def __init__(self, parents):
        super().__init__(parents)

    def crossover(self):
        """Two-point crossover."""
        recombinator = NPointBinaryRecombinator(self.parents, n=2)
        return recombinator.crossover()


class NPointBinaryRecombinator(BinaryRecombinator):

    def __init__(self, parents, n):
        super().__init__(parents)
        self.n = n

    def crossover(self):
        """n-point crossover"""
        # TODO: use np.where instead

        crossover_points = sorted(np.random.choice(np.arange(0, self.L), size=self.n, replace=False))
        child_1 = self.parent_1
        child_2 = self.parent_2

        for i, cx_point in enumerate(crossover_points):
            if i == len(crossover_points) - 1:
                break
            next_cx_point = crossover_points[i + 1]
            if i % 2 == 0:
                child_1[cx_point:next_cx_point] = self.parent_2[cx_point:next_cx_point]
            else:
                child_2[cx_point:next_cx_point] = self.parent_1[cx_point:next_cx_point]

        return child_1, child_2


class UniformBinaryRecombinator(BinaryRecombinator):

    def __init__(self, parents):
        super().__init__(parents)

    def crossover(self):
        """Uniform crossover."""
        raise NotImplementedError("crossover_uniform not yet implemented")
