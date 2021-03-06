from typing import List, Callable, Tuple

import numpy as np

# Decorators
from src.evalg.serialization import Serializable


def check_gte_two_parents(f: Callable) -> Callable:
    """Check for at least two parents.

    :param f:
    :return:
    """
    def wrapper(self, parents):
        if len(parents) < 2:
            raise ValueError('At least two parents are required.')
        return f(self, parents)

    return wrapper


def check_two_parents(f: Callable) -> Callable:
    """Check for exactly two parents.

    :param f:
    :return:
    """
    def wrapper(self, parents):
        if len(parents) != 2:
            raise ValueError('Exactly two parents are required.')
        return f(self, parents)

    return wrapper


class Recombinator(Serializable):

    @check_gte_two_parents
    def crossover(self, parents: list) -> list:
        """Crossover all parents.

        :param parents:
        :return:
        """
        raise NotImplementedError("crossover must be implemented in a child class")


class OnePointBinaryRecombinator(Recombinator):

    @check_two_parents
    def crossover(self, parents: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """One-point crossover.

        :param parents:
        :return:
        """
        parent_1 = parents[0]
        parent_2 = parents[1]
        gene_size = parent_1.size

        crossover_point = np.random.randint(0, gene_size)
        child_1 = np.hstack((parent_1[:crossover_point], parent_2[crossover_point:]))
        child_2 = np.hstack((parent_2[:crossover_point], parent_1[crossover_point:]))

        return child_1, child_2


class TwoPointBinaryRecombinator(Recombinator):

    @check_two_parents
    def crossover(self, parents: List[np.ndarray]) -> List[np.ndarray]:
        """Two-point crossover.

        :param parents:
        :return:
        """
        recombinator = NPointBinaryRecombinator(n_points=2)
        return recombinator.crossover(parents)


class NPointBinaryRecombinator(Recombinator):
    _n_points: int

    def __init__(self, n_points: int):
        self._n_points = n_points

    @property
    def n_points(self) -> int:
        return self._n_points

    @n_points.setter
    def n_points(self, n_points: int) -> None:
        if n_points < 1:
            raise ValueError('n_points must be at least 1')
        self._n_points = n_points

    @check_two_parents
    def crossover(self, parents: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """n-point crossover.

        :param parents:
        :return:
        """
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
    def crossover(self, parents: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover.

        :param parents:
        :return:
        """
        raise NotImplementedError("crossover_uniform not yet implemented")
