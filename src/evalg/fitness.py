from typing import List

import numpy as np


def parsimony_pressure(fitness: float,
                       size: int,
                       p_coeff: float) -> float:
    """Parsimony pressure method.

    Koza, 1992; Zhang & Muhlenbein, 1993; Zhang et al., 1993

    :param fitness: Original fitness
    :param size: Size of individual
    :param p_coeff: Parsimony coefficient
    :return:
    """
    return fitness - p_coeff * size


def covariant_parsimony_pressure(fitness: float,
                                 size: int,
                                 fitness_list: List[float],
                                 sizes: List[float]) -> float:
    """Covariant parsimony pressure method.

    Recalculates the parsimony coefficient each generation

    Poli & McPhee, 2008b

    :param fitness:
    :param size:
    :param fitness_list:
    :param sizes:
    :return:
    """
    cov = np.cov(sizes, fitness_list)
    cov_lf = cov[0, 1]
    var_l = cov[0, 0]
    c = cov_lf / var_l
    return parsimony_pressure(fitness, size, c)
