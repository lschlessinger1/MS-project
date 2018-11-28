import numpy as np


def crossover_one_point(parent_1, parent_2):
    """One-point crossover
    """

    L = parent_1.size
    crossover_point = np.random.randint(0, L)
    child_1 = np.hstack((parent_1[:crossover_point], parent_2[crossover_point:]))
    child_2 = np.hstack((parent_2[:crossover_point], parent_1[crossover_point:]))

    return child_1, child_2


def crossover_two_point(parent_1, parent_2):
    """ Two-point crossover
    """
    return crossover_n_point(parent_1, parent_2, n=2)


def crossover_n_point(parent_1, parent_2, n):
    """ n-point crossover
    """
    # TODO: use np.where instead

    L = parent_1.size
    crossover_points = sorted(np.random.choice(np.arange(0, L), size=n, replace=False))
    child_1 = parent_1.copy()
    child_2 = parent_2.copy()

    for i, cpoint in enumerate(crossover_points):
        if i == len(crossover_points) - 1:
            break
        next_cpoint = crossover_points[i + 1]
        if i % 2 == 0:
            child_1[cpoint:next_cpoint] = parent_2[cpoint:next_cpoint]
        else:
            child_2[cpoint:next_cpoint] = parent_1[cpoint:next_cpoint]

    return child_1, child_2


def crossover_uniform(parent_1, parent_2):
    """Uniform crossover
    """
    raise NotImplementedError("crossover_uniform not yet implemented")
