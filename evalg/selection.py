import numpy as np


def argselect_uniform(population, k):
    """Uniform Stochastic Selection

    Select the arguments of k individuals
    with replacement from the population uniformly
    at random.
    """
    pop_size = population.shape[0]

    ind = np.random.choice(pop_size, size=k, replace=True)

    return ind


def select_uniform(population, k):
    """Uniform Stochastic Selection

    Select k individuals with replacement from the
    population uniformly at random.
    """

    ind = argselect_uniform(population, k)
    individuals = population[ind]

    return individuals


def select_fitness_proportional(population, fitness_list, k):
    """Fitness-Proportional Selection

    Select k individuals with replacement from the population
    uniformly at random proportional to the fitness of
    the individual. This is also known as roulette-wheel
    selection or stochastic sampling with replacement.
    """
    pop_size = population.shape[0]

    probabilities = fitness_list / np.sum(fitness_list)
    ind = np.random.choice(pop_size, size=k, replace=True, p=probabilities)
    individuals = population[ind]

    return individuals


def select_stochast_univ_samp():
    """ Stochastic Universal Sampling
    """
    raise NotImplementedError("select_stochast_univ_samp not yet implemented")


def select_sigma_scaling(population, fitness_list, k):
    """Sigma scaling selection
    """
    pop_size = population.shape[0]

    sigma = np.std(fitness_list)
    expected_cnts = np.empty(pop_size)
    if sigma > 0.0001:
        expected_cnts[:] = 1 + (fitness_list - np.mean(fitness_list)) / sigma
    else:
        expected_cnts[:] = 1

    max_exp_cnt = 1.5
    min_exp_cnt = 0

    expected_cnts[expected_cnts > max_exp_cnt] = max_exp_cnt
    expected_cnts[expected_cnts < min_exp_cnt] = min_exp_cnt

    probabilities = expected_cnts / np.sum(expected_cnts)
    ind = np.random.choice(pop_size, size=k, replace=True, p=probabilities)
    individuals = population[ind]

    return individuals


def select_boltzmann(population, fitness_list, k, T, prev_pop_avg):
    """Boltzmann Selection
    """
    raise NotImplementedError("select_boltzmann not yet implemented")


def select_k_best(population, fitness_list, k):
    """Truncation selection

    Select k best from population according to fitness_list.

    k: number of individuals
    """

    top_k_idxs = np.argpartition(fitness_list, -k)[-k:]
    individuals = population[top_k_idxs]

    return individuals


def select_linear_ranking(population, fitness_list, k):
    """Linear Ranking Selection

    Select k individuals with replacement from the population
    uniformly at random proportional to the relative fitness
    ranking of the individual.
    """
    pop_size = population.shape[0]

    rankings_asc = np.argsort(np.argsort(fitness_list)) + 1
    probabilities = rankings_asc / np.sum(rankings_asc)

    ind = np.random.choice(pop_size, size=k, replace=True, p=probabilities)

    individuals = population[ind]

    return individuals


def select_exponential_ranking(population, fitness_list, k, c=0.99):
    """Exponential Ranking Selection
    """
    if c <= 0 or c >= 1:
        raise ValueError("0 < c < 1 must hold")

    pop_size = population.shape[0]

    rankings_asc = np.argsort(np.argsort(fitness_list))
    probabilities = (c ** (pop_size - rankings_asc)) / np.sum(c ** (pop_size - rankings_asc))

    ind = np.random.choice(pop_size, size=k, replace=True, p=probabilities)

    individuals = population[ind]

    return individuals


def select_tournament(population, fitness_list, k, n_way=2):
    """Tournament Selection
    Uniformly at random select `n_way` individuals from the
    population then selecting the best (or worst) individual
    from the `n_way` competitors as the winner (or loser). There
    will be `k` tournaments run with replacement on the
    population.

    Notes
    -----
    Binary tournaments (`n_way` = 2) are equivalent to linear
    ranking selection in expectation. If `n_way` = 3, it is
    equivalent, in expectation, to quadratic ranking selection.
    """

    if n_way < 2 or n_way > len(population):
        raise ValueError("The number of competitors in the \
                         tournament must be greater than 1 and less \
                         than the number of individuals in the \
                         population")

    individuals = []
    for i in range(k):
        ind = argselect_uniform(population, n_way)
        winner = np.argmax(fitness_list[ind], axis=0)
        individuals.append(population[winner])

    return individuals
