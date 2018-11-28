import numpy as np


def crossover_all(parents, n_offspring, crossover_operator, n_way=2, n_points=2, c_prob=1.):
    '''Crossover applied to all parents

    n_way: number of parents in crossover
    n_points: number of points in crossover operator
    c_prob: probability of crossover
    '''

    offspring = []
    for i in range(0, n_offspring, n_way):
        if np.random.rand() < c_prob:
            selected_parents = [parents[(i + j) % len(parents)] for j in range(n_way)]
            children = crossover_operator(selected_parents, n_points)

            # add children to offspring
            for j, child in enumerate(children):
                if len(offspring) < n_offspring:
                    offspring.append(child)

    return offspring


def mutate_all(offspring, mutate_operator, m_prob=1.):
    '''Mutation applied to all offspring

    m_prob: probability of mutation
    '''

    offspring_mut = []

    for child in offspring:
        if np.random.rand() < m_prob:
            child = mutate_operator(child)

        offspring_mut.append(child)

    return offspring_mut


def crossover_and_mutate_all(parents, n_offspring, crossover_operator, mutate_operator, c_prob=1., m_prob=1.):
    ''' Perform both crossover and mutation
    '''

    offspring = crossover_all(parents, n_offspring, crossover_operator, c_prob=c_prob)
    offspring = mutate_all(offspring, mutate_operator, m_prob=m_prob)

    return offspring
