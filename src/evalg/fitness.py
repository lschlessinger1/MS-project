import itertools
from typing import List, Callable, Optional, Iterable

import numpy as np
from scipy.spatial.distance import squareform

from src.evalg.encoding import BinaryTreeNode, BinaryTree


def shared_fitness_scores(individuals,
                          raw_fitness_scores,
                          metric: Callable,
                          share_radius: float = 7.5,
                          alpha: float = 1):
    """Compute shared fitness scores

    Fitness sharing aims to allocate individuals to niches in proportion to the niche fitness. Consider all possible
    pairs of individuals and calculates distance d(i, j) between them. Raw fitness F is adjusted according to number of
    individuals falling within some constant radius sigma_share using a power-law distribution.

    F'(i) = F(i) / sum_j (sh(d(i, j))), where

    sh(d) = { 1 - (d/sigma_share)^alpha, if d <= sigma_share
              0                        , otherwise

    Goldberg, David E., and Jon Richardson. "Genetic algorithms with sharing for multimodal function optimization."
    Genetic algorithms and their applications: Proceedings of the Second International Conference on Genetic
    Algorithms. Hillsdale, NJ: Lawrence Erlbaum, 1987.


    :param individuals: Items in a population
    :param raw_fitness_scores: Unscaled fitness scores.
    :param metric: Distance metric between pairs of individuals. Can be genotypic or phenotypic (preferred).
    :param share_radius: Decides both how many niches can be maintained and the granularity with which different niches
    can be discriminated. A default range of 5 - 10 is suggested, unless the number of niches in known in advance.
    AKA sigma_share
    :param alpha: Shape parameter. Determines the shape of the sharing function: for alpha=1, the function is linear,
    but for values greater than this the effect of similar individuals in reducing a solution's fitness falls off more
    rapidly with distance.
    :return: The shared fitness values.
    """
    dist_matrix = compute_distance(individuals, metric)
    return shared_fitness(dist_matrix, raw_fitness_scores, share_radius, alpha)


def shared_fitness(distance_matrix: np.ndarray,
                   raw_fitness_scores,
                   share_radius: float = 7.5,
                   alpha: float = 1.):
    """Only using a distance matrix."""
    shared_dists = np.where(distance_matrix <= share_radius, 1 - (distance_matrix / share_radius) ** alpha, 0)
    return raw_fitness_scores / np.sum(shared_dists, axis=0)


def compute_distance(items: Iterable, metric: Callable):
    # items iterable, metric, callable two args of type items, returning float
    """Compute a distance matrix between all individuals given a metric."""
    dists = np.array([metric(a, b) for a, b in itertools.combinations(items, 2)])
    return squareform(dists)


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


# TODO: make this work with any general tree type
def structural_hamming_dist(tree_1: BinaryTree,
                            tree_2: BinaryTree,
                            hd: Optional[Callable[[BinaryTreeNode, BinaryTreeNode], float]] = None) -> float:
    """Structural Hamming distance (SHD)

    A syntactic distance measure between trees ranging from 0 (trees are equal) to a maximum distance of 1.

    Moraglio and Poli (2005)
    """
    if hd is None:
        hd = _hd
    return shd(tree_1.root, tree_2.root, hd)


def shd(node_1: BinaryTreeNode,
        node_2: BinaryTreeNode,
        hd: Callable[[BinaryTreeNode, BinaryTreeNode], float]) -> float:
    """Structural Hamming distance (SHD)

    :param node_1:
    :param node_2:
    :param hd:
    :return:
    """
    if node_1 is None or node_2 is None:
        return 1
    # first get arity of each node
    arity_1 = 0
    arity_2 = 0
    if node_1.has_left_child():
        arity_1 += 1
    if node_1.has_right_child():
        arity_1 += 1
    if node_2.has_left_child():
        arity_2 += 1
    if node_2.has_right_child():
        arity_2 += 1

    if arity_1 != arity_2:
        return 1
    else:
        if arity_1 == 0:
            # both are leaves
            return hd(node_1, node_2)
        else:
            m = arity_1
            ham_dist = hd(node_1, node_2)
            children_dist_sum = sum([shd(node_1.left, node_2.left, hd), shd(node_1.right, node_2.right, hd)])
            return (1 / (m + 1)) * (ham_dist + children_dist_sum)


def _hd(node_1: BinaryTreeNode,
        node_2: BinaryTreeNode) -> float:
    """Hamming distance between p and q

    0 if p = q (Both terminal nodes of equal value)
    1 otherwise (different terminal node type or internal node)
    """
    if node_1.is_leaf() and node_2.is_leaf() and node_1.value == node_2.value:
        return 0
    else:
        return 1
