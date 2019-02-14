import numpy as np

from evalg.util import swap


class Mutator:

    def __init__(self, individual):
        self.individual = individual.copy()
        self.L = individual.size

    def mutate(self):
        raise NotImplementedError("Implement mutate in a child class.")


class BitFlipMutator(Mutator):

    def __init__(self, individual, gene_mut_prob=None):
        super().__init__(individual)
        if gene_mut_prob is None:
            self.gene_mut_prob = 1 / self.L
        else:
            self.gene_mut_prob = gene_mut_prob

    def mutate(self):
        """Bit-flip mutation"""
        mutated = np.random.uniform(0, 1, size=self.L) < self.gene_mut_prob
        indiv_mut = np.where(mutated, ~self.individual.astype(bool), self.individual)

        return indiv_mut


class InterchangeMutator(Mutator):

    def __init__(self, individual):
        super().__init__(individual)

    def mutate(self):
        """Interchange mutation"""
        ind = np.random.randint(0, self.L, size=2)
        # swap first and second genes
        indiv_mut = swap(self.individual, ind[0], ind[1])

        return indiv_mut


class ReverseMutator(Mutator):

    def __init__(self, individual):
        super().__init__(individual)

    def mutate(self):
        """Reverse mutation."""
        ind = np.random.randint(0, self.L)
        indiv_mut = swap(self.individual, ind, ind - 1)

        return indiv_mut


class GaussianMutator(Mutator):

    def __init__(self, individual):
        super().__init__(individual)

    def mutate(self):
        """Gaussian mutation."""
        raise NotImplementedError("Gaussian mutation not yet implemented")


class BoundaryMutator(Mutator):

    def __init__(self, individual):
        super().__init__(individual)

    def mutate(self):
        """Boundary mutation."""
        raise NotImplementedError("Boundary mutation not yet implemented")


class UniformMutator(Mutator):

    def __init__(self, individual):
        super().__init__(individual)

    def mutate(self):
        """Uniform mutation."""
        raise NotImplementedError("Uniform mutation not yet implemented")


class NonuniformMutator(Mutator):

    def __init__(self, individual):
        super().__init__(individual)

    def mutate(self):
        """Non-Uniform mutation."""
        raise NotImplementedError("Non-Uniform mutation not yet implemented")


class ShrinkMutator(Mutator):

    def __init__(self, individual):
        super().__init__(individual)

    def mutate(self):
        """Shrink mutation."""
        raise NotImplementedError("Shrink mutation not yet implemented")
