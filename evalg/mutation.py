from abc import ABC

import numpy as np

from evalg.util import swap


class Mutator:

    def __init__(self, individual):
        self.individual = individual

    def mutate(self):
        raise NotImplementedError("Implement mutate in a child class.")


class GAMutator(Mutator, ABC):

    def __init__(self, individual):
        super().__init__(individual)
        self.gene_size = individual.size

    def __repr__(self):
        return f'{self.__class__.__name__}('f'individual={self.individual!r}, gene_size={self.gene_size!r})'


class BitFlipMutator(GAMutator):

    def __init__(self, individual, gene_mut_prob=None):
        super().__init__(individual)
        if gene_mut_prob is None:
            self.gene_mut_prob = 1 / self.gene_size
        else:
            self.gene_mut_prob = gene_mut_prob

    def mutate(self):
        """Bit-flip mutation"""
        mutated = np.random.uniform(0, 1, size=self.gene_size) < self.gene_mut_prob
        indiv_mut = np.where(mutated, ~self.individual.astype(bool), self.individual)

        return indiv_mut

    def __repr__(self):
        return f'{self.__class__.__name__}('f'individual={self.individual!r}, gene_size={self.gene_size!r}, ' \
            f'gene_mut_prob={self.gene_mut_prob!r})'


class InterchangeMutator(GAMutator):

    def __init__(self, individual):
        super().__init__(individual)

    def mutate(self):
        """Interchange mutation"""
        ind = np.random.randint(0, self.gene_size, size=2)
        # swap first and second genes
        indiv_mut = swap(self.individual, ind[0], ind[1])

        return indiv_mut

    def __repr__(self):
        return f'{self.__class__.__name__}('f'individual={self.individual!r}, gene_size={self.gene_size!r})'


class ReverseMutator(GAMutator):

    def __init__(self, individual):
        super().__init__(individual)

    def mutate(self):
        """Reverse mutation."""
        ind = np.random.randint(0, self.gene_size)
        indiv_mut = swap(self.individual, ind, ind - 1)

        return indiv_mut

    def __repr__(self):
        return f'{self.__class__.__name__}('f'individual={self.individual!r}, gene_size={self.gene_size!r})'


class GaussianMutator(GAMutator):

    def __init__(self, individual):
        super().__init__(individual)

    def mutate(self):
        """Gaussian mutation."""
        raise NotImplementedError("Gaussian mutation not yet implemented")

    def __repr__(self):
        return f'{self.__class__.__name__}('f'individual={self.individual!r}, gene_size={self.gene_size!r})'


class BoundaryMutator(GAMutator):

    def __init__(self, individual):
        super().__init__(individual)

    def mutate(self):
        """Boundary mutation."""
        raise NotImplementedError("Boundary mutation not yet implemented")

    def __repr__(self):
        return f'{self.__class__.__name__}('f'individual={self.individual!r}, gene_size={self.gene_size!r})'


class UniformMutator(GAMutator):

    def __init__(self, individual):
        super().__init__(individual)

    def mutate(self):
        """Uniform mutation."""
        raise NotImplementedError("Uniform mutation not yet implemented")

    def __repr__(self):
        return f'{self.__class__.__name__}('f'individual={self.individual!r}, gene_size={self.gene_size!r})'


class NonuniformMutator(GAMutator):

    def __init__(self, individual):
        super().__init__(individual)

    def mutate(self):
        """Non-Uniform mutation."""
        raise NotImplementedError("Non-Uniform mutation not yet implemented")

    def __repr__(self):
        return f'{self.__class__.__name__}('f'individual={self.individual!r}, gene_size={self.gene_size!r})'


class ShrinkMutator(GAMutator):

    def __init__(self, individual):
        super().__init__(individual)

    def mutate(self):
        """Shrink mutation."""
        raise NotImplementedError("Shrink mutation not yet implemented")

    def __repr__(self):
        return f'{self.__class__.__name__}('f'individual={self.individual!r}, gene_size={self.gene_size!r})'
