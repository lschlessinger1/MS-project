from collections import Counter


class Population(Counter):
    """A multiset of genotypes.

    Individuals do not change.
    Keys are individuals and values are counts.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def variety(self) -> int:
        """The number of distinct individuals in the population.

        :return:
        """
        return len(self)

    def size(self) -> int:
        """The number of individuals in the population (counting duplicates).

        :return:
        """
        return sum(value for value in self.values())

    def print_all(self):
        for item in self:
            print(item)
