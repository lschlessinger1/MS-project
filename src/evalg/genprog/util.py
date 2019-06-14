from typing import Callable, List

from src.evalg.encoding import BinaryTree


def check_binary_trees(f: Callable) -> Callable:
    def wrapper(self, parents: List[BinaryTree]):
        if not all(isinstance(parent, BinaryTree) for parent in parents):
            raise TypeError('all parents must be of type %s' % BinaryTree.__name__)
        return f(self, parents)

    return wrapper
