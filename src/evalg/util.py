from typing import TypeVar, List

T = TypeVar('T')


def swap(arr: List[T], i: int, j: int) -> List[T]:
    """Swap two array elements.

    :param arr:
    :param i:
    :param j:
    :return:
    """
    arr[i], arr[j] = arr[j], arr[i]
    return arr
