from typing import List, Iterable, Sequence, TypeVar


def arg_sort(unsorted_iterable: Iterable) -> List[int]:
    """Argument-sort an iterable.

    :param unsorted_iterable: An iterable of unsorted items.
    :return: A list of sorted items' indices.
    """
    return [i[0] for i in sorted(enumerate(unsorted_iterable), key=lambda x: x[1])]


def arg_unique(data: Iterable) -> List[int]:
    """Get the indices of the unique elements in an iterable.

    :param data: An iterable for which to find unique values.
    :return: The indices of unique items of the iterable.
    """
    unique_vals = set()
    unique_ind = []

    for i, datum in enumerate(data):
        if datum not in unique_vals:
            unique_vals.add(datum)
            unique_ind.append(i)

    return unique_ind


T = TypeVar('T')


def remove_duplicates(data: Sequence,
                      values: Sequence[T]) -> List[T]:
    """Remove duplicates of data and replace with values.

    :param data: A sequence of data items.
    :param values: A sequence of values.
    :return: A list of values without duplicates.
    """
    if len(data) != len(values):
        raise ValueError('The length of data must be be equal to the length of values')

    unique_ind = arg_unique(data)
    return [values[i] for i in unique_ind]
