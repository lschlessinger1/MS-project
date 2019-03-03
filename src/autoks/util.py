from typing import List, Iterable


def arg_sort(unsorted_iterable: Iterable) -> List[int]:
    """Argument-sort an iterable.

    :param unsorted_iterable: An iterable of unsorted items.
    :return: A list of sorted items' indices.
    """
    return [i[0] for i in sorted(enumerate(unsorted_iterable), key=lambda x: x[1])]


def arg_unique(data: list) -> List[int]:
    """Get the indices of the unique elements in a list.

    :param data:
    :return:
    """
    unique_vals = set()
    unique_ind = []

    for i, datum in enumerate(data):
        if datum not in unique_vals:
            unique_vals.add(datum)
            unique_ind.append(i)

    return unique_ind


def remove_duplicates(data: list, values: list) -> list:
    """Remove duplicates of data and replace with values.

    :param data:
    :param values:
    :return:
    """
    if len(data) != len(values):
        raise ValueError('The length of data must be be equal to the length of values')

    unique_ind = arg_unique(data)
    return [values[i] for i in unique_ind]
