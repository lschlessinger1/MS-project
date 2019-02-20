def argsort(unsorted_list: list):
    """Argument-sort a list

    :param unsorted_list: a list of unsorted items
    :return:
    """
    return [i[0] for i in sorted(enumerate(unsorted_list), key=lambda x: x[1])]


def argunique(data: list):
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


def remove_duplicates(data: list, values: list):
    """Remove duplicates of data and replace with values.

    :param data:
    :param values:
    :return:
    """
    if len(data) != len(values):
        raise ValueError('The length of data must be be equal to the length of values')

    unique_ind = argunique(data)
    return [values[i] for i in unique_ind]
