def argsort(unsorted_list):
    return [i[0] for i in sorted(enumerate(unsorted_list), key=lambda x: x[1])]


def argunique(data):
    """ Get the indices of the unique elements in a list
    """
    unique_vals = set()
    unique_ind = []

    for i, datum in enumerate(data):
        if datum not in unique_vals:
            unique_vals.add(datum)
            unique_ind.append(i)

    return unique_ind


def remove_duplicates(data, values):
    """ Remove duplicates of data and replace with values
    """
    if len(data) != len(values):
        raise ValueError('The length of data must be be equal to the length of values')

    unique_ind = argunique(data)
    return [values[i] for i in unique_ind]
