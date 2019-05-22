from typing import List, Iterable, Sequence, TypeVar, Type


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


def tokenize(list_nd: list) -> list:
    """Tokenize a list.

    :param list_nd:
    :return:
    """
    if not list_nd:
        return []
    if isinstance(list_nd, list):
        return ['('] + [tokenize(s) for s in list_nd] + [')']
    return list_nd


def flatten(list_nd: list) -> list:
    """Flatten a list.

    :param list_nd:
    :return:
    """
    return [list_nd] if not isinstance(list_nd, list) else [x for X in list_nd for x in flatten(X)]


T2 = TypeVar('T2')


def remove_outer_parens(list_nd: List[T2]) -> List[T2]:
    """Remove outer parentheses from a list of tokens.

    :param list_nd:
    :return:
    """
    if len(list_nd) >= 2:
        if list_nd[0] == '(' and list_nd[-1] == ')':
            return list_nd[1:-1]
        else:
            raise ValueError('List must start with \'(\' and end with \')\' ')
    else:
        raise ValueError('List must have length >= 2')


def join_operands(operands: list,
                  operator: str) -> list:
    """Join operands using operators

    :param operands:
    :param operator:
    :return:
    """
    joined = []
    for i, operand in enumerate(operands):
        joined += [operand]
        if i < len(operands) - 1:
            joined += [operator]
    return joined


def type_count(a: Iterable,
               cls: Type) -> int:
    """Count how many items of a given type there are in an iterable.

    :param a:
    :param cls:
    :return:
    """
    return sum(isinstance(x, cls) for x in a)


def pretty_time_delta(seconds: float) -> str:
    """Return a human-readable string of a duration in seconds.

    modified from: https://gist.github.com/thatalextaylor/7408395

    :param seconds:
    :return:
    """
    ms = float(seconds * 1000)
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%dd%dh%dm%ds' % (days, hours, minutes, seconds)
    elif hours > 0:
        return '%dh%dm%ds' % (hours, minutes, seconds)
    elif minutes > 0:
        return '%dm%ds' % (minutes, seconds)
    elif seconds > 0:
        return '%ds' % seconds
    else:
        return '%.1fms' % ms
