def swap(arr: list, i: int, j: int) -> list:
    """Swap two array elements.

    :param arr:
    :param i:
    :param j:
    :return:
    """
    arr[i], arr[j] = arr[j], arr[i]
    return arr
