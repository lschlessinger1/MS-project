def swap(arr, idx_1, idx_2):
    """Swap two array elements
    """
    arr_copy = arr.copy()
    first = arr_copy[idx_1]
    second = arr_copy[idx_2]
    arr_copy[idx_1] = second
    arr_copy[idx_2] = first

    return arr_copy
