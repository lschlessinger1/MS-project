def argsort(unsorted_list):
    return [i[0] for i in sorted(enumerate(unsorted_list), key=lambda x: x[1])]
