def standardize(x, x_mean, x_std):
    """Make the values of each dimension in the data to have zero-mean and unit-variance.

    See https://en.wikipedia.org/wiki/Feature_scaling#Standardization for more.
    """
    return (x - x_mean) / x_std


def inverse_standardize(x, x_mean, x_std):
    """Inverse standardization"""
    return (x * x_std) + x_mean
