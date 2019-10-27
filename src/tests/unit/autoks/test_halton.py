import numpy as np

from src.autoks.distance.sampling.halton import generate_halton, generate_generalized_halton, halton_sample


def test_generate_halton_one_d():
    n = 9
    d = 1
    samples = generate_halton(n, d)
    assert isinstance(samples, list)
    assert len(samples) == n
    assert isinstance(samples[0], list)
    assert len(samples[0]) == d
    expected = [[1 / 2], [1 / 4], [3 / 4], [1 / 8], [5 / 8], [3 / 8], [7 / 8], [1 / 16], [9 / 16]]
    assert samples == expected


def test_generate_halton_two_d():
    n = 9
    d = 2
    samples = generate_halton(n, d)
    assert isinstance(samples, list)
    assert len(samples) == n
    assert isinstance(samples[0], list)
    assert len(samples[0]) == d
    expected = [[1 / 2, 1 / 3], [1 / 4, 2 / 3], [3 / 4, 1 / 9], [1 / 8, 4 / 9],
                [5 / 8, 7 / 9], [3 / 8, 2 / 9], [7 / 8, 5 / 9], [1 / 16, 8 / 9], [9 / 16, 1 / 27]]
    assert samples == expected


def test_generate_generalized_halton():
    n = 100
    d = 5
    samples = generate_generalized_halton(n, d)
    assert isinstance(samples, list)
    assert len(samples) == n
    assert isinstance(samples[0], list)
    assert len(samples[0]) == d
    expected = [1 / 2, 2 / 3, 4 / 5, 6 / 7, 72 / 99]
    assert samples[0] == expected


def test_halton_sample_0_leap_no_scramble():
    n = 9
    d = 1
    samples = halton_sample(n, d, scramble=False)
    assert isinstance(samples, list)
    assert len(samples) == n
    assert isinstance(samples[0], list)
    assert len(samples[0]) == d
    expected = [[1 / 2], [1 / 4], [3 / 4], [1 / 8], [5 / 8], [3 / 8], [7 / 8], [1 / 16], [9 / 16]]
    assert samples == expected

def test_halton_sample_0_leap_scramble():
    np.random.seed(42)
    n = 9
    d = 1
    samples = halton_sample(n, d, scramble=True)
    assert isinstance(samples, list)
    assert len(samples) == n
    assert isinstance(samples[0], list)
    assert len(samples[0]) == d
    expected = [[1 / 2], [1 / 4], [3 / 4], [1 / 8], [5 / 8], [3 / 8], [7 / 8], [1 / 16], [9 / 16]]
    assert samples == expected
    np.random.seed()
