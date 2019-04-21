from typing import List

import numpy as np

from src.autoks.distance.distance import DistanceBuilder


def has_no_nans(a):
    return (~np.isnan(a)).all()


def test_kernel(distance_builder: DistanceBuilder,
                n_active_models: int,
                selected_ind: List[int],
                all_candidate_indices: List[int]):
    """For debugging"""
    # get kernel kernel
    K = distance_builder.get_kernel(n_active_models)

    KxX = K[selected_ind][:, all_candidate_indices]
    KXX = K[selected_ind][:, selected_ind]

    assert KxX.shape == (len(selected_ind), len(all_candidate_indices))
    assert KXX.shape == (len(selected_ind), len(selected_ind))
    assert has_no_nans(KxX)
    assert has_no_nans(KXX)
