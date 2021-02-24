"""
Evaluation Tests
----------------
"""

import numpy as np
from causeinfer import evaluation


def test_signal_to_noise(y_split, w_split):
    sn_ration = evaluation.signal_to_noise(y=y_split, w=w_split)
    assert type(sn_ration) == float or type(sn_ration) == np.float64
