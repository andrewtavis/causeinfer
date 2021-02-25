"""
CMF Microfinance Tests
----------------------
"""

import os

from causeinfer.data import cmf_micro

# *Currently disabled because the dataset is behind a paywall now
# def test_download_cmf_micro():
#     cmf_micro.download_cmf_micro(
#         data_path=None, url=""
#     )


def test_load_cmf_micro():
    data_cmf_micro = cmf_micro.load_cmf_micro(
        file_path="./causeinfer/data/datasets/cmf_micro",
        format_covariates=True,
        normalize=True,
    )
    assert len(data_cmf_micro) == 8

    data_cmf_micro = cmf_micro.load_cmf_micro(
        file_path="./causeinfer/data/datasets/cmf_micro",
        format_covariates=False,
        normalize=False,
    )
    assert len(data_cmf_micro) == 8
