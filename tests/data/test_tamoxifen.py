# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for Tamoxifen data.
"""

import os

from causeinfer.data import tamoxifen

# ! download_tamoxifen is deprecated as the dataset is bundled in the repository
# def test_download_tamoxifen():
#     tamoxifen.download_tamoxifen(data_path=None, url="")


def test_load_tamoxifen():
    tamoxifen_data = tamoxifen.load_tamoxifen(
        file_path="./src/causeinfer/data/datasets/tamoxifen.txt",
        format_covariates=True,
        normalize=True,
    )
    assert len(tamoxifen_data) == 7

    tamoxifen_data = tamoxifen.load_tamoxifen(
        file_path="./src/causeinfer/data/datasets/tamoxifen.txt",
        format_covariates=False,
        normalize=False,
    )
    assert len(tamoxifen_data) == 7

    os.system("rm -rf ./datasets")
