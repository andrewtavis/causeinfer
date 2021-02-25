"""
Mayo PBC Tests
--------------
"""

import os

from causeinfer.data import mayo_pbc

# *Currently disabled because of issues with the Mayo Clinic website
# def test_download_mayo_pbc():
#     mayo_pbc.download_mayo_pbc(
#         data_path=None, url="http://www.mayo.edu/research/documents/pbcdat/DOC-10026921"
#     )


def test_load_mayo_pbc():
    mayo_pbc_data = mayo_pbc.load_mayo_pbc(
        file_path="./causeinfer/data/datasets/mayo_pbc.text",
        format_covariates=True,
        normalize=True,
    )
    assert len(mayo_pbc_data) == 7

    mayo_pbc_data = mayo_pbc.load_mayo_pbc(
        file_path="./causeinfer/data/datasets/mayo_pbc.text",
        format_covariates=False,
        normalize=False,
    )
    assert len(mayo_pbc_data) == 7

    os.system("rm -rf ./datasets")
