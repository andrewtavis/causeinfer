"""
Hillstrom Tests
---------------
"""

import os

from causeinfer.data import hillstrom


def test_download_hillstrom():
    hillstrom.download_hillstrom(
        data_path=None,
        url="http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
    )


def test_load_hillstrom():
    hillstrom_data = hillstrom.load_hillstrom(
        file_path="./datasets/hillstrom.csv", format_covariates=True, normalize=True
    )
    assert len(hillstrom_data) == 9

    hillstrom_data = hillstrom.load_hillstrom(
        file_path="./datasets/hillstrom.csv", format_covariates=False, normalize=False
    )
    assert len(hillstrom_data) == 9

    os.system("rm -rf ./datasets")
