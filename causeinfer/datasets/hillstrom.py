# =============================================================================
# An email marketing dataset from Kevin Hillstrom's MineThatData blog
# 
# Description found at
# --------------------
#   https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html
#
# Contents
# --------
#   0. No Class
#       download_hillstrom
#       __format_data
#       load_hillstrom
# =============================================================================

import os
import numpy as np
import pandas as pd
from causeinfer.datasets.download_utilities import download_file, get_download_paths

def download_hillstrom(
    data_path=None,
    url='http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv'
):
    """
    Downloads the dataset from Kevin Hillstrom's blog

    Result
    ------
        The data 'hillstrom.csv' in a 'datasets' folder, unless otherwise specified
    """
    data_path, dataset_path = get_download_paths(data_path, 'datasets', 'hillstrom.csv')
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    if not os.path.exists(dataset_path):
        download_file(url, dataset_path, zip_file = False)


def __format_data(df):
    """
    Formats the data upon loading for consistent data preparation
    """
    # Split away the history segment index
    df['history_segment'] = df['history_segment'].apply(lambda s: s.split(') ')[1])
    
    # Create dummy columns for zip_code, history_segment, and channel
    dummy_cols = ['zip_code', 'history_segment', 'channel']
    for col in dummy_cols:
        df = pd.get_dummies(df, columns=[col], prefix=col)

    # Encode the segment column
    segment_encoder = {'No E-Mail': 0, 'Mens E-Mail': 1, 'Womens E-Mail': 2}
    df['segment'] = df['segment'].apply(lambda x: segment_encoder[x])
    
    return df


def load_hillstrom(
    data_path=None,
    load_raw_data=False,
    download_if_missing=True,
    normalize=True
):
    """
    Parameters
    ----------
        load_raw_data : bool, default: False
            Indicates whether the raw data should be loaded without '__format_data'

        data_path : str, optional (default=None)
            Specify another download and cache folder for the dataset.
            By default the dataset will be stored in the 'datasets' folder in the cwd

        download_if_missing : bool, optional (default=True)
            Download the dataset if it is not downloaded before using 'download_hillstrom'

        normalize : bool, optional (default=True)
            Normalize the dataset to prepare it for ML methods

    Returns
    -------
        dataset : dict object with the following attributes:

            dataset.description : str
                A description of the Hillstrom email marketing dataset.
            dataset.dataset_full : ndarray, shape (64000, 12)
                The full dataset with features, treatment, and target variables
            dataset.data : ndarray, shape (64000, 8)
                Each row corresponding to the 8 feature values in order.
            dataset.feature_names : list, size 8
                List of feature names.
            dataset.treatment : ndarray, shape (64000,)
                Each value corresponds to the treatment.
            dataset.target : numpy array of shape (64000,)
                Each value corresponds to one of the outcomes. By default, it's `spend` outcome (look at `target_spend` below).
            dataset.target_spend : numpy array of shape (64000,)
                Each value corresponds to how much customers spent during the two-week outcome period.
            dataset.target_visit : numpy array of shape (64000,)
                Each value corresponds to whether people visited the site during the two-week outcome period.
            dataset.target_conversion : numpy array of shape (64000,)
                Each value corresponds to whether they purchased at the site (i.e. converted) during the two-week outcome period.
    """
    # Check that the dataset exists
    data_path, dataset_path = get_download_paths(data_path, 'datasets', 'hillstrom.csv')
    if not os.path.exists(dataset_path):
        if download_if_missing:
            download_hillstrom(data_path)
        else:
            raise FileNotFoundError(
                "The dataset does not exist."
                "Use the 'download_hillstrom' function to download the dataset."
            )

    # Load formated or raw data
    df = pd.read_csv(dataset_path)
    if not load_raw_data:
        df = __format_data(df)

    description = 'The Hilstrom dataset contains 64,000 customers who purchased within twelve months.' \
                  'The customers were involved in an e-mail marketing test.' \
                  '1/3 were randomly chosen to receive an e-mail campaign featuring Mens merchandise.' \
                  '1/3 were randomly chosen to receive an e-mail campaign featuring Womens merchandise.' \
                  '1/3 were randomly chosen to not receive an e-mail campaign.' \
                  'During a period of two weeks following the e-mail campaign, results were tracked.' \
                  'Targeting for causal inference can be derived using visit, conversion, or total spent.'

    # Fields dropped to split the data for the user
    drop_fields = ['spend', 'visit', 'conversion', 'segment']
    covariate_fields = [col for col in df.columns if col not in drop_fields]

    # Normalize data for the user
    if normalize:
        df[covariate_fields] = (df[covariate_fields] - df[covariate_fields].mean()) / df[covariate_fields].std()
    
    data = {
        'description': description,
        'dataset_full' : df.values,
        'data': df.drop(drop_fields, axis=1).values,
        'feature_names': np.array(list(filter(lambda x: x not in drop_fields, df.columns))),
        'treatment': df['segment'].values,
        'target': df['spend'].values,
        'target_spend': df['spend'].values,
        'target_visit': df['visit'].values,
        'target_conversion': df['conversion'].values,
    }

    return data