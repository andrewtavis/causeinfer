# =============================================================================
# A dataset on medical trials to combat primary biliary cirrhosis of the liver from the Mayo Clinic
#
# Description found at
# --------------------
#   https://www.mayo.edu/research/documents/pbchtml/DOC-10027635
# 
# Contents
# --------
#   0. No Class
#       download_mayo_pbc
#       __format_data
#       load_mayo_pbc
# =============================================================================

import os
import numpy as np
import pandas as pd
from causeinfer.datasets.download_utilities import download_file, get_download_paths

def download_mayo_pbc(
    data_path=None,
    url='http://www.mayo.edu/research/documents/pbcdat/DOC-10026921'
):
    """
    Downloads the dataset from the Mayo Clinic's research documents.

    Result
    ------
        The text file 'mayo_pbc' in a 'datasets' folder, unless otherwise specified
    """
    data_path, dataset_path = get_download_paths(data_path, 'datasets', 'mayo_pbc.text')
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    if not os.path.exists(dataset_path):
        download_file(url, dataset_path, zip_file = False)


def __format_data(dataset_path):
    """
    Formats the data upon loading for consistent data preparation.

    The original file is a text file with inconsistent spacing, and periods for NaNs. 
    Furthermore, process only loads those units that took part in the randomized trial, 
    as there are 106 cases that were monitored, but not in the trial.
    """
    # Read in the text file
    with open(dataset_path, 'r') as f:
        data = f.read().splitlines()
    
    # The following converts the text file into a list of lists
    # The three iterations account for initial spaces for single, double, and tripple digit numbers
    data_list_of_lists = [data[i][2:].replace('    ', ',').replace('   ', ',').replace('  ', ',').replace(' ', ',').split(',')[1:] for i in range(10)]
    data_list_of_lists.extend([data[i][1:].replace('    ', ',').replace('   ', ',').replace('  ', ',').replace(' ', ',').split(',')[1:] for i in list(range(99))[10:]])
    data_list_of_lists.extend([data[i].replace('    ', ',').replace('   ', ',').replace('  ', ',').replace(' ', ',').split(',')[1:] for i in list(range(312))[99:]])

    df = pd.DataFrame(data_list_of_lists)

    col_names = ['days_since_register', 'status', 'treatment', 'age',
                 'sex', 'ascites', 'hepatomegaly', 'spiders', 'edema',
                 'bilirubin', 'cholesterol', 'albumin', 'copper', 
                 'alkaline', 'SGOT', 'triglicerides', 'platelets', 
                 'prothrombin', 'histologic stage']
    df.columns = col_names

    # Filling NaNs with column averages (they occur in cholesterol, copper, triglicerides and platelets)
    df = df.replace('.', np.nan)
    df = df.astype(float)
    df.fillna(df.mean(), inplace=True)

    # Replace control from 2 to 0
    df.loc[df['treatment'] == 2, 'treatment'] = 0

    return df


def load_mayo_pbc(
    data_path=None,
    # load_raw_data=False,
    download_if_missing=True,
    normalize=True
):
    """
    Parameters
    ----------
        load_raw_data : not included, as original data isn't in table form

        data_path : str, optional (default=None)
            Specify another download and cache folder for the dataset.
            By default the dataset will be stored in the 'datasets' folder in the cwd

        download_if_missing : bool, optional (default=True)
            Download the dataset if it is not downloaded before using 'download_mayo_pbc'

        normalize : bool, optional (default=True)
            Normalize the dataset to prepare it for ML methods

    Returns
    -------
        dataset : dict object with the following attributes:

            dataset.description : str
                A description of the Mayo Clinic PBC dataset.
            dataset.dataset_full : ndarray, shape (312, 19)
                The full dataset with features, treatment, and target variables
            dataset.data : ndarray, shape (312, 17)
                Each row corresponding to the 17 feature values in order.
            dataset.feature_names : list, size 17
                List of feature names.
            dataset.treatment : ndarray, shape (312,)
                Each value corresponds to the treatment (1 = treat, 0 = control).
            dataset.target : numpy array of shape (312,)
                Each value corresponds to one of the outcomes (0 = alive, 1 = liver transplant, 2 = dead).
    """
    # Check that the dataset exists
    data_path, dataset_path = get_download_paths(data_path, 'datasets', 'mayo_pbc.text')
    if not os.path.exists(dataset_path):
        if download_if_missing:
            download_mayo_pbc(data_path)
        else:
            raise FileNotFoundError(
                "The dataset does not exist."
                "Use the 'download_mayo_pbc' function to download the dataset."
            )

    # Load formated data
    df = __format_data(dataset_path)

    description = 'The data is from the Mayo Clinic trial in primary biliary cirrhosis (PBC) of the liver conducted between 1974 and 1984.' \
                  'A total of 424 PBC patients, referred to Mayo Clinic during that ten-year interval, met eligibility criteria for the randomized placebo controlled trial of the drug D-penicillamine.' \
                  'The first 312 cases in the data set participated in the randomized trial and contain largely complete data.'\
                  'For modeling purposes, alive (target=0) will be modelled against a resulting transplant (1) and death (2).'
    
    # Fields dropped to split the data for the user
    drop_fields = ['status', 'treatment']
    covariate_fields = [col for col in df.columns if col not in drop_fields]

    # Normalize data for the user
    if normalize:
        df[covariate_fields] = (df[covariate_fields] - df[covariate_fields].mean()) / df[covariate_fields].std()
    
    data = {
        'description': description,
        'dataset_full' : df.values,
        'data': df.drop(drop_fields, axis=1).values,
        'feature_names': np.array(list(filter(lambda x: x not in drop_fields, df.columns))),
        'treatment': df['treatment'].values,
        'target': df['status'].values
    }

    return data