# =============================================================================
# A dataset on microfinance from The Centre for Micro Finance (CMF) at the Institute for Financial Management Research (Chennai, India)
# 
# Description found at
# --------------------
#   https://www.aeaweb.org/articles?id=10.1257/app.20130533 (see paper)
# 
# Contents
# --------
#   0. No Class
#       download_cmf_microfinance
#       __format_data
#       load_cmf_microfinance
# =============================================================================

import os
import numpy as np
import pandas as pd
from causeinfer.data.download_utilities import download_file, get_download_paths

def download_cmf_microfinance(
    data_path=None,
    url='https://www.aeaweb.org/aej/app/data/0701/2013-0533_data.zip'
):
    """
    Downloads the dataset from the American Economic Assosciation's website

    Parameters
    ----------
        data_path : str, optional (default=None)
            A user specified path for where the data should go

        url : str
            The url from which the data is to be downloaded

    Result
    ------
        A folder with the data in a 'datasets' folder, unless otherwise specified
    """
    directory_path, dataset_path = get_download_paths(data_path, 
                                                      file_directory = 'datasets', 
                                                      file_name = 'cmf_microfinance.zip'
                                                    )
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)
        print('/{} has been created in your local directory'.format(directory_path.split('/')[-1]))

    if not os.path.exists(dataset_path):
        download_file(url = url, output_path = dataset_path, zip_file = True)
    else:
        print('The dataset already exists at {}'.format(dataset_path))


def __format_data(
    dataset_path, 
    load_raw_data=False,
    normalize=True
    ):
    """
    Formats the data upon loading for consistent data preparation
    Source: https://github.com/thmstang/apa19-microfinance/blob/master/helpers.r (R-version)
    
    Parameters
    ----------
        dataset_path : str
            The original file is a folder that has various .dta sets

        normalize : bool, optional
            Normalization step controlled in load_cmf_microfinance

    Returns
    -------
        A formated version of the data
    """
    # Read in Stata .dta data
    df = pd.read_stata(dataset_path+'/2013-0533_data (TO SUBMIT)/2013-0533_data_endlines1and2.dta')

    if not load_raw_data: # Formats the data if False
    
        # Derive columns for an initial segment based on study baselines
        columns_to_keep = list(df.columns[:15]) # initially select all variables before endline specific variables
        columns_to_keep.extend([col for col in df.columns if col[-len('_1'):] == '_1']) # all variables relevant to Endline1
        columns_to_keep = [col for col in columns_to_keep if col not in ['w', 'w1', 'w2', 'sample1', 'sample2', 'visitday_1', 'visitmonth_1', 'visityear_1']] # exclude survey variables
        columns_to_keep = [col for col in columns_to_keep if col[:len('area_')+1] != 'area_'] # exclude area-level variables
        columns_to_keep = [col for col in columns_to_keep if col[-len('_mo_1'):] != '_mo_1'] # exclude monthly & annual expenses variables (only keep the per capita version)
        columns_to_keep = [col for col in columns_to_keep if col[-len('_annual_1'):] != '_annual_1']

        yes_no_columns = [col for col in columns_to_keep if df[str(col)].isin(['Yes']).any() or df[str(col)].isin(['No']).any()]
        
        df = df[df.columns.intersection(columns_to_keep)]
        df[yes_no_columns] = df[yes_no_columns].eq('Yes').mul(1)
        df['treatment'] = df['treatment'].eq('Treatment').mul(1)

        redundant_cols = [('old_biz', 'any_old_biz'), ('total_biz_1', 'any_biz_1'), ('newbiz_1', 'any_new_biz_1')]

        for tup in redundant_cols:
            mask = list(df[df[tup[1]] == 0].index)
            mask.extend(list(df[df[tup[1]] == np.nan].index))
            df.loc[mask, tup[0]] = 0

        # Removing redundant variables
        redundant_remove = [tup[1] for tup in redundant_cols]
        redundant_remove.extend(['hhsize_1', 'anymfi_1', 'anymfi_amt_1',
                                'anyloan_1', 'anyloan_amt_1', 'hours_week_1', 
                                'hours_headspouse_week_1', 'hours_child1620_week_1', 'total_exp_mo_pc_1'])
        for col in redundant_remove:
            del df[col]

        # Cleaning NaNs - replace, with all remaining NaN rows dropped
        # First remove columns with more than 10% NaN values
        nan_threshold = 0.1
        df = df[df.columns[df.isnull().mean() < nan_threshold]]
        
        # Replce business variables with 0 if total_biz_1 is 0 or NaN
        total_biz_mask = list(df[df['total_biz_1'] == 0].index)
        total_biz_mask.extend(list(df[df['total_biz_1'] == np.nan].index))

        for column in [col for col in df.columns if col[:len('biz')] == 'biz']:
            df.loc[total_biz_mask, column] = 0

        # Fill all non-index variables with their mean
        for column in [col for col in df.columns if 'index' not in col]:
            df[column].fillna(df[column].mean())

        df = df.dropna()

        # Exclude those columns with outliers in the expense related variables
        # Interquartile ranges are used, with 5*iqr times the quartiles being dropped

        exp_col = [col for col in df.columns if 'exp_mo_pc' in col]
        exp_col.extend(['informal_amt_1'])  

        for col in exp_col:
            q75, q25 = np.percentile(df[col], [75 ,25])
            iqr = q75 - q25
            iqr # Here so VS Code will leave me alone about an unused variable (see @iqr below)
            # Filtering for values between q25-5*iqr and q75+5*iqr
            df = df.query('(@q25 - 5 * @iqr) <= {} <= (@q75 + 5 * @iqr)'.format(col))

        # Convert the unit of expense-related & loan-related variables from Rupees to USD
        conv = 9.1768
        for col in ['spandana_amt_1', 'othermfi_amt_1', 'bank_amt_1',
                    'informal_amt_1', 'durables_exp_mo_pc_1', 
                    'nondurable_exp_mo_pc_1', 'food_exp_mo_pc_1',
                    'health_exp_mo_pc_1', 'temptation_exp_mo_pc_1',
                    'festival_exp_mo_pc_1']:
            df[col] = df[col].div(conv)

        # Normalize data for the user
        if normalize:
            normalization_fields = []
            df[normalization_fields] = (df[normalization_fields] - df[normalization_fields].mean()) / df[normalization_fields].std()

    return df


def load_cmf_microfinance(
    data_path=None,
    load_raw_data=False,
    download_if_missing=True,
    normalize=True
):
    """
    Parameters
    ----------
        data_path : str, optional (default=None)
            Specify another download and cache folder for the dataset
            By default the dataset should be stored in the 'datasets' folder in the cwd
        
        load_raw_data : bool, default: False
            loads an unformated version of the data (detault=False)

        download_if_missing : bool, optional (default=True)
            Download the dataset if it is not downloaded before using 'download_cmf_microfinance'

        normalize : bool, optional (default=True)
            Normalize the dataset to prepare it for ML methods

    Returns
    -------
        data : dict object with the following attributes:

            data.description : str
                A description of the CMF microfinance data
            data.dataset_full : ndarray, shape (5328, 61) or formatted (XYZ, XYZ)
                The full dataset with features, treatment, and target variables
            data.dataset_full_names : list, size 61
                List of dataset variables names
            data.features : ndarray, shape (5328, 58) or formatted (XYZ, XYZ)
                Each row corresponding to the 58 feature values in order (note that other target can be a feature)
            data.feature_names : list, size 58
                List of feature names
            data.treatment : ndarray, shape (5328,)
                Each value corresponds to the treatment (1 = treat, 0 = control)
            data.target_biz_index : numpy array of shape (5328,)
                Each value corresponds to the business index of each of the participants
            data.target_women_emp : numpy array of shape (5328,)
                Each value corresponds to the women's empowerment index of each of the participants
    """
    # Check that the dataset exists
    data_path, dataset_path = get_download_paths(data_path, 'datasets', 'cmf_microfinance')
    if not os.path.exists(dataset_path):
        if download_if_missing:
            download_cmf_microfinance(data_path)
        else:
            raise FileNotFoundError(
                "The dataset does not exist."
                "Use the 'download_cmf_microfinance' function to download the dataset."
            )

    # Load formated or raw data
    if not load_raw_data:
        df = __format_data(dataset_path)
    else:
        df = __format_data(dataset_path, load_raw_data=True)

    description = "The data comes from The Centre for Micro Finance (CMF) at the Institute for Financial Management Research (Chennai, India)"\
                  "The feature set can be used to derive the effects of microfinance on various post-treatment indexes."\
                  "Specifically we will focus on the post-treatment business and women's empowerment indexes."\
                  "The other target value can be added into the dataset as a feature."
    
    # Fields dropped to split the data for the user
    drop_fields = ['biz_index_all_1', 'women_emp_index_1', 'treatment']
    
    data = {
        'description': description,
        'dataset_full' : df.values,
        'dataset_full_names': np.array(df.columns),
        'features': df.drop(drop_fields, axis=1).values,
        'feature_names': np.array(list(filter(lambda x: x not in drop_fields, df.columns))),
        'treatment': df['treatment'].values,
        # The target that isn't of interest can also be used as a feature
        'target_biz_index': df['biz_index_all_1'].values,
        'target_women_emp': df['women_emp_index_1'].values
    }

    return data