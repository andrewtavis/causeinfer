"""
CMF Microfinance
----------------

A dataset on microfinance from The Centre for Micro Finance (CMF) at the Institute for Financial Management Research (Chennai, India).

See an example using this data at `causeinfer/examples/socioeconomic_cmf_micro <https://github.com/andrewtavis/causeinfer/blob/main/examples/socioeconomic_cmf_micro.ipynb>`_.

Description found at
    https://www.aeaweb.org/articles?id=10.1257/app.20130533 (see paper)

Contents
    download_cmf_micro (deprecated),
    _format_data,
    load_cmf_micro
"""

import os

import numpy as np
import pandas as pd
from causeinfer.data.download_utils import get_download_paths  # download_file

# The dataset can be found within CauseInfer at: https://github.com/andrewtavis/causeinfer/tree/master/causeinfer/data/datasets
# The distribution of the data is: https://www.openicpsr.org/openicpsr/project/113599/version/V1/view

# def download_cmf_micro(
#     data_path=None,
#     url='https://www.aeaweb.org/aej/app/data/0701/2013-0533_data.zip'
# ):
#     """
#     ! download_cmf_micro is deprecated as the dataset now requires an account to download
#     Downloads the dataset from the American Economic Association's website.

#     Parameters
#     ----------
#         data_path : str : optional (default=None)
#             A user specified path for where the data should go

#         url : str
#             The url from which the data is to be downloaded

#     Returns
#     -------
#         A folder with the data in a 'datasets' folder, unless otherwise specified
#     """
#     directory_path, dataset_path = get_download_paths(data_path,
#                                                       file_directory = 'datasets',
#                                                       file_name = 'cmf_micro.zip'
#                                                     )
#     if not os.path.isdir(directory_path):
#         os.makedirs(directory_path)
#         print('/{} has been created in your local directory'.format(directory_path.split('/')[-1]))

#     if not os.path.exists(dataset_path):
#         download_file(url = url, output_path = dataset_path, zip_file = True)
#     else:
#         print('The dataset already exists at {}'.format(dataset_path))


def _format_data(dataset_path, format_covariates=True, normalize=True):
    """
    Formats the data upon loading for consistent data preparation.

    Source: https://github.com/thmstang/apa19-microfinance/blob/master/helpers.r (R-version)

    Parameters
    ----------
        dataset_path : str
            The original file is a folder that has various .dta sets

        format_covariates : bool : optional (default=True)
            True: creates dummy columns and encodes the data

            False: only steps for data readability will be taken

        normalize : bool : optional (default=True)
            Normalization step controlled in load_cmf_micro

    Returns
    -------
        df : A formated version of the data
    """
    # Read in Stata .dta data
    df = pd.read_stata(
        dataset_path + "/2013-0533_data_endlines1and2.dta"
    )  # Loads Endline1 and Endline2 data, but only formats Endline1

    # Convert binary columns to numeric
    yes_no_columns = [
        col
        for col in df.columns
        if df[str(col)].isin(["Yes"]).any() or df[str(col)].isin(["No"]).any()
    ]
    df[yes_no_columns] = df[yes_no_columns].eq("Yes").mul(1)
    df["treatment"] = df["treatment"].eq("Treatment").mul(1)

    # Column types to numeric
    df = df.apply(pd.to_numeric)

    # Rename columns
    df = df.rename(columns={"areaid": "area_id"})

    if format_covariates:

        # Derive columns for an initial segment based on study baselines
        columns_to_keep = list(
            df.columns[:15]
        )  # initially select all variables before endline specific variables
        columns_to_keep.extend(
            [col for col in df.columns if col[-len("_1") :] == "_1"]
        )  # all variables relevant to Endline1
        columns_to_keep = [
            col
            for col in columns_to_keep
            if col
            not in [
                "w",
                "w1",
                "w2",
                "sample1",
                "sample2",
                "visitday_1",
                "visitmonth_1",
                "visityear_1",
            ]
        ]  # exclude survey variables
        columns_to_keep = [
            col for col in columns_to_keep if col[: len("area_") + 1] != "area_"
        ]  # exclude area-level variables
        columns_to_keep = [
            col for col in columns_to_keep if col[-len("_mo_1") :] != "_mo_1"
        ]  # exclude monthly & annual expenses variables (only keep the per capita version)
        columns_to_keep = [
            col for col in columns_to_keep if col[-len("_annual_1") :] != "_annual_1"
        ]
        df = df[df.columns.intersection(columns_to_keep)]

        # Filling NaNs in any column from a redundant column that will be dropped
        redundant_cols = [
            ("old_biz", "any_old_biz"),
            ("total_biz_1", "any_biz_1"),
            ("newbiz_1", "any_new_biz_1"),
        ]
        for tup in redundant_cols:
            mask = list(df[df[tup[1]] == 0].index)
            mask.extend(list(df[df[tup[1]] == np.nan].index))
            df.loc[mask, tup[0]] = 0

        # Removing redundant variables
        redundant_remove = [tup[1] for tup in redundant_cols]
        redundant_remove.extend(
            [
                "hhsize_1",
                "anymfi_1",
                "anymfi_amt_1",
                "anyloan_1",
                "anyloan_amt_1",
                "hours_week_1",
                "hours_headspouse_week_1",
                "hours_child1620_week_1",
                "total_exp_mo_pc_1",
            ]
        )
        for col in redundant_remove:
            del df[col]

        # Cleaning NaNs - replace, with all remaining NaN rows dropped
        # First remove columns with more than 10% NaN values
        nan_threshold = 0.1
        df = df[df.columns[df.isnull().mean() < nan_threshold]]

        # Replace business variables with 0 if total_biz_1 is 0 or NaN
        total_biz_mask = list(df[df["total_biz_1"] == 0].index)
        total_biz_mask.extend(list(df[df["total_biz_1"] == np.nan].index))

        for column in [col for col in df.columns if col[: len("biz")] == "biz"]:
            df.loc[total_biz_mask, column] = 0

        # Fill all non-index variables with their mean
        for column in [col for col in df.columns if "index" not in col]:
            df[column].fillna(df[column].mean())

        df = df.dropna()

        # Exclude those columns with outliers in the expense related variables
        # Interquartile ranges are used, with 5*iqr times the quartiles being dropped
        exp_col = [col for col in df.columns if "exp_mo_pc" in col]
        exp_col.extend(["informal_amt_1"])

        for col in exp_col:
            q75, q25 = np.percentile(df[col], [75, 25])
            iqr = q75 - q25  # pylint: disable=unused-variable
            # Filtering for values between q25-5*iqr and q75+5*iqr
            df = df.query("(@q25 - 5 * @iqr) <= {} <= (@q75 + 5 * @iqr)".format(col))

        # Convert the unit of expense-related & loan-related variables from Rupees to USD
        conv = 9.1768
        for col in [
            "spandana_amt_1",
            "othermfi_amt_1",
            "bank_amt_1",
            "informal_amt_1",
            "durables_exp_mo_pc_1",
            "nondurable_exp_mo_pc_1",
            "food_exp_mo_pc_1",
            "health_exp_mo_pc_1",
            "temptation_exp_mo_pc_1",
            "festival_exp_mo_pc_1",
        ]:
            df.loc[:, col] = df.loc[:, col].div(conv)

        # Create dummy columns
        dummy_cols = ["area_id"]
        for col in dummy_cols:
            df = pd.get_dummies(df, columns=[col], prefix=col)

    # Normalize data for the user (exclude binaries, treatment, and responses)
    if normalize:
        non_normalization_fields = [
            "treatment",
            "biz_index_all_1",
            "women_emp_index_1",
            "male_head_1",
            "head_noeduc_1",
            "anychild1318_1",
            "spouse_literate_1",
            "spouse_works_wage_1",
            "ownland_hyderabad_1",
            "ownland_village_1",
            "spandana_1",
            "othermfi_1",
            "anybank_1",
            "anyinformal_1",
            "everlate_1",
        ]
        non_normalization_fields += [
            col for col in df.columns if col[: len("area_id_")] == "area_id_"
        ]
        df[df.columns.difference(non_normalization_fields)] = (
            df[df.columns.difference(non_normalization_fields)]
            - df[df.columns.difference(non_normalization_fields)].mean()
        ) / df[df.columns.difference(non_normalization_fields)].std()

    # Drop household id
    df.drop("hhid", axis=1, inplace=True)

    # Put treatment and response at the front and end of the df respectively
    cols = list(df.columns)
    cols.insert(-1, cols.pop(cols.index("women_emp_index_1")))
    cols.insert(-1, cols.pop(cols.index("biz_index_all_1")))
    cols.insert(0, cols.pop(cols.index("treatment")))
    df = df.loc[:, cols]

    return df


def load_cmf_micro(
    file_path=None,
    format_covariates=True,
    # download_if_missing=True, Deprecated: data requires an account to download now
    normalize=True,
):
    """
    Loads the CMF micro dataset with formatting if desired.

    Parameters
    ----------
        file_path : str : optional (default=None)
            Specify another path for the dataset

            By default the dataset should be stored in the 'datasets' folder in the cwd

        load_raw_data : bool : optional (default=True)
            Indicates whether raw data should be loaded without covariate manipulation

        download_if_missing : bool : optional (default=True) (Deprecated)
            Download the dataset if it is not downloaded before using 'download_cmf_micro'

        normalize : bool : optional (default=True)
            Normalize the dataset to prepare it for ML methods

    Returns
    -------
        data : dict object with the following attributes:

            data.description : str
                A description of the CMF microfinance data

            data.dataset_full : numpy.ndarray : (5328, 183) or formatted (5328, 60)
                The full dataset with features, treatment, and target variables

            data.dataset_full_names : list, size 61
                List of dataset variables names

            data.features : numpy.ndarray : (5328, 186) or formatted (5328, 57)
                Each row corresponding to the 58 feature values in order (note that other target can be a feature)

            data.feature_names : list, size 58
                List of feature names

            data.treatment : numpy.ndarray : (5328,)
                Each value corresponds to the treatment (1 = treat, 0 = control)

            data.response_biz_index : numpy.ndarray : (5328,)
                Each value corresponds to the business index of each of the participants

            data.response_women_emp : numpy.ndarray : (5328,)
                Each value corresponds to the women's empowerment index of each of the participants
    """
    # Check that the dataset exists
    (
        directory_path,  # pylint: disable=unused-variable
        dataset_path,
    ) = get_download_paths(
        file_path=file_path, file_directory="datasets", file_name="cmf_micro",
    )
    # Fill above path if not
    if not os.path.exists(dataset_path):
        # if download_if_missing:
        #     download_cmf_micro(directory_path)
        # else:
        print(
            "The dataset does not exist. "
            "The dataset can be found within CauseInfer at: https://github.com/andrewtavis/causeinfer/tree/master/causeinfer/data/datasets. "
            "The distribution of the data is: https://www.openicpsr.org/openicpsr/project/113599/version/V1/view"
        )
        return

    # Load formated or raw data
    if format_covariates:
        if normalize:
            df = _format_data(dataset_path, format_covariates=True, normalize=True)
        else:
            df = _format_data(dataset_path, format_covariates=True, normalize=False)

    else:
        if normalize:
            df = _format_data(dataset_path, format_covariates=False, normalize=True)
        else:
            df = _format_data(dataset_path, format_covariates=False, normalize=False)

    description = (
        "The data comes from The Centre for Micro Finance (CMF) at the Institute for Financial Management Research (Chennai, India)"
        "The feature set can be used to derive the effects of microfinance on various post-treatment indexes."
        "Specifically we will focus on the post-treatment business and women's empowerment indexes."
        "The other target value can be added into the dataset as a feature."
    )

    # Fields dropped to split the data for the user
    drop_fields = ["biz_index_all_1", "women_emp_index_1", "treatment"]

    return {
        "description": description,
        "dataset_full": df.values,
        "dataset_full_names": np.array(df.columns),
        "features": df.drop(drop_fields, axis=1).values,
        "feature_names": np.array(
            list(filter(lambda x: x not in drop_fields, df.columns))
        ),
        "treatment": df["treatment"].values,
        # The target that isn't of interest can also be used as a feature, but should be normalized
        "response_biz_index": df["biz_index_all_1"].values,
        "response_women_emp": df["women_emp_index_1"].values,
    }
