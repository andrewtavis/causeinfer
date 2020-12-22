"""
An email marketing dataset from Kevin Hillstrom's MineThatData blog

Description found at
--------------------
  https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html

Contents
--------
  0. No Class
      download_hillstrom
      __format_data
      load_hillstrom
"""

import os
import numpy as np
import pandas as pd
from causeinfer.data.download_utils import download_file, get_download_paths


def download_hillstrom(
    data_path=None,
    url="http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
):
    """
    Downloads the dataset from Kevin Hillstrom's blog

    Parameters
    ----------
        data_path : str : optional (default=None)
            A user specified path for where the data should go

        url : str
            The url from which the data is to be downloaded

    Returns
    -------
        The data 'hillstrom.csv' in a 'datasets' folder, unless otherwise specified
    """
    directory_path, dataset_path = get_download_paths(
        data_path, file_directory="datasets", file_name="hillstrom.csv"
    )
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)
        print(
            "/{} has been created in your local directory".format(
                directory_path.split("/")[-1]
            )
        )

    if not os.path.exists(dataset_path):
        download_file(url=url, output_path=dataset_path, zip_file=False)
    else:
        print("The dataset already exists at {}".format(dataset_path))


def __format_data(df, format_covariates=True, normalize=True):
    """
    Formats the data upon loading for consistent data preparation

    Parameters
    ----------
        df : pd.DataFrame
            The original unformatted version of the data

        format_covariates : bool : optional (default=True), controlled in load_hillstrom
            True: creates dummy columns and encodes the data
            False: only steps for data readability will be taken

        normalize : bool : optional (default=True), controlled in load_hillstrom
            Normalize dataset columns to prepare them for ML methods

    Returns
    -------
        df : A formated version of the data
    """
    # Split away the history segment index within the values and other formatting
    df["history_segment"] = df["history_segment"].apply(lambda s: s.split(") ")[1])
    df["history_segment"] = df["history_segment"].astype(str)
    df["history_segment"] = [
        i.replace("$", "").replace(",", "").replace("-", "_").replace(" ", "")
        for i in df["history_segment"]
    ]

    # Column types to numeric
    df[
        [
            "recency",
            "history",
            "mens",
            "womens",
            "newbie",
            "conversion",
            "visit",
            "spend",
        ]
    ] = df[
        [
            "recency",
            "history",
            "mens",
            "womens",
            "newbie",
            "conversion",
            "visit",
            "spend",
        ]
    ].apply(
        pd.to_numeric
    )

    # Rename columns
    df = df.rename(columns={"segment": "treatment"})

    if format_covariates:

        # Create dummy columns
        dummy_cols = ["zip_code", "history_segment", "channel"]
        for col in dummy_cols:
            df = pd.get_dummies(df, columns=[col], prefix=col)

        # Encode the treatment column
        treatment_encoder = {"No E-Mail": 0, "Mens E-Mail": 1, "Womens E-Mail": 2}
        df["treatment"] = df["treatment"].apply(lambda x: treatment_encoder[x])

    if normalize:

        normalization_fields = ["recency", "history"]
        df[normalization_fields] = (
            df[normalization_fields] - df[normalization_fields].mean()
        ) / df[normalization_fields].std()

    # Format column names
    df.rename(columns=lambda x: x.lower(), inplace=True)

    # Put treatment and response at the front and end of the df respectively
    cols = list(df.columns)
    cols.insert(-1, cols.pop(cols.index("spend")))
    cols.insert(-1, cols.pop(cols.index("conversion")))
    cols.insert(-1, cols.pop(cols.index("visit")))
    cols.insert(0, cols.pop(cols.index("treatment")))
    df = df.loc[:, cols]

    return df


def load_hillstrom(
    user_file_path=None,
    format_covariates=True,
    download_if_missing=True,
    normalize=True,
):
    """
    Parameters
    ----------
        user_file_path : str : optional (default=None)
            Specify another path for the dataset
            By default the dataset should be stored in the 'datasets' folder in the cwd

        format_covariates : bool : optional (default=True)
            Indicates whether raw data should be loaded without covariate manipulation

        download_if_missing : bool : optional (default=True)
            Download the dataset if it is not downloaded before using 'download_hillstrom'

        normalize : bool : optional (default=True)
            Normalize dataset columns to prepare them for ML methods

    Returns
    -------
        data : dict object with the following attributes:

            data.description : str
                A description of the Hillstrom email marketing dataset
            data.dataset_full : numpy.ndarray : (64000, 12) or formatted (64000, 22)
                The full dataset with features, treatment, and target variables
            data.dataset_full_names : list, size 12 or formatted 22
                List of dataset variables names
            data.features : numpy.ndarray : (64000, 8) or formatted (64000, 18)
                Each row corresponding to the 8 feature values in order
            data.feature_names : list, size 8 or formatted 18
                List of feature names
            data.treatment : numpy.ndarray : (64000,)
                Each value corresponds to the treatment
            data.response_spend : numpy.ndarray : (64000,)
                Each value corresponds to how much customers spent during the two-week outcome period
            data.response_visit : numpy.ndarray : (64000,)
                Each value corresponds to whether people visited the site during the two-week outcome period
            data.response_conversion : numpy.ndarray : (64000,)
                Each value corresponds to whether they purchased at the site (i.e. converted) during the two-week outcome period
    """
    # Check that the dataset exists
    directory_path, dataset_path = get_download_paths(
        user_file_path=user_file_path,
        file_directory="datasets",
        file_name="hillstrom.csv",
    )
    # Fill above path if not
    if not os.path.exists(dataset_path):
        if download_if_missing:
            download_hillstrom(directory_path)
        else:
            raise FileNotFoundError(
                "The dataset does not exist."
                "Use the 'download_hillstrom' function to download the dataset."
            )

    # Read data
    df = pd.read_csv(dataset_path)

    # Load formated or raw data
    if format_covariates:
        if normalize:
            df = __format_data(df, format_covariates=True, normalize=True)
        else:
            df = __format_data(df, format_covariates=True, normalize=False)

    else:
        if normalize:
            df = __format_data(df, format_covariates=False, normalize=True)
        else:
            df = __format_data(df, format_covariates=False, normalize=False)

    description = (
        "The Hilstrom dataset contains 64,000 customers who purchased within twelve months."
        "The customers were involved in an e-mail marketing test."
        "1/3 were randomly chosen to receive an e-mail campaign featuring Mens merchandise."
        "1/3 were randomly chosen to receive an e-mail campaign featuring Womens merchandise."
        "1/3 were randomly chosen to not receive an e-mail campaign."
        "During a period of two weeks following the e-mail campaign, results were tracked."
        "Targeting for causal inference can be derived using visit, conversion, or total spent."
    )

    # Fields dropped to split the data for the user
    drop_fields = ["spend", "visit", "conversion", "treatment"]

    data = {
        "description": description,
        "dataset_full": df.values,
        "dataset_full_names": np.array(df.columns),
        "features": df.drop(drop_fields, axis=1).values,
        "feature_names": np.array(
            list(filter(lambda x: x not in drop_fields, df.columns))
        ),
        "treatment": df["treatment"].values,
        "response_spend": df["spend"].values,
        "response_visit": df["visit"].values,
        "response_conversion": df["conversion"].values,
    }

    return data
