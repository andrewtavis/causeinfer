"""
Mayo Clinic PBC
---------------

A dataset on medical trials to combat primary biliary cholangitis (PBC, formerly cirrhosis) of the liver from the Mayo Clinic.

See an example using this data at `causeinfer/examples/medical_mayo_pbc <https://github.com/andrewtavis/causeinfer/blob/main/examples/medical_mayo_pbc.ipynb.>`_.

Description found at
    https://www.mayo.edu/research/documents/pbchtml/DOC-10027635

Contents
    download_mayo_pbc,
    _format_data,
    load_mayo_pbc
"""

import os

import numpy as np
import pandas as pd
from causeinfer.data.download_utils import download_file, get_download_paths


def download_mayo_pbc(
    data_path=None, url="http://www.mayo.edu/research/documents/pbcdat/DOC-10026921"
):
    """
    Downloads the dataset from the Mayo Clinic's research documents.

    Parameters
    ----------
        data_path : str : optional (default=None)
            A user specified path for where the data should go

        url : str
            The url from which the data is to be downloaded

    Returns
    -------
        The text file 'mayo_pbc' in a 'datasets' folder, unless otherwise specified
    """
    directory_path, dataset_path = get_download_paths(
        data_path, file_directory="datasets", file_name="mayo_pbc.text"
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


def _format_data(dataset_path, format_covariates=True, normalize=True):
    """
    Formats the data upon loading for consistent data preparation.

    Parameters
    ----------
        dataset_path : str
            The original file is a text file with inconsistent spacing, and periods for NaNs.

            Furthermore, process only loads those units that took part in the randomized trial,
            as there are 106 cases that were monitored, but not in the trial.

        format_covariates : bool : optional (default=True)
            True: creates dummy columns and encodes the data

            False: only steps for data readability will be taken

        normalize : bool : optional (default=True)
            Normalization step controlled in load_mayo_pbc

    Returns
    -------
        df : A formated version of the data
    """
    # Read in the text file
    with open(dataset_path, "r") as file:
        data = file.read().splitlines()

    # The following converts the text file into a list of lists
    # The three iterations account for initial spaces for single, double, and tripple digit numbers
    data_list_of_lists = [
        data[i][2:]
        .replace("    ", ",")
        .replace("   ", ",")
        .replace("  ", ",")
        .replace(" ", ",")
        .split(",")[1:]
        for i in range(10)
    ]
    data_list_of_lists.extend(
        [
            data[i][1:]
            .replace("    ", ",")
            .replace("   ", ",")
            .replace("  ", ",")
            .replace(" ", ",")
            .split(",")[1:]
            for i in list(range(99))[10:]
        ]
    )
    data_list_of_lists.extend(
        [
            data[i]
            .replace("    ", ",")
            .replace("   ", ",")
            .replace("  ", ",")
            .replace(" ", ",")
            .split(",")[1:]
            for i in list(range(312))[99:]
        ]
    )

    df = pd.DataFrame(data_list_of_lists)

    col_names = [
        "days_since_register",
        "status",
        "treatment",
        "age",
        "sex",
        "ascites",
        "hepatomegaly",
        "spiders",
        "edema",
        "bilirubin",
        "cholesterol",
        "albumin",
        "copper",
        "alkaline",
        "sgot",
        "triglicerides",
        "platelets",
        "prothrombin",
        "histologic_stage",
    ]
    df.columns = col_names

    # Filling NaNs with column averages (they occur in cholesterol, copper, triglicerides and platelets)
    df = df.replace(".", np.nan)
    df = df.astype(float)
    df.fillna(df.mean(), inplace=True)

    # Column types to numeric
    df = df.apply(pd.to_numeric)

    if format_covariates:

        # Create dummy columns for edema and histologic_stage
        dummy_cols = ["edema", "histologic_stage"]
        for col in dummy_cols:
            df = pd.get_dummies(df, columns=[col], prefix=col)

        # Cleaning edema and histologic_stage column names
        df = df.rename(
            columns={
                "edema_0.0": "no_edema_no_diuretics",
                "edema_0.5": "yes_edema_no_diuretics",
                "edema_1.0": "yes_edema_yes_diuretics",
            }
        )

        df.rename(
            columns=lambda x: x.split(".")[0]
            if x[: len("histologic_stage")] == "histologic_stage"
            else x,
            inplace=True,
        )

    # Replace control from 2 to 0
    df.loc[df["treatment"] == 2, "treatment"] = 0

    if normalize:

        normalization_fields = [
            "days_since_register",
            "age",
            "bilirubin",
            "cholesterol",
            "albumin",
            "copper",
            "alkaline",
            "sgot",
            "triglicerides",
            "platelets",
            "prothrombin",
        ]
        df[normalization_fields] = (
            df[normalization_fields] - df[normalization_fields].mean()
        ) / df[normalization_fields].std()

    # Put treatment and response at the front and end of the df respectively
    cols = list(df.columns)
    cols.insert(-1, cols.pop(cols.index("status")))
    cols.insert(0, cols.pop(cols.index("treatment")))
    df = df.loc[:, cols]

    return df


def load_mayo_pbc(
    file_path=None, format_covariates=True, download_if_missing=True, normalize=True,
):
    """
    Loads the Mayo PBC dataset with formatting if desired.

    Parameters
    ----------
        file_path : str : optional (default=None)
            Specify another path for the dataset

            By default the dataset should be stored in the 'datasets' folder in the cwd

        format_covariates : bool : optional (default=True)
            Indicates whether raw data should be loaded without covariate manipulation

        download_if_missing : bool : optional (default=True)
            Download the dataset if it is not downloaded before using 'download_mayo_pbc'

        normalize : bool : optional (default=True)
            Normalize the dataset to prepare it for ML methods

    Returns
    -------
        data : dict object with the following attributes:

            data.description : str
                A description of the Mayo Clinic PBC dataset

            data.dataset_full : numpy.ndarray : 312, 19) or formatted (312, 24)
                The full dataset with features, treatment, and target variables

            data.dataset_full_names : list, size 19 or formatted 24
                List of dataset variables names

            data.features : numpy.ndarray : (312, 17) or formatted (312, 22)
                Each row corresponding to the 17 feature values in order

            data.feature_names : list, size 17 or formatted 22
                List of feature names

            data.treatment : numpy.ndarray : (312,)
                Each value corresponds to the treatment (1 = treat, 0 = control)

            data.response : numpy.ndarray : (312,)
                Each value corresponds to one of the outcomes (0 = alive, 1 = liver transplant, 2 = dead)
    """
    # Check that the dataset exists
    directory_path, dataset_path = get_download_paths(
        file_path=file_path, file_directory="datasets", file_name="mayo_pbc.text",
    )
    # Fill above path if not
    if not os.path.exists(dataset_path):
        if download_if_missing:
            download_mayo_pbc(directory_path)
        else:
            raise FileNotFoundError(
                "The dataset does not exist."
                "Use the 'download_mayo_pbc' function to download the dataset."
            )

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
        "The data is from the Mayo Clinic trial in primary biliary cholangitis (PBC, formerly cirrhosis) of the liver conducted between 1974 and 1984."
        "A total of 424 PBC patients, referred to Mayo Clinic during that ten-year interval, met eligibility criteria for the randomized placebo controlled trial of the drug D-penicillamine."
        "The first 312 cases in the data set participated in the randomized trial and contain largely complete data."
        "For modeling purposes, alive (target=0) will be modelled against a resulting transplant (1) and death (2)."
    )

    # Fields dropped to split the data for the user
    drop_fields = ["status", "treatment"]

    return {
        "description": description,
        "dataset_full": df.values,
        "dataset_full_names": np.array(df.columns),
        "features": df.drop(drop_fields, axis=1).values,
        "feature_names": np.array(
            list(filter(lambda x: x not in drop_fields, df.columns))
        ),
        "treatment": df["treatment"].values,
        "response": df["status"].values,
    }
