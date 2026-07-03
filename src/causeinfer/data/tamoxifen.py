# SPDX-License-Identifier: BSD-3-Clause
"""
Pintilie Tamoxifen
------------------

A dataset on competing risks from a breast cancer trial of tamoxifen versus control.

An example notebook is planned (see issue #19).

Description found at:
    https://onlinelibrary.wiley.com/doi/book/10.1002/9780470870709

Based on
    Pintilie, M. "Competing Risks: A Practical Perspective". Wiley, 2006.
    URL: https://onlinelibrary.wiley.com/doi/book/10.1002/9780470870709.

The original source is no longer available, so the dataset is bundled in
'src/causeinfer/data/datasets/tamoxifen.txt'.

Contents
    download_tamoxifen (deprecated),
    _format_data,
    load_tamoxifen
"""

import os

import numpy as np
import pandas as pd

from causeinfer.data.download_utils import get_download_paths


def download_tamoxifen(data_path=None, url=""):
    """
    ! download_tamoxifen is deprecated as the dataset is bundled in the repository.

    The original source is no longer available online, so the dataset is distributed
    directly in 'src/causeinfer/data/datasets/tamoxifen.txt'.

    Parameters
    ----------
        data_path : str : optional (default=None)
            A user specified path for where the data should go.

        url : str
            The url from which the data is to be downloaded.

    Returns
    -------
        None
        This function is deprecated and raises NotImplementedError. The dataset is
        bundled in 'src/causeinfer/data/datasets/tamoxifen.txt'.
    """
    raise NotImplementedError(
        "download_tamoxifen is deprecated as the original source is no longer "
        "available. The dataset is bundled in "
        "'src/causeinfer/data/datasets/tamoxifen.txt'."
    )


def _format_data(df, format_covariates=True, normalize=True):
    """
    Formats the data upon loading for consistent data preparation.

    Parameters
    ----------
        df : pd.DataFrame
            The original unformatted version of the data as read from the bundled
            'tamoxifen.txt' file.

        format_covariates : bool : optional (default=True), controlled in load_tamoxifen
            - True: creates dummy columns and encodes the data.

            - False: only steps for data readability will be taken.

        normalize : bool : optional (default=True), controlled in load_tamoxifen
            Normalize dataset columns to prepare them for ML methods.

    Returns
    -------
        df : pd.DataFrame
            A formated version of the data.
    """
    # Drop non-feature columns: the patient id, the survival time (its event
    # indicator is 'stat' and would leak the response), and the competing-risks
    # time/censor pairs.
    drop_cols = [
        "stnum",
        "survtime",
        "loctime",
        "lcens",
        "axltime",
        "acens",
        "distime",
        "dcens",
        "maltime",
        "mcens",
    ]
    df = df.drop(columns=drop_cols)

    # Recode tx into a binary treatment (T=1 tamoxifen, B=0 control).
    df = df.rename(columns={"tx": "treatment"})
    df["treatment"] = df["treatment"].map({"T": 1, "B": 0})

    if format_covariates:
        # Create dummy columns for the categorical baseline covariates.
        dummy_cols = ["hist", "nodediss", "hrlevel"]
        for col in dummy_cols:
            df = pd.get_dummies(df, columns=[col], prefix=col)
    else:
        # Label-encode the categorical string columns so the resulting
        # DataFrame is fully numeric (castable to float), matching the numeric
        # behavior of mayo_pbc's format_covariates=False path.
        for col in ["hist", "nodediss", "hrlevel"]:
            df[col] = df[col].astype("category").cat.codes

    if normalize:
        normalization_fields = ["pathsize", "hgb", "age"]
        df[normalization_fields] = (
            df[normalization_fields] - df[normalization_fields].mean()
        ) / df[normalization_fields].std()

    # Put treatment and response at the front and end of the df respectively.
    cols = list(df.columns)
    cols.insert(0, cols.pop(cols.index("treatment")))
    cols.append(cols.pop(cols.index("stat")))
    df = df.loc[:, cols]

    df = df.astype(float)

    return df


def load_tamoxifen(
    file_path=None,
    format_covariates=True,
    download_if_missing=True,
    normalize=True,
):
    """
    Loads the Pintilie Tamoxifen dataset with formatting if desired.

    Parameters
    ----------
        file_path : str : optional (default=None)
            Specify another path for the dataset.

            When None, the bundled dataset at
            src/causeinfer/data/datasets/tamoxifen.txt is used.

        format_covariates : bool : optional (default=True)
            Indicates whether raw data should be loaded without covariate manipulation.

        download_if_missing : bool : optional (default=True)
            If True and the dataset is missing, attempt download via
            'download_tamoxifen' (deprecated; raises NotImplementedError). The
            dataset is bundled and is loaded by default from
            src/causeinfer/data/datasets/tamoxifen.txt.

        normalize : bool : optional (default=True)
            Normalize the dataset to prepare it for ML methods.

    Returns
    -------
        data : dict object with the following attributes:

            data.description : str
                A description of the Pintilie Tamoxifen dataset.

            data.dataset_full : numpy.ndarray : (641, 8) or formatted (641, 15)
                The full dataset with features, treatment, and target variables.

            data.dataset_full_names : list, size 8 or formatted 15
                List of dataset variables names.

            data.features : numpy.ndarray : (641, 6) or formatted (641, 13)
                Each row corresponding to the 6 feature values in order.

            data.feature_names : list, size 6 or formatted 13
                List of feature names.

            data.treatment : numpy.ndarray : (641,)
                Each value corresponds to the treatment (1 = tamoxifen, 0 = control).

            data.response : numpy.ndarray : (641,)
                Each value corresponds to the overall event indicator (0 = censored, 1 = event).

    Notes
    -----
        `survtime` and the competing-risks time/censor columns are intentionally
        dropped as post-treatment outcomes to avoid leakage, even though
        `mayo_pbc` retains its baseline time column (`days_since_register`).
    """
    # Default to the bundled dataset so load_tamoxifen() works out of the box.
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), "datasets", "tamoxifen.txt")

    # Check that the dataset exists.
    directory_path, dataset_path = get_download_paths(
        file_path=file_path,
        file_directory="datasets",
        file_name="tamoxifen.txt",
    )
    # Fill above path if not.
    if not os.path.exists(dataset_path):
        if download_if_missing:
            download_tamoxifen(directory_path)
        else:
            raise FileNotFoundError(
                "The dataset does not exist. The dataset is bundled in "
                "'src/causeinfer/data/datasets/tamoxifen.txt'."
            )

    # Read the data.
    df = pd.read_csv(dataset_path)

    # Load formatted or raw data.
    if format_covariates:
        df = (
            _format_data(df, format_covariates=True, normalize=True)
            if normalize
            else _format_data(df, format_covariates=True, normalize=False)
        )

    elif normalize:
        df = _format_data(df, format_covariates=False, normalize=True)

    else:
        df = _format_data(df, format_covariates=False, normalize=False)

    description = (
        'The Tamoxifen dataset from Pintilie, M. "Competing Risks: A Practical Perspective" (Wiley, 2006) '
        "comes from a breast cancer clinical trial comparing tamoxifen (treatment) to a control. "
        "The response 'stat' is the overall event indicator (0 = censored, 1 = event). "
        "Baseline covariates include pathsize, hist, hgb, nodediss, age, and hrlevel. "
        "The survival time and competing-risks time/censor columns are dropped to avoid leakage, "
        "leaving only baseline covariates."
    )

    # Fields dropped to split the data for the user.
    drop_fields = ["stat", "treatment"]

    return {
        "description": description,
        "dataset_full": df.values,
        "dataset_full_names": np.array(df.columns),
        "features": df.drop(drop_fields, axis=1).values,
        "feature_names": np.array(
            list(filter(lambda x: x not in drop_fields, df.columns))
        ),
        "treatment": df["treatment"].values,
        "response": df["stat"].values,
    }
