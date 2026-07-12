# SPDX-License-Identifier: BSD-3-Clause
"""
Utility functions for downloading data.

Based on
    Kuchumov, A. pyuplift: Lightweight uplift modeling framework for Python. (2019).
    URL: https://github.com/duketemon/pyuplift.
    License: https://github.com/duketemon/pyuplift/blob/master/LICENSE.

Contents
    download_file,
    get_download_paths
"""

import os
import urllib
import zipfile

import requests


def download_file(url: str, output_path: str, zip_file=False):
    """
    Download a file from a url to a specified path.

    Parameters
    ----------
    url : str
        The URL from which the file can be downloaded from.

    output_path : str
        A user specified path, which defaults to a 'files' folder in the cwd.

    zip_file : bool (default=False)
        Whether to zip the resulting file.
    """
    print(f"Attempting to download file to '{output_path}'...")

    # Set header for requests.get(), which is required for some websites.
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
    }

    res = requests.get(url, headers=headers)
    # Check if the response is ok (200).
    status_code = res.status_code
    if status_code == 200:
        if zip_file:
            file = urllib.request.urlretrieve(url, output_path)
            with zipfile.ZipFile(output_path, "r") as zip_ref:
                print(f"Unzipping '{output_path}'...")
                zip_ref.extractall(output_path.split(".zip")[0])

                os.remove(output_path)
                print("File unzipped - deleting .zip file")

                print("Download complete")

        else:
            with open(output_path, "wb") as file:
                # A chunk of 128 bytes.
                for chunk in res:
                    file.write(chunk)
                print("Download complete")

    elif status_code == 403:
        raise Exception(f"Forbidden URL: {url}")

    elif status_code == 404:
        raise Exception(f"Wrong URL: {url}")


def get_download_paths(file_path, file_directory="files", file_name="file"):
    """
    Derive paths for a file folder and a file.

    Parameters
    ----------
    file_path : str
        A user specified path that the data should go to.

    file_directory : str (default=files)
        A user specified directory.

    file_name : str (default=file)
        The name to call the file.

    Returns
    -------
    str, str
        The directory path and file path of the download.
    """
    if file_path is None:
        directory_path = os.path.join(f"{os.getcwd()}/{file_directory}")
        file_path = os.path.join(f"{directory_path}/{file_name}")

    else:
        directory_path = file_path.split("/")[0]
        file_path = file_path

    return directory_path, file_path
