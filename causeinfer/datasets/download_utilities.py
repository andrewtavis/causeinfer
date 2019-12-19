# =============================================================================
# Contents
# --------
#   0. No Class
#       download_file
#       get_download_paths
# =============================================================================

import os
import requests
import urllib
import zipfile

def download_file(url: str, output_path: str, zip_file = False):
    """
    Downloads a file from a url to a specified path

    Parameters
    ----------
        url : str
            the URL from which the file can be downloaded from

        output_path : str
            a user specified path, which defaults to a 'files' folder in the cwd
    """
    # Check if the file exists, and delete if so
    if os.path.isfile(output_path):
        os.remove(output_path)

    print("Downloading file to '{}' ...".format(output_path))
    
    res = requests.get(url)
    # Check if the response is ok (200)
    status_code = int(res.status_code)
    if status_code == 200:
        if zip_file == True:
            file = urllib.request.urlretrieve(url, "{}.zip".format(output_path))
            with zipfile.ZipFile("{}.zip".format(output_path), 'r') as zip_ref:
                print("Unzipping '{}'.zip ...".format(output_path))
                zip_ref.extractall("{}.zip".format(output_path).split(".zip")[0])
                os.remove("{}.zip".format(output_path))
                print("File unzipped - deleting .zip file")

        else:
            with open(output_path, 'wb') as file:
                # A chunk of 128 bytes
                for chunk in res:
                    file.write(chunk)
                    
    elif status_code == 404:
        raise Exception('Wrong URL (' + url + ').')


def get_download_paths(file_path, file_folder = 'files', file_name = 'file'):
    """
    Derives paths for a file folder and a file path
    """
    if file_path is None:
        file_path = os.path.join(os.getcwd() + '/' + file_folder)

    file_folder = os.path.join(file_path + '/' + file_name)
    
    return file_path, file_folder