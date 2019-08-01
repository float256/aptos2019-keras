import os
import json
from zipfile import ZipFile
import pandas as pd


def download_dataset(username: str, token: str, dataset_folder: str,
                     config_folder="~/"):
    """This function download dataset using Kaggle API. Please note that Kaggle
    must be installed in a virtual environment in which the script is running
    using this function.

    Args:
        username: Username on Kaggle
        token: The token issued by Kaggle to use the API. In case you do not
            know where to get it, use the documentation:
            https://www.kaggle.com/docs/api
        dataset_folder: Folder to download the dataset
        config_folder: The directory where json should be stored with the
            config for Kaggle API. The default home directory is selected.
    """
    if os.system("pip install kaggle") != 0:
        raise ConnectionError('Error loading the package "kaggle"')
    try:
        os.makedirs(f"{config_folder}.kaggle")
    except FileExistsError:
        pass
    with open(f"{config_folder}.kaggle/kaggle.json", "w") as file:
        json.dump({"username": username, "key": token}, file)
    command_exit_code = os.system(
        f'kaggle competitions download -c aptos2019-blindness-detection -p '
        f'{dataset_folder}')
    if command_exit_code != 0:
        raise ConnectionError("Error loading dataset. You may have "
                              "entered incorrect data.")

    # Extracting the training dataset
    os.makedirs(dataset_folder + "train_images/")
    ZipFile(dataset_folder + "train_images.zip").extractall(
        dataset_folder + "train_images/")
    os.remove(dataset_folder + "train_images.zip")

    # Extracting test dataset
    os.makedirs(dataset_folder + "test_images/")
    ZipFile(dataset_folder + "test_images.zip").extractall(
        dataset_folder + "test_images/")
    os.remove(dataset_folder + "test_images.zip")


def process_csv(dataframe: pd.DataFrame, image_column_name: str,
                label_column_name: str,
                folder_with_images: str) -> pd.DataFrame:
    """This function process Pandas DataFrame, which contains image filenames
    and their corresponding labels.

    Args:
        dataframe: Pandas DataFrame object. It should consist of 2 columns
        image_column_name: The name of the column containing the image
            filenames
        label_column_name: The name of the column containing the image
            labels
        folder_with_images: Folder with images

    Returns:
        dataframe: processed DataFrame with full paths to images
    """
    dataframe[image_column_name] = dataframe[image_column_name].apply(
        lambda x: f"{folder_with_images}{x}.png")
    dataframe[label_column_name] = dataframe[label_column_name].astype('str')
    return dataframe

