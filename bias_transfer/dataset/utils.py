import os
import tarfile
import zipfile
import requests
import shutil
import numpy as np
from io import BytesIO


def compute_mean_std(train_set):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    mean = np.mean(train_set.dataset.data, axis=(0, 1, 2)) / 255
    std = np.std(train_set.dataset.data, axis=(0, 1, 2)) / 255
    return mean, std


def create_ImageFolder_format(dataset_dir: str):
    """
    This method is responsible for separating validation images into separate sub folders

    Args:
        dataset_dir (str): "/path_to_your_dataset/dataset_folder"
    """
    val_dir = os.path.join(dataset_dir, "val")
    img_dir = os.path.join(val_dir, "images")

    fp = open(os.path.join(val_dir, "val_annotations.txt"), "r")
    data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present and move images into proper folders
    for img, folder in val_img_dict.items():
        newpath = os.path.join(img_dir, folder)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))


def get_dataset(url: str, data_dir: str, dataset_cls: str, download: bool) -> str:
    """
    Downloads the dataset from an online downloadable link and
    sets up the folders according to torch ImageFolder required
    format

    Args:
        url (str): download link of the dataset from the internet
        data_dir (str): the directory where to download the dataset
        dataset_cls (str): name of the dataset's folder
        download (bool): download the data if not present
    Returns:
        dataset_dir (str): full path to the dataset incl. dataset folder
    """
    dataset_dir = os.path.join(data_dir, dataset_cls)
    if os.path.isdir(dataset_dir):
        print("Images already downloaded...")
        return dataset_dir
    if not download:
        raise FileNotFoundError("Images not present but download not allowed! Please check data folder!")
    os.makedirs(dataset_dir)
    r = requests.get(url, stream=True)
    print("Downloading " + url)
    if url.endswith(".zip"):
        zip_ref = zipfile.ZipFile(BytesIO(r.content))
        zip_ref.extractall(dataset_dir)
        extract_dir = os.path.join(dataset_dir, sorted(zip_ref.namelist())[0])
        zip_ref.close()
    elif url.endswith(".tar"):
        tar_ref = tarfile.TarFile(fileobj=BytesIO(r.content))
        tar_ref.extractall(dataset_dir)
        extract_dir = os.path.join(dataset_dir, sorted(tar_ref.getnames())[0])
        tar_ref.close()
    else:
        raise NotImplementedError("Unsupported dataset format.")
    # move to final destination
    for content in os.listdir(extract_dir):
        shutil.move(os.path.join(extract_dir, content), dataset_dir)
    shutil.rmtree(extract_dir)
    return dataset_dir
