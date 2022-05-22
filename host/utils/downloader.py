import zipfile
import wget
import os
import logging
from tqdm import tqdm
import requests
import glob

logging.basicConfig(level=logging.INFO)


def unzip(archive, out_dir):
    """
    It extracts the contents of a zip file into a directory

    :param archive: The path to the zip file you want to extract
    :param out_dir: The directory to extract the archive into
    :param path: The path to the file to extract. If this is not provided, all files are extracted
    """
    with zipfile.ZipFile(archive) as zf:
        for member in tqdm(zf.infolist(), desc=f'Extracting {archive} into {out_dir}', unit='files'):
            try:
                zf.extract(member, out_dir)
            except zipfile.error as e:
                logging.error(f"Unable to extract {member.filename}: {e}")


class Downloader:
    train_set = {
        "synthetic": {
            "url": "https://drive.switch.ch/index.php/s/ETqGSXZXABDv6RE/download",
            "zip_name": "train_synthetic.zip",
            "folder_name": "../data/train_synthetic",
            "extraction_path": "../data",
            "format": "png",
        },
        "real": {
            "url": None,
            "zip_name": "train_real.zip",
            "folder_name": "../data/train_real",
            "extraction_path": "../data",
            "format": "mp4",
        },
        "mixed": {
            "url": None,
            "zip_name": "train_mixed.zip",
            "folder_name": "../data/train_mixed",
            "extraction_path": "../data",
            "format": "mp4",
        }
    }

    validation_set = {
        "synthetic": {
            "url": "https://drive.switch.ch/index.php/s/CtR090Rd5AvGC6R/download",
            "zip_name": "validation_synthetic.zip",
            "folder_name": "../data/validation_synthetic",
            "extraction_path": "../data",
        },
        "real": {
            "url": None,
            "zip_name": "validation_real.zip",
            "folder_name": "../data/validation_real",
            "extraction_path": "../data",
        },
        "mixed": {
            "url": None,
            "zip_name": "validation_mixed.zip",
            "folder_name": "../data/validation_mixed",
            "extraction_path": "../data",
        }
    }

    test_set = {
        "synthetic": {
            "url": "https://drive.switch.ch/index.php/s/bxnqNmYhOQf4kVg/download",
            "zip_name": "test_synthetic.zip",
            "folder_name": "../data/test_synthetic",
            "extraction_path": "../data",
        },
        "real": {
            "url": None,
            "zip_name": "test_real.zip",
            "folder_name": "../data/test_real",
            "extraction_path": "../data",
        },
        "mixed": {
            "url": None,
            "zip_name": "test_mixed.zip",
            "folder_name": "../data/test_mixed",
            "extraction_path": "../data",
        }
    }

    def __init__(self, dataset_type, train=False, validation=False, test=False):
        """
        The function takes in a string, and if the string is not one of the three valid values, it raises a ValueError

        :param dataset_type: This is the type of dataset you want to use. Valid values are: synthetic, real, mixed
        :param train: Whether to load the training set (True), the validation set (False), or both (None), defaults to False
        (optional)
        :param validation: If True, the dataset will be split into training and validation sets, defaults to False
        (optional)
        :param test: If True, the dataset will be the test dataset, defaults to False (optional)
        """
        if dataset_type.lower() not in ["synthetic", "real", "mixed"]:
            raise ValueError(f"Invalid dataset type: {dataset_type}\nValid values are: synthetic, real, mixed")
        self.dataset_type = dataset_type.lower()
        self.train = train
        self.validation = validation
        self.test = test

    def _download(self, part):
        """
        It downloads the file from the url, and saves it to the zip_name

        :param part: This is the part of the dataset that we want to download
        :return: A boolean value.
        """
        # Streaming, so we can iterate over the response.
        response = requests.get(part[self.dataset_type]["url"], stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(part[self.dataset_type]["zip_name"], 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            logging.error("Something went wrong")
            return False
        else:
            logging.info("Download complete")
            return True

    def download(self):
        """
        It downloads the dataset, unzips it, and then unzips, if any, the zip files in the extracted folder.
        """
        if not os.path.exists("robomaster_surfer/vision/.tmp"):
            logging.debug("Creating .tmp folder")
            os.mkdir("robomaster_surfer/vision/.tmp")
        if not os.path.exists("robomaster_surfer/vision/data"):
            logging.debug("Creating data folder")
            os.mkdir("robomaster_surfer/vision/data")

        if os.getcwd().__contains__("robomaster_surfer"):
            os.chdir("robomaster_surfer/vision/.tmp")
        else:
            user = os.environ["USER"]
            os.chdir(f"/home/{user}/dev_ws/src/robomaster_surfer/vision/.tmp")

        try:
            if self.train:
                logging.info("Downloading train set")
                if self._download(self.train_set):
                    unzip(self.train_set[self.dataset_type]["zip_name"],
                          self.train_set[self.dataset_type]["extraction_path"])
                    os.system(f'mv {self.train_set[self.dataset_type]["extraction_path"]}/train '
                              f'{self.train_set[self.dataset_type]["folder_name"]}')
            if self.validation:
                logging.info("Downloading validation set")
                if self._download(self.validation_set):
                    unzip(self.validation_set[self.dataset_type]["zip_name"],
                          self.validation_set[self.dataset_type]["extraction_path"])
                    os.system(f'mv {self.validation_set[self.dataset_type]["extraction_path"]}/validation '
                              f'{self.validation_set[self.dataset_type]["folder_name"]}')
            if self.test:
                logging.info("Downloading test set")
                if self._download(self.test_set):
                    unzip(self.test_set[self.dataset_type]["zip_name"],
                          self.test_set[self.dataset_type]["extraction_path"])
                    os.system(f'mv {self.test_set[self.dataset_type]["extraction_path"]}/test '
                              f'{self.test_set[self.dataset_type]["folder_name"]}')

            if os.getcwd().__contains__(".tmp"):
                os.chdir("../..")
                os.system("rm -rf .tmp")
                os.chdir('data')
                for folder in glob.glob("*"):
                    if os.path.isdir(folder):
                        os.chdir(folder)
                    else:
                        continue
                    for zf in glob.glob("*.zip"):
                        unzip(zf, zf.split('.')[0])
                        os.system(f"rm {zf}")
                    os.chdir("../..")
        except KeyboardInterrupt:
            logging.info("Download interrupted")
            if os.getcwd().__contains__(".tmp"):
                os.chdir("../..")
                os.system("rm -rf .tmp")
