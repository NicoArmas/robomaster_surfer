import zipfile
import wget
import os
import logging
from tqdm import tqdm
import requests
import glob


TRAIN_SET = {
    "url": "https://drive.switch.ch/index.php/s/ETqGSXZXABDv6RE/download",
    "zip_name": "train.zip",
    "folder_name": "train",
    "extraction_path": "../data",
}

VALIDATION_SET = {
    "url": "https://drive.switch.ch/index.php/s/CtR090Rd5AvGC6R/download",
    "zip_name": "validation.zip",
    "folder_name": "validation",
    "extraction_path": "../data",
}

TEST_SET = {
    "url": "https://drive.switch.ch/index.php/s/bxnqNmYhOQf4kVg/download",
    "zip_name": "test.zip",
    "folder_name": "test",
    "extraction_path": "../data",
}


def download(url, filename):
    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(filename, 'wb') as file:
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


def unzip(archive, path):
    with zipfile.ZipFile(archive) as zf:
        for member in tqdm(zf.infolist(), desc=f'Extracting {archive}', unit='files'):
            try:
                zf.extract(member, path)
            except zipfile.error as e:
                logging.error(f"Unable to extract {member.filename}: {e}")


def download_dataset(train=True, validation=True, test=True):
    if not os.path.exists("robomaster_surfer/vision/tmp"):
        logging.debug("Creating tmp folder")
        os.mkdir("robomaster_surfer/vision/tmp")

    if os.getcwd().__contains__("robomaster_surfer"):
        os.chdir("robomaster_surfer/vision/tmp")
    else:
        user = os.environ["USER"]
        os.chdir(f"/home/{user}/dev_ws/src/robomaster_surfer/vision/tmp")

    if train:
        logging.info("Downloading train set")
        if download(TRAIN_SET["url"], TRAIN_SET["zip_name"]):
            unzip(TRAIN_SET["zip_name"], TRAIN_SET["extraction_path"])
    if validation:
        logging.info("Downloading validation set")
        if download(TRAIN_SET["url"], TRAIN_SET["zip_name"]):
            unzip(TRAIN_SET["zip_name"], TRAIN_SET["extraction_path"])
    if test:
        logging.info("Downloading test set")
        if wget.download(TRAIN_SET["url"], TRAIN_SET["zip_name"]):
            unzip(TRAIN_SET["zip_name"], TRAIN_SET["extraction_path"])

    if os.getcwd().__contains__("tmp"):
        os.chdir("..")
        os.system("rm -rf tmp")
        os.chdir('data')
        for folder in glob.glob("*"):
            if os.path.isdir(folder):
                os.chdir(folder)
            else:
                continue
            for zf in glob.glob("*.zip"):
                unzip(zf, zf.split('.')[0])
                os.system(f"rm {zf}")
            os.chdir("..")

