import glob

import cv2
import os
import logging
import argparse

import numpy as np
from tqdm.auto import tqdm


def preprocess_data(data_dir, output_dir):

    if not data_dir[-1] == '/':
        data_dir += '/'
    if not output_dir[-1] == '/':
        output_dir += '/'

    cwd = os.getcwd()

    if not os.path.exists("./host/preprocessed_test"):
        logging.debug("Creating preprocessed folder")
        os.mkdir("./host/data/preprocessed_test")

    if os.getcwd().__contains__("robomaster_surfer"):
        os.chdir("./host/data/preprocessed_test")
    else:
        user = os.environ["USER"]
        if user == "usi":
            os.chdir(f"/home/{user}/dev_ws/src/robomaster_surfer/host/preprocessed_test")
        elif user == "alind":
            os.chdir(f"/home/{user}/PycharmProjects/robomaster_surfer/host/preprocessed_test")

    counter = 0

    for i, folder in tqdm(enumerate(os.listdir(data_dir))):
        if not os.path.exists(f"{folder}".split("/")[-1]):
            os.mkdir(f"{folder}".split("/")[-1])
        for file in tqdm(glob.glob(f"{data_dir}{folder}/*.png")):
            image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            image = np.array(cv2.resize(image, (128, 128))).astype(np.uint8)
            cv2.imwrite(f"{folder}/frame_{counter}.png", image)
            counter += 1
            logging.debug(f"Preprocessed {file}")

    os.chdir(cwd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument("--data_dir", type=str, default="./host/data/test",
                        help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="./host/data/preprocessed_test",
                        help="Path to output directory")
    args = parser.parse_args()
    preprocess_data(args.data_dir, args.output_dir)
