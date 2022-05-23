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

    if not os.path.exists("./robomaster_surfer/vision/preprocessed"):
        logging.debug("Creating preprocessed folder")
        os.mkdir("./robomaster_surfer/vision/data/preprocessed")

    if os.getcwd().__contains__("robomaster_surfer"):
        os.chdir("./robomaster_surfer/vision/data/preprocessed")
    else:
        user = os.environ["USER"]
        if user == "usi":
            os.chdir(f"/home/{user}/dev_ws/src/robomaster_surfer/robomaster_surfer/vision/preprocessed")
        elif user == "alind":
            os.chdir(f"/home/{user}/PycharmProjects/robomaster_surfer/robomaster_surfer/vision/preprocessed")

    counter = 0

    for i, folder in tqdm(enumerate(os.listdir('../train'))):
        if not os.path.exists(f"{folder}".split("/")[-1]):
            os.mkdir(f"{folder}".split("/")[-1])
        for file in tqdm(glob.glob(f"../train/{folder}/*.png")):
            image = cv2.imread(file)
            image = np.array(cv2.resize(image, (128, 128))).astype(np.uint8)
            cv2.imwrite(f"{folder}/frame_{counter}.png", image)
            counter += 1
            logging.debug(f"Preprocessed {file}")

    os.chdir(cwd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument("--data_dir", type=str, default="./robomaster_surfer/vision/data/train/",
                        help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="./robomaster_surfer/vision/data/preprocessed/",
                        help="Path to output directory")
    args = parser.parse_args()
    preprocess_data(args.data_dir, args.output_dir)
