import argparse
import glob
import logging
import os

import cv2
import numpy as np
from tqdm.auto import tqdm


def preprocess_data(data_dir, output_dir):

    if not data_dir[-1] == '/':
        data_dir += '/'
    if not output_dir[-1] == '/':
        output_dir += '/'

    cwd = os.getcwd()

    if not os.path.exists("./robomaster_surfer/vision/data/"):
        logging.debug("Creating preprocessed folder")
        os.mkdir("./robomaster_surfer/vision/data/")

    if os.getcwd().__contains__("robomaster_surfer"):
        os.chdir("./robomaster_surfer/vision/data/")
    else:
        user = os.environ["USER"]
        if user == "usi":
            os.chdir(f"/home/{user}/dev_ws/src/robomaster_surfer/robomaster_surfer/vision/data")
        elif user == "alind":
            os.chdir(f"/home/{user}/PycharmProjects/robomaster_surfer/robomaster_surfer/vision/data")

    counter = 0
    split = f'../raw_data'

    for i, folder in tqdm(enumerate(os.listdir(split))):
        if not os.path.exists(f"{folder}".split("/")[-1]):
            os.mkdir(f"{folder}".split("/")[-1])
        for i, file in tqdm(enumerate(glob.glob(f"{split}/{folder}/*.png"))):
            image = cv2.imread(f'{split}/{folder}/img_{i}.png')
            image = np.array(cv2.resize(image, (128, 128))).astype(np.uint8)
            cv2.imwrite(f"{folder}/img_{i}.png", image)
            counter += 1
            logging.debug(f"Preprocessed {file}")

    os.chdir(cwd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument("--data_dir", type=str, default="./robomaster_surfer/vision/data/obstacle_avoidance/",
                        help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="./robomaster_surfer/vision/data/preprocessed/",
                        help="Path to output directory")
    args = parser.parse_args()
    preprocess_data(args.data_dir, args.output_dir)
