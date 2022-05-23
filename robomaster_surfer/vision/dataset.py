import glob
import os

import cv2
import torch
import torchvision.transforms
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data.dataset import T_co


class LaneDataset(Dataset):

    def __init__(self, data_dir, transform=transforms.ToTensor()):
        self.samples = []
        self.data_dir = data_dir
        self.transform = transform

        for lane in os.listdir(data_dir):
            lane_folder = os.path.join(data_dir, lane)

            for name in os.listdir(lane_folder):
                self.samples.append((name, lane))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]

        img = cv2.imread(f'{self.data_dir}/{label}/{img}')

        if self.transform is not None:
            img = self.transform(img)
        return img, label


if __name__ == '__main__':
    dataset = LaneDataset('robomaster_surfer/vision/data/preprocessed')
    print(len(dataset))
    print(dataset[0])
    print('The shape of tensor for 50th image in train dataset: ', dataset[-1][0].shape)
    print('The label for 50th image in train dataset: ', dataset[-1][1])
