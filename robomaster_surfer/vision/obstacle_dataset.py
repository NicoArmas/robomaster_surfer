import glob
import os

import cv2
import torch
import torchvision.transforms
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from torch.utils.data.dataset import T_co


class ObstacleDataset(Dataset):

    def __init__(self, data_dir, transform=transforms.ToTensor()):
        self.samples = []
        self.data_dir = data_dir
        self.transform = transform
        targets = pd.read_csv(f'{os.getcwd()}/{data_dir}/targets.csv').to_numpy()

        for idx, target in targets:
            target = list(map(int, target.strip('(').strip(')').split(',')))
            if len(target) == 1:
                continue
            t = target[0] + target[1] * 2 + target[2] * 4
            target = torch.zeros(8, dtype=torch.float32)
            target[t] = 1
            self.samples.append((f'img_{idx}.png', target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]

        img = cv2.imread(f'{self.data_dir}/{img}')

        if self.transform is not None:
            img = self.transform(img)
        return img, label


if __name__ == '__main__':
    dataset = ObstacleDataset('robomaster_surfer/vision/data/obstacle_avoidance')
    print(len(dataset))
    print(dataset[0])
    print('The shape of tensor for 50th image in train dataset: ', dataset[-1][0].shape)
    print('The label for 50th image in train dataset: ', dataset[-1][1])
