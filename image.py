# image Dataset
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from skimage import io, transform


class ImageDataset(Dataset):

    def __init__(self):

        self.path_dir = os.path.join(os.getcwd() + "/vascular_tree_data/training_data/input")

        self.file_names = []
        
        for name in os.listdir(self.path_dir): 
          if os.path.isfile(os.path.join(self.path_dir, name)) and name.endswith('x.png'):
            # print(name)
            self.file_names.append(name)
        self.file_names.sort()
        self.dataset_length = len(self.file_names)

    def __len__(self):
        return self.dataset_length - 1

    def __getitem__(self, idx):

        image_path = os.path.join(self.path_dir, self.file_names[idx])
        image = io.imread(image_path)

        return self.file_names[idx], image


class ImageTestDataset(Dataset):

    def __init__(self):

        self.path_dir = os.path.join(os.getcwd() + "/vascular_tree_data/test_data/input")

        self.file_names = []
        
        for name in os.listdir(self.path_dir): 
          if os.path.isfile(os.path.join(self.path_dir, name)) and name.endswith('x.png'):
            # print(name)
            self.file_names.append(name)
        self.file_names.sort()
        self.dataset_length = len(self.file_names)

    def __len__(self):
        return self.dataset_length - 1

    def __getitem__(self, idx):

        image_path = os.path.join(self.path_dir, self.file_names[idx])
        image = io.imread(image_path)

        return self.file_names[idx], image
