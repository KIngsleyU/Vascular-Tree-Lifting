# ground truth dataset 
from torch.utils.data import Dataset
import glob
import numpy as np
import scipy.io as io
import os
import cv2
import torch

class GroundTruthDataset(Dataset):

    def __init__(self):
        self.path_dir = os.path.join(os.getcwd() + "/vascular_tree_data/training_data/ground_truth")

        self.ground_truth = []
        
        for name in os.listdir(self.path_dir): 
          if os.path.isfile(os.path.join(self.path_dir, name)) and name.endswith('ground_truth.mat'):
            self.ground_truth.append(name)

        self.ground_truth.sort()

        self.dataset_length = len(self.ground_truth)

    def __len__(self):
        return self.dataset_length - 1

    def getItem (self, name):
        name = name.split('x.png', 1)
        name = name[0] + "ground_truth.mat"
        gt_path = os.path.join(self.path_dir, name)
        mat_directory = io.loadmat(gt_path)
        truth_model = mat_directory['I']
        return torch.from_numpy(truth_model)


class GroundTruthTestDataset(Dataset):

    def __init__(self):
        self.path_dir = os.path.join(os.getcwd() + "/vascular_tree_data/test_data/ground_truth")

        self.ground_truth = []
        
        for name in os.listdir(self.path_dir): 
          if os.path.isfile(os.path.join(self.path_dir, name)) and name.endswith('ground_truth.mat'):
            self.ground_truth.append(name)

        self.ground_truth.sort()

        self.dataset_length = len(self.ground_truth)

    def __len__(self):
        return self.dataset_length - 1

    def getItem (self, name):
        name = name.split('x.png', 1)
        name = name[0] + "ground_truth.mat"
        gt_path = os.path.join(self.path_dir, name)
        mat_directory = io.loadmat(gt_path)
        truth_model = mat_directory['I']
        return torch.from_numpy(truth_model)
