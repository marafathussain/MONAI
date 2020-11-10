import sys
import logging
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import NiftiDataset
from monai.transforms import Compose, CastToType, SpatialPad, AddChannel, ScaleIntensity, Resize, RandRotate90, RandZoom, ToTensor
import os
import matplotlib.pyplot as plt


data_dir = '/home/marafath/scratch/kits/training'

for patient in os.listdir(data_dir):
    
    ct = nib.load(os.path.join(data_dir,patient,'imaging.nii.gz'))
    ct_img = ct.get_fdata()
    np.save(os.path.join(data_dir,patient,'image'),ct_img)

    gt = nib.load(os.path.join(data_dir,patient,'segmentation.nii.gz'))
    gt = gt.get_fdata()
    gt[gt == 2] = 0
    np.save(os.path.join(data_dir,patient,'mask'),gt)