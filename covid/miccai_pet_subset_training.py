import os
import sys
import tempfile
import shutil
from glob import glob
import logging
from sklearn import preprocessing
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from cachedataset import *
from time import time
from unet_rw import *
import cv2
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import monai
from monai.data import NiftiDataset, create_test_image_3d
from monai.networks.layers import Norm
from monai.transforms import \
    Compose, LoadNiftid, AddChanneld, ScaleIntensityRanged, CropForegroundd, \
    RandCropByPosNegLabeld, RandAffined, Spacingd, SpatialPadd, Orientationd, ToTensord, ScaleIntensityd

from monai.metrics import compute_meandice
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
monai.config.print_config()


##********************************************************************
## Loading PET Data
##********************************************************************
root= "/local-scratch/tgatsak/PSMA_Data/hecktor_nii/"
files = sorted(os.listdir(root))
portion_of_pet_data = files[1:26]


ct_scans = list()
pet_scans = list()
segs = list()
resized_pets = list()

for folder in portion_of_pet_data:
    full_path = root+folder
    fs = os.listdir(full_path)
    
    pet_file = [i for i in fs if '_pt.nii' in i][0]
    pet_scans.append(full_path+"/"+pet_file)
    
    ct_file = [i for i in fs if '_ct.nii' in i][0]
    ct_scans.append(full_path+"/"+ct_file)
    
    segmentation = [i for i in fs if '_gtvt.nii' in i][0]
    segs.append(full_path+"/"+segmentation)
    
    resized_pt = [i for i in fs if 'resized_pt.nii' in i][0]
    resized_pets.append(full_path+"/"+resized_pt)

data_dicts = [{'image': image_name, 'label': label_name}
                for image_name, label_name in zip(resized_pets, segs)]
train_files = data_dicts[:]

print('train files: ', train_files)
#print('clean files: ', clean_files)
print('done storing data files')



##********************************************************************
## Transforms used:
##********************************************************************
train_transforms = Compose([
    LoadNiftid(keys=['image', 'label']),
    AddChanneld(keys=['image', 'label']),
    Spacingd(keys=['image', 'label'], pixdim=(2.5, 2.5, 2.), interp_order=(3, 0), mode='nearest'),
    #SpatialPadd(keys=['image', 'label'], spatial_size =(100, 00, 150)),
    #Orientationd(keys=['image', 'label'], axcodes='RAS'),
#     ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    ScaleIntensityRanged(keys=['image'], a_min=-0.024, a_max=37.7, b_min=0.0, b_max=1.0, clip=True),
    #CropForegroundd(keys=['image', 'label'], source_key='image'),
    # randomly crop out patch samples from big image based on pos / neg ratio
    # the image centers of negative samples must be in valid image area
    RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label', size=(96, 96, 96), pos=1,
                          neg=1, num_samples=4, image_key='image', image_threshold=0),
    # user can also add other random transforms
#     RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=1.0, spatial_size=(96, 96, 96),
#                 rotate_range=(0, 0, np.pi/15), scale_range=(0.1, 0.1, 0.1)),
    ToTensord(keys=['image', 'label'])])



##********************************************************************
## Datasets and data loaders:
##********************************************************************
# create a training data loader
train_ds = CacheDataset(
    data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=list_data_collate)

print('Datasets cached!')


##********************************************************************
## Model initialization:
##********************************************************************
print('initializing the model')
import torch
from monai.networks.utils import one_hot

t1 = time()
device = torch.device("cuda:0")

model = monai.networks.nets.UNet(dimensions=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256),
                                 strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH).to(device)
loss_function = monai.losses.DiceLoss(to_onehot_y=True, do_softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

##********************************************************************
## Start training
##********************************************************************

epoch_num = 400
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()

for epoch in range(epoch_num):
    print('-' * 10)
    print(f"epoch {epoch + 1}/{epoch_num}")
    model.train()
    epoch_loss = 0
    step = 0
    
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    if (epoch+1)%25==0:
        path = "/local-scratch/tgatsak/PSMA_Data/hecktor_nii/trained_models/miccai_25petscans_aug12_"+str(epoch+1)+"_epochs.pth"
        torch.save(model.state_dict(), path)
    torch.save(model.state_dict(), "/local-scratch/tgatsak/PSMA_Data/hecktor_nii/trained_models/backup_aug12_best_metric_model.pth")
    
    
    
with open('/local-scratch/tgatsak/PSMA_Data/hecktor_nii/trained_models/aug12_miccai_25pets_losses.txt', 'w') as filehandle:
    for listitem in epoch_loss_values:
        filehandle.write('%s\n' % listitem)
