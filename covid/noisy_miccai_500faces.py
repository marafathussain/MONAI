import os
import sys
import tempfile
import shutil
from glob import glob
import logging
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from cachedataset import CacheDataset, list_data_collate
from time import time
from unet_rw import *
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import monai
from monai.data import NiftiDataset, create_test_image_3d
from monai.transforms import \
    Compose, LoadNiftid, AddChanneld, ScaleIntensityRanged, CropForegroundd, \
    RandCropByPosNegLabeld, RandAffined, Spacingd, SpatialPadd, Orientationd, ToTensord, ScaleIntensityd

from monai.metrics import compute_meandice
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
monai.config.print_config()


##********************************************************************
## Loading MICCAI Data
##********************************************************************
root= "/local-scratch/tgatsak/PSMA_Data/hecktor_nii/"
files = sorted(os.listdir(root))
portion_of_pet_data = files[1:26]


segs = list()
resized_pets = list()

for folder in portion_of_pet_data:
    full_path = root+folder
    fs = os.listdir(full_path)
    
    
    segmentation = [i for i in fs if '_gtvt.nii' in i][0]
    segs.append(full_path+"/"+segmentation)
    
    resized_pt = [i for i in fs if 'resized_pt.nii' in i][0]
    resized_pets.append(full_path+"/"+resized_pt)

data_dicts_clean = [{'image': image_name, 'label': label_name}
                for image_name, label_name in zip(resized_pets, segs)]


noisy_root = "/local-scratch/tgatsak/PSMA_Data/hecktor_nii/noisy_500faces_test2/"
noisy_segs = sorted(glob(os.path.join(noisy_root, '*.nii')))


data_dicts_noisy = [{'image': image_name, 'label': label_name}
                for image_name, label_name in zip(resized_pets, noisy_segs)]



#segs_noisy = sorted(glob(os.path.join(data_root, 'labelsNoisy1', '*.nii.gz')))
#images_first12 = images[:12]


train_files, clean_files = data_dicts_noisy[:12], data_dicts_clean[12:24]
print('train files: ', train_files)
print('clean files: ', clean_files)
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
                          neg=1, num_samples=2, image_key='image', image_threshold=0),
    # user can also add other random transforms
#     RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=1.0, spatial_size=(96, 96, 96),
#                 rotate_range=(0, 0, np.pi/15), scale_range=(0.1, 0.1, 0.1)),
    ToTensord(keys=['image', 'label'])])

##********************************************************************
## Datasets and data loaders:
##********************************************************************
# create a training data loader
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, collate_fn=list_data_collate)


cl_ds = CacheDataset(data=clean_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
cl_loader = DataLoader(cl_ds, batch_size=1, shuffle=True, num_workers=4, collate_fn=list_data_collate)

print('Datasets cached!')


##********************************************************************
## Model initialization:
##********************************************************************
print('initializing the model')
t1 = time()
device = torch.device("cuda:0")

model=UNet()
model = torch.nn.DataParallel(model, device_ids=[0]).to(device)
#loss_function = monai.losses.DiceLoss(do_sigmoid=True)
optimizer = torch.optim.Adam(model.module.params(), 1e-4)


##********************************************************************
## Start training
##********************************************************************


val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()
print(device, 'starting to train')

for epoch in range(400):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{10}")
    model.train(True)
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        print(batch_data['image'].shape)
        meta_net = UNet()
        meta_net.load_state_dict(model.module.state_dict())
        meta_net.to(device)

        inputs, labels = to_var(batch_data['image'],requires_grad=False), to_var(batch_data['label'], requires_grad=False)
        y_f_hat  = meta_net(inputs)
        labels = labels.type(torch.float)
#         print('input and label shapes: ', inputs.shape, labels.shape)
        cost = F.binary_cross_entropy_with_logits(y_f_hat.squeeze(),labels.squeeze(), reduce=False)
            #assign one weight per image
        tmp=torch.mean(cost,1)
        cost=torch.mean(tmp,1)

        eps = to_var(torch.zeros(cost.size()))
        l_f_meta = torch.sum(cost * eps)
            
        meta_net.zero_grad()
        
            #perform a parameter update
        grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
        meta_net.update_params(1e-4, source_params=grads)
            
	      #2nd forward pass and getting the gradients with respect to epsilon
        k=step%12
        if k==0:
            cl=iter(cl_loader)
        data = next(cl)

        step += 1
        val_im = data['image']
        val_gt = data['label']
#         print('shapes clean images and labels: ', val_im.shape, val_gt.shape)
        val_im = to_var(val_im, requires_grad=False)
        val_gt = to_var(val_gt, requires_grad=False)
            
        y_g_hat = meta_net(val_im)
        val_gt = val_gt.type(torch.float)

        l_g_meta = F.binary_cross_entropy_with_logits(y_g_hat.squeeze(),val_gt.squeeze())
            
            
        grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]
        
        #computing and normalizing the weights
        w_tilde = torch.clamp(-grad_eps,min=0)
            
        norm_c = torch.sum(w_tilde)

        if norm_c.item() == 0:
       	    w = w_tilde  
        else:
       	    w = w_tilde /norm_c

        # computing the loss with the computed weights
        # and then perform a parameter update
        y_f_hat = model(inputs)
            
        cost = F.binary_cross_entropy_with_logits(y_f_hat.squeeze(), labels.squeeze(), reduce=False)
        tmp=torch.mean(cost,1)
        cost=torch.mean(tmp,1)
            
        l_f = torch.sum(cost * w)

        optimizer.zero_grad()
        l_f.backward()
        optimizer.step()

        #epoch_loss += loss.item()
        epoch_loss += l_f.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {l_f.item():.4f}")
        #writer.add_scalar("train_loss", l_f.item(), epoch_len * epoch + step)
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch+1)%25==0:
        path = "/local-scratch/tgatsak/PSMA_Data/hecktor_nii/trained_models/miccai_noisy12_clean12petscans_aug24_"+str(epoch+1)+"_epochs.pth"
        torch.save(model.state_dict(), path)
    torch.save(model.state_dict(), "/local-scratch/tgatsak/PSMA_Data/hecktor_nii/trained_models/backup_aug24_best_metric_model.pth")
    
    
    
with open('/local-scratch/tgatsak/PSMA_Data/hecktor_nii/trained_models/aug24_miccai_noisy12_clean12_pets_losses.txt', 'w') as filehandle:
    for listitem in epoch_loss_values:
        filehandle.write('%s\n' % listitem)

