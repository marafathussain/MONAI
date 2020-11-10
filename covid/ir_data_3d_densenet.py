import sys
import logging
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import NiftiDataset
from monai.transforms import Compose, CastToType, SpatialPad, AddChannel, ScaleIntensity, Resize, RandRotate, SpatialCrop, ToTensor
import os
import matplotlib.pyplot as plt

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Training data paths
    data_dir = '/home/marafath/scratch/iran_organized_data2'

    covid_pat = 0
    non_covid_pat = 0

    images_p = []
    labels_p = []
    images_n = []
    labels_n = []

    for patient in os.listdir(data_dir):
        if int(patient[-1]) == 0 and non_covid_pat > 236:
            continue 

        if int(patient[-1]) == 1:
            covid_pat += 1
            for series in os.listdir(os.path.join(data_dir,patient)):
                labels_p.append(1)
                images_p.append(os.path.join(data_dir,patient,series,'cropped_and_resized_image.nii.gz'))
        else:
            non_covid_pat += 1
            for series in os.listdir(os.path.join(data_dir,patient)):
                labels_n.append(0)
                images_n.append(os.path.join(data_dir,patient,series,'cropped_and_resized_image.nii.gz'))
            
    train_images = []
    train_labels = []

    val_images = []
    val_labels = []

    for i in range(0,len(images_p)):
        if i < 407:
            train_images.append(images_p[i])
            train_labels.append(labels_p[i])
        else:
            val_images.append(images_p[i])
            val_labels.append(labels_p[i])

    for i in range(0,len(images_n)):
        if i < 405:
            train_images.append(images_n[i])
            train_labels.append(labels_n[i])
        else:
            val_images.append(images_n[i])
            val_labels.append(labels_n[i])  
    
    train_labels = np.asarray(train_labels,np.int64)
    val_labels = np.asarray(val_labels,np.int64)


    # Define transforms
    train_transforms = Compose([
        ScaleIntensity(),
        AddChannel(),
        RandRotate(range_x=10.0, range_y=10.0, range_z=10.0, prob=0.5),
        #SpatialPad((256, 256, 92), mode='constant'),
        #SpatialCrop((256, 256, 92),roi_start=(0, 0, 0), roi_end=(255, 255, 91)),
        ToTensor()
    ])
    
    val_transforms = Compose([
        ScaleIntensity(),
        AddChannel(),
        #SpatialPad((256, 256, 92), mode='constant'),
        #SpatialCrop((256, 256, 92),roi_start=(0, 0, 0), roi_end=(255, 255, 91)),
        ToTensor()
    ])

    # create a training data loader
    train_ds = NiftiDataset(image_files=train_images, labels=train_labels, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    # create a validation data loader
    val_ds = NiftiDataset(image_files=val_images, labels=val_labels, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=2, pin_memory=torch.cuda.is_available())
    
    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device('cuda:0')
    model = monai.networks.nets.densenet.densenet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
    ).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    
    # finetuning
    model.load_state_dict(torch.load('/home/marafath/scratch/saved_models/best_metric_model_d121_common_sized.pth'))

    # start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    epc = 300 # Number of epoch
    for epoch in range(epc):
        print('-' * 10)
        print('epoch {}/{}'.format(epoch + 1, epc))
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device=device, dtype=torch.int64)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print('{}/{}, train_loss: {:.4f}'.format(step, epoch_len, loss.item()))
            writer.add_scalar('train_loss', loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print('epoch {} average loss: {:.4f}'.format(epoch + 1, epoch_loss))

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                num_correct = 0.
                metric_count = 0
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    val_outputs = model(val_images)
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                    metric_count += len(value)
                    num_correct += value.sum().item()
                metric = num_correct / metric_count
                metric_values.append(metric)
                #torch.save(model.state_dict(), 'model_d121_epoch_{}.pth'.format(epoch + 1))
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), '/home/marafath/scratch/saved_models/best_metric_model_d121_common_sized.pth')
                    print('saved new best metric model')
                print('current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}'.format(
                    epoch + 1, metric, best_metric, best_metric_epoch))
                writer.add_scalar('val_accuracy', metric, epoch + 1)
    print('train completed, best_metric: {:.4f} at epoch: {}'.format(best_metric, best_metric_epoch))
    writer.close()

if __name__ == '__main__':
    main()
