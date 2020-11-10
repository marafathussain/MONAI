# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import NiftiDataset
from monai.transforms import Compose, SpatialPad, AddChannel, ScaleIntensity, Resize, RandRotate90, RandZoom, ToTensor
import os

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    data_dir = '/home/marafath/scratch/eu_data'
    labels = np.load('eu_labels.npy')
    train_images = []
    train_labels = []

    val_images = []
    val_labels = []

    n_count = 0
    p_count = 0
    idx = 0
    for case in os.listdir(data_dir): 
        if p_count < 13 and labels[idx] == 1:
            val_images.append(os.path.join(data_dir,case,'image_masked.nii.gz'))
            val_labels.append(labels[idx])
            p_count += 1
            idx += 1
        elif n_count < 11 and labels[idx] == 0:    
            val_images.append(os.path.join(data_dir,case,'image_masked.nii.gz'))
            val_labels.append(labels[idx])
            n_count += 1
            idx += 1
        else:    
            train_images.append(os.path.join(data_dir,case,'image_masked.nii.gz'))
            train_labels.append(labels[idx])
            idx += 1

    # Define transforms
    train_transforms = Compose([
        ScaleIntensity(),
        AddChannel(),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        SpatialPad((256, 256, 92), mode='constant'),
        Resize((256, 256, 92)),
        ToTensor()
    ])
    
    val_transforms = Compose([
        ScaleIntensity(),
        AddChannel(),
        SpatialPad((256, 256, 92), mode='constant'),
        Resize((256, 256, 92)),
        ToTensor()
    ])

    # create a training data loader
    train_ds = NiftiDataset(image_files=train_images, labels=train_labels, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

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
    #model.load_state_dict(torch.load('best_metric_model_d121.pth'))

    # start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    epc = 100 # Number of epoch
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
                    torch.save(model.state_dict(), '/home/marafath/scratch/saved_models/best_metric_model_d121.pth')
                    print('saved new best metric model')
                print('current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}'.format(
                    epoch + 1, metric, best_metric, best_metric_epoch))
                writer.add_scalar('val_accuracy', metric, epoch + 1)
    print('train completed, best_metric: {:.4f} at epoch: {}'.format(best_metric, best_metric_epoch))
    writer.close()

if __name__ == '__main__':
    main()
