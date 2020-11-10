from collections import OrderedDict
from typing import Callable, Sequence

import torch
import torch.nn as nn

from monai.networks.layers.factories import Conv, Dropout, Pool, Norm


class _DenseLayer(nn.Sequential):
    def __init__(
        self, spatial_dims: int, in_channels: int, growth_rate: int, bn_size: int, dropout_prob: float
    ) -> None:
        super(_DenseLayer, self).__init__()

        out_channels = bn_size * growth_rate
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]
        dropout_type: Callable = Dropout[Dropout.DROPOUT, spatial_dims]

        self.add_module("norm1", norm_type(in_channels))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", conv_type(in_channels, out_channels, kernel_size=1, bias=False))

        self.add_module("norm2", norm_type(out_channels))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", conv_type(out_channels, growth_rate, kernel_size=3, padding=1, bias=False))

        if dropout_prob > 0:
            self.add_module("dropout", dropout_type(dropout_prob))

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(
        self, spatial_dims: int, layers: int, in_channels: int, bn_size: int, growth_rate: int, dropout_prob: float
    ) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(layers):
            layer = _DenseLayer(spatial_dims, in_channels, growth_rate, bn_size, dropout_prob)
            in_channels += growth_rate
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int) -> None:
        super(_Transition, self).__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]
        pool_type: Callable = Pool[Pool.AVG, spatial_dims]

        self.add_module("norm", norm_type(in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", conv_type(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module("pool", pool_type(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """
    Densenet based on: "Densely Connected Convolutional Networks" https://arxiv.org/pdf/1608.06993.pdf
    Adapted from PyTorch Hub 2D version:
    https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        init_features: number of filters in the first convolution layer.
        growth_rate: how many filters to add each layer (k in paper).
        block_config: how many layers in each pooling block.
        bn_size: multiplicative factor for number of bottle neck layers.
                      (i.e. bn_size * k features in the bottleneck layer)
        dropout_prob: dropout rate after each dense layer.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        bn_size: int = 4,
        dropout_prob: float = 0.0,
    ) -> None:

        super(DenseNet, self).__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]
        pool_type: Callable = Pool[Pool.MAX, spatial_dims]
        avg_pool_type: Callable = Pool[Pool.ADAPTIVEAVG, spatial_dims]

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", conv_type(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", norm_type(init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", pool_type(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        in_channels = init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                spatial_dims=spatial_dims,
                layers=num_layers,
                in_channels=in_channels,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_prob=dropout_prob,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            in_channels += num_layers * growth_rate
            if i == len(block_config) - 1:
                self.features.add_module("norm5", norm_type(in_channels))
            else:
                _out_channels = in_channels // 2
                trans = _Transition(spatial_dims, in_channels=in_channels, out_channels=_out_channels)
                self.features.add_module("transition%d" % (i + 1), trans)
                in_channels = _out_channels

        # pooling and classification
        '''
        self.class_layers = nn.Sequential(
            OrderedDict(
                [
                    ("relu", nn.ReLU(inplace=True)),
                    ("norm", avg_pool_type(1)),
                    ("flatten", nn.Flatten(1)), 
                    ("class", nn.Linear(in_channels, out_channels)),
                ]
            )
        )
        '''
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(1024*4*4*4, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, out_channels)

        # Avoid Built-in function isinstance was called with the wrong arguments warning
        # pytype: disable=wrong-arg-types
        for m in self.modules():
            if isinstance(m, conv_type):  # type: ignore
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, norm_type):  # type: ignore
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        # pytype: enable=wrong-arg-types

    def forward(self, x):
        x = self.features(x)
        #print(x.shape)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def densenet121(**kwargs) -> DenseNet:
    model = DenseNet(init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
    return model


def densenet169(**kwargs) -> DenseNet:
    model = DenseNet(init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)
    return model


def densenet201(**kwargs) -> DenseNet:
    model = DenseNet(init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)
    return model


def densenet264(**kwargs) -> DenseNet:
    model = DenseNet(init_features=64, growth_rate=32, block_config=(6, 12, 64, 48), **kwargs)
    return model



import sys
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import NiftiDataset
from monai.transforms import Compose, SpatialPad, AddChannel, ScaleIntensity, Resize, RandRotate90, RandRotate, RandZoom, ToTensor
import os

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print("Densenet121 with two FCL of size 1024 on only resized Infection")

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
                images_p.append(os.path.join(data_dir,patient,series,'cropped_and_resized_infection.nii.gz'))
        else:
            non_covid_pat += 1
            for series in os.listdir(os.path.join(data_dir,patient)):
                labels_n.append(0)
                images_n.append(os.path.join(data_dir,patient,series,'cropped_and_resized_infection.nii.gz'))
            
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
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        #SpatialPad((128, 128, 92), mode='constant'),
        #Resize((128, 128, 92)),
        ToTensor()
    ])
    
    val_transforms = Compose([
        ScaleIntensity(),
        AddChannel(),
        #SpatialPad((128, 128, 92), mode='constant'),
        #Resize((128, 128, 92)),
        ToTensor()
    ])

    # create a training data loader
    train_ds = NiftiDataset(image_files=train_images, labels=train_labels, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    # create a validation data loader
    val_ds = NiftiDataset(image_files=val_images, labels=val_labels, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=2, pin_memory=torch.cuda.is_available())
    
    device = torch.device('cuda:0')
    model = densenet121(
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
                    torch.save(model.state_dict(),'/home/marafath/scratch/saved_models/best_no_avgpool_2fc_infection.pth')
                    print('saved new best metric model')
                print('current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}'.format(
                    epoch + 1, metric, best_metric, best_metric_epoch))
                writer.add_scalar('val_accuracy', metric, epoch + 1)
    print('train completed, best_metric: {:.4f} at epoch: {}'.format(best_metric, best_metric_epoch))
    writer.close()

if __name__ == '__main__':
    main()