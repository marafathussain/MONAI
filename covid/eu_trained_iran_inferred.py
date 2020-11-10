import sys
import logging
import numpy as np
import os
import torch
from torch.utils.data import DataLoader

import monai
from monai.data import NiftiDataset, CSVSaver
from monai.transforms import Compose, AddChannel, ScaleIntensity, Resize, ToTensor, SpatialPad
from sklearn.metrics import classification_report


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    images = []
    labels = []
    class_names = ['Healthy', 'Pneumonia']

    data_dir = '/home/marafath/scratch/iran_organized_data2'
    for patient in os.listdir(data_dir):
        label = int(patient[-1])
        for series in os.listdir(os.path.join(data_dir,patient)):
            images.append(os.path.join(data_dir,patient,series,'masked_image.nii.gz'))
            labels.append(label)

    labels = np.asarray(labels,np.int64)

    # Define transforms for image
    val_transforms = Compose([
        ScaleIntensity(),
        AddChannel(),
        SpatialPad((256, 256, 92), mode='constant'),
        Resize((256, 256, 92)),
        ToTensor()
    ])

    # create a validation data loader
    val_ds = NiftiDataset(image_files=images, labels=labels, transform=val_transforms, image_only=False)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=2, pin_memory=torch.cuda.is_available())

    # Create DenseNet121
    device = torch.device('cuda:0')
    model = monai.networks.nets.densenet.densenet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
    ).to(device)

    model.load_state_dict(torch.load("/home/marafath/scratch/saved_models/best_metric_model_d121.pth"))
    model.eval()
    
    y_true = list()
    y_pred = list()
    
    with torch.no_grad():
        num_correct = 0.
        metric_count = 0
        saver = CSVSaver(output_dir='./output')
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            val_outputs = model(val_images).argmax(dim=1)
            value = torch.eq(val_outputs, val_labels)
            metric_count += len(value)
            num_correct += value.sum().item()
            saver.save_batch(val_outputs, val_data[2])
            
            for i in range(len(val_outputs)):
                y_true.append(val_labels[i].item())
                y_pred.append(val_outputs[i].item())
            
        metric = num_correct / metric_count
        print('evaluation metric:', metric)
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
        saver.finalize()


if __name__ == "__main__":
    main()