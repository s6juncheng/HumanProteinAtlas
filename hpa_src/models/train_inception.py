#-----------------
# Parameters 
bz = 16
input_size = 512
lr = 0.001
criterion = FocalLoss(gamma=2, logits=True)

#-------------
from hpa_src.config import get_data_dir, name_label_dict
from hpa_src.data.datasets import readimg, HpaDataset, TestDataset#, train_val_split
import pandas as pd
import numpy as np

DATA = get_data_dir()

# Split train eval
image_df = pd.read_csv(DATA + "raw/png/train.csv")
image_df['target_list'] = image_df['Target'].map(lambda x: [int(a) for a in x.split(' ')])
import itertools
all_labels = np.array(list(itertools.chain(*image_df.target_list.values)))
class_n = np.unique(all_labels, return_counts=True)[1]
alpha = class_n / sum(class_n)
image_df = image_df.drop(['target_list'], axis=1)

# Split train eval
# This code only run once
train_indx, val_indx = train_test_split(np.arange(image_df.shape[0]), test_size=0.2)

training = image_df.iloc[train_indx].reset_index(drop=True)
validation = image_df.iloc[val_indx].reset_index(drop=True)

training.to_csv(DATA+'raw/png/training.csv')
validation.to_csv(DATA+'raw/png/validation.csv')

#------------
# Build model
#------------
from torch.utils.data import DataLoader, random_split
from PIL import Image
from torchvision import transforms
from hpa_src.data.transforms import ToPIL
from torch.nn import BCEWithLogitsLoss
from hpa_src.models.loss import FocalLoss
import torch
import torch.nn as nn

import torch.optim as optim
from torch.optim import lr_scheduler

# input_size = 299
train_transform = transforms.Compose([
    ToPIL(),
    transforms.RandomResizedCrop(input_size, scale=(0.5,1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
#     transforms.Normalize((0.1149, 0.0922, 0.0553),
#                          (0.1694, 0.1381, 0.1551))
    transforms.Normalize((0.08069, 0.05258, 0.05487, 0.08282),
                         (0.13704, 0.10145, 0.15313, 0.13814))
])
val_transform = transforms.Compose([
    ToPIL(),
#     transforms.Resize(input_size),
    transforms.ToTensor(),
#     transforms.Normalize((0.1149, 0.0922, 0.0553),
#                          (0.1694, 0.1381, 0.1551))
    transforms.Normalize((0.08069, 0.05258, 0.05487, 0.08282),
                         (0.13704, 0.10145, 0.15313, 0.13814))
])

test_transform = transforms.Compose([
    ToPIL(),
#     transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize((0.05913, 0.0454 , 0.04066, 0.05928),
                         (0.11734, 0.09503, 0.129 , 0.11528))
])

train_dataset = HpaDataset(DATA + 'raw/png/training.csv', transform=train_transform)
val_dataset = HpaDataset(DATA + 'raw/png/validation.csv', transform=val_transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=bz, #sampler=train_sampler,
    num_workers=bz
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=bz, #sampler=val_sampler,
    num_workers=bz
)

dataloaders = {'train': train_loader, 'val': val_loader}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model
from hpa_src.models.inception import inceptionresnetv2
model = inceptionresnetv2(pretrained='imagenet')
model = nn.DataParallel(model, device_ids=[0,1])

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)