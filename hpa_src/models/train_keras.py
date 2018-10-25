from hpa_src.config import get_data_dir, name_label_dict
from hpa_src.data.datasets import readimg, HpaDataset, TestDataset, train_val_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA = get_data_dir()

from torch.utils.data import DataLoader, random_split
from PIL import Image
from torchvision import transforms
from hpa_src.data.transforms import ToPIL, ToNumpy
from hpa_src.data.functional import iterable_cycle
import torch

image_df = pd.read_csv(DATA + "raw/png/train.csv")

input_size = 299
train_transform = transforms.Compose([
    ToPIL(),
    #transforms.Resize(input_size),
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize((0.1149, 0.0922, 0.0553),
                         (0.1694, 0.1381, 0.1551)),
    ToNumpy()
])
val_transform = transforms.Compose([
    ToPIL(),
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize((0.1149, 0.0922, 0.0553),
                         (0.1694, 0.1381, 0.1551)),
    ToNumpy()
])

train_sampler, val_sampler = train_val_split(image_df.shape[0])
train_dataset = HpaDataset(DATA + 'raw/png/train.csv', transform=train_transform)
val_dataset = HpaDataset(DATA + 'raw/png/train.csv', transform=val_transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, sampler=train_sampler,
    num_workers=16
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=16, sampler=val_sampler,
    num_workers=16
)

dataloaders = {'train': iterable_cycle(train_loader), 'val': iterable_cycle(val_loader)}

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Input, Conv2D
from keras.models import load_model, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score
from hpa_src.models.metrics import f1
from keras.optimizers import Adam 

epochs = 10; batch_size = 16
checkpoint = ModelCheckpoint(DATA+'../models/InceptionResNetV2.h5', monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min')
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, 
                                   verbose=1, mode='auto', min_delta=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=5)
callbacks_list = [checkpoint, early, reduceLROnPlat]

def create_model(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    base_model = InceptionResNetV2(include_top=False,
                   weights='imagenet',
                   input_shape=input_shape)
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = Conv2D(32, kernel_size=(1,1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    
    return model

# Define model and train
model = create_model(
    input_shape=(input_size,input_size,3), 
    n_out=28)

for layer in model.layers:
    layer.trainable = False
model.layers[-1].trainable = True
model.layers[-2].trainable = True
model.layers[-3].trainable = True
model.layers[-4].trainable = True
model.layers[-5].trainable = True
model.layers[-6].trainable = True

# First train added layers
model.compile(
    loss='binary_crossentropy', 
    optimizer=Adam(1e-03),
    metrics=[f1])

model.fit_generator(
    dataloaders['train'],
    steps_per_epoch=np.ceil(float(len(train_sampler.indices)) / float(batch_size)),
    validation_data=dataloaders['val'],
    validation_steps=np.ceil(float(len(val_sampler.indices)) / float(batch_size)),
    epochs=3,
    verbose=1)

# Second train everything
for layer in model.layers:
    layer.trainable = True
model.compile(loss='binary_crossentropy',
            optimizer=Adam(lr=1e-4),
            metrics=[f1])
model.fit_generator(
    dataloaders['train'],
    steps_per_epoch=np.ceil(float(len(train_sampler.indices)) / float(batch_size)),
    validation_data=dataloaders['val'],
    validation_steps=np.ceil(float(len(val_sampler.indices)) / float(batch_size)),
    epochs=100, 
    verbose=1,
    callbacks=callbacks_list)