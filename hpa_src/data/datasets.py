import pandas as pd
import numpy as np
from skimage import io
from torch.utils.data import Dataset
from hpa_src.config import get_data_dir

__all__ = ['readimg', 'HpaDataset', 'TestDataset']

DATADIR = get_data_dir()

CHANELS = ['_yellow', '_red', '_green', '_blue']
def readimg(imgid, 
            datadir=DATADIR+'raw/png/train/', 
            suffix='.png', 
            rgb=True, 
            stack=True):
    imgs = [io.imread(datadir + imgid + cl + suffix) for cl in CHANELS]
    if rgb:
        imgs[1] += (imgs[0]/2).astype(np.uint8)
        imgs[2] += (imgs[0]/2).astype(np.uint8)
        img = np.stack(imgs[1:], -1)
        return img
    else:
        if stack:
            img = np.stack(imgs, -1)
            return img
        return imgs

class HpaDataset(Dataset):
    '''
    Args:
        dt: csv file describing the data
        transform (callable, optional): Optional transform to be applied
                on a sample.
    '''
    def __init__(self, dt, transform=None):
        imgs = pd.read_csv(dt)
        imgs['target_list'] = imgs['Target'].map(lambda x: [int(a) for a in x.split(' ')])
        self.ids = imgs.Id
        self.targets = imgs.target_list
        self.transform = transform
        self.length = imgs.shape[0]
        self.num_inputs = self.length
        self.num_targets = self.length

    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        labels = np.zeros(28, dtype=np.float32)
        labels[self.targets[idx]] = 1
        #labels = np.array(self.targets[idx], dtype=np.int64)
        img = readimg(self.ids[idx])
        #img = img.permute(2,0,1)
        if self.transform:
            img = self.transform(img)
        return img, labels

    
class TestDataset(Dataset):
    def __init__(self, dt, transform=None):
        imgs = pd.read_csv(dt)
        self.ids = imgs.Id
        self.transform = transform
        self.length = imgs.shape[0]
        self.num_inputs = self.length
        self.num_targets = self.length

    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        img = readimg(self.ids[idx], datadir=DATADIR+'raw/png/test/')
        if self.transform:
            img = self.transform(img)
        return img
 

# Train, Validation split
from torch.utils.data.sampler import SubsetRandomSampler
#INDICES = list(range(pd.read_csv(DATADIR + "raw/png/train.csv").shape[0]))

def train_val_split(datasize, validation_split=0.2):
    """ Split a data of size `datasize`, return sampler object
    Args:
        datasize: int, number of data points
    """
    indices = list(range(datasize))
    np.random.shuffle(indices)
    num_val = int(validation_split * len(indices))
    train_idx, valid_idx = indices[num_val:], indices[:num_val]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)
    return train_sampler, val_sampler
