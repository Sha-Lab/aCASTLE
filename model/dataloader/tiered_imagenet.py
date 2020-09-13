from __future__ import print_function

import os
import os.path as osp
import numpy as np
import pickle
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

# Set the appropriate paths of the datasets here.
_TIERED_IMAGENET_DATASET_DIR = osp.join(osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..')), 'data/tieredimagenet')
file_path = {'train':[osp.join(_TIERED_IMAGENET_DATASET_DIR, 'train_images.npz'), osp.join(_TIERED_IMAGENET_DATASET_DIR, 'train_labels.pkl')],
             'val':[osp.join(_TIERED_IMAGENET_DATASET_DIR, 'val_images.npz'), osp.join(_TIERED_IMAGENET_DATASET_DIR,'val_labels.pkl')],
             'test':[osp.join(_TIERED_IMAGENET_DATASET_DIR, 'test_images.npz'), osp.join(_TIERED_IMAGENET_DATASET_DIR,'test_labels.pkl')],
             'aux_val':[osp.join(_TIERED_IMAGENET_DATASET_DIR, 'aux_data.npz'), osp.join(_TIERED_IMAGENET_DATASET_DIR,'aux_labels.pkl')],
             'aux_test':[osp.join(_TIERED_IMAGENET_DATASET_DIR, 'aux_data.npz'), osp.join(_TIERED_IMAGENET_DATASET_DIR,'aux_labels.pkl')]}

def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


def load_data(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data
    
class tieredImageNet(data.Dataset):
    def __init__(self, setname, args, augment=False):
        image_path = file_path[setname][0]
        label_path = file_path[setname][1]

        data_train = load_data(label_path)
        labels = data_train['labels']
        self.data = np.load(image_path)['images']

        labelset = list(sorted(set(labels)))    
        label_index_map = {k:i for i,k in enumerate(labelset)}
        label = [label_index_map[e] for e in labels]
            
        self.label = label
        self.num_class = len(set(label))

        if augment and setname == 'train':
            transforms_list = [
                  transforms.RandomCrop(84, padding=8),
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                  transforms.ToTensor(),
                ]

        # Transformation
        if args.backbone_class == 'ConvNet':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif args.backbone_class == 'Res12':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])        
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')


    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]
        img = self.transform(Image.fromarray(np.uint8(img)))
        return img, label

    def __len__(self):
        return len(self.data)
