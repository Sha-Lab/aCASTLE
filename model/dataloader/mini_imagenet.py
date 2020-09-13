import torch
import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
CACHE_PATH = osp.join(ROOT_PATH, '.cache/')
IMAGE_PATH1 = osp.join(osp.abspath(osp.join(THIS_PATH, '..', '..', '..')), 'data/miniimagenet/images')
IMAGE_PATH2 = osp.join(osp.abspath(osp.join(THIS_PATH, '..', '..', '..')), 'data/miniimagenetaux/images')
SPLIT_PATH = osp.join(ROOT_PATH, 'data/miniimagenet/split')
split_map = {'train':IMAGE_PATH1, 'val':IMAGE_PATH1, 'test':IMAGE_PATH1, 'aux_val':IMAGE_PATH2, 'aux_test':IMAGE_PATH2}

def identity(x):
    return x

class MiniImageNet(Dataset):
    """ Usage:
    """
    def __init__(self, setname, args, augment=False):
        im_size = args.orig_imsize
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        cache_path = osp.join( CACHE_PATH, "{}.{}.{}.pt".format(self.__class__.__name__, setname, im_size) )

        self.use_im_cache = ( im_size != -1 ) # not using cache
        if self.use_im_cache:
            if not osp.exists(cache_path):
                print('* Cache miss... Preprocessing {}...'.format(setname))
                resize_ = identity if im_size < 0 else transforms.Resize(im_size)
                data, label = self.parse_csv(csv_path, setname)
                self.data = [ resize_(Image.open(path).convert('RGB')) for path in data ]
                self.label = label
                print('* Dump cache from {}'.format(cache_path))
                torch.save({'data': self.data, 'label': self.label }, cache_path)
            else:
                print('* Load cache from {}'.format(cache_path))
                cache = torch.load(cache_path)
                self.data  = cache['data']
                self.label = cache['label']
        else:
            self.data, self.label = self.parse_csv(csv_path, setname)

        self.num_class = len(set(self.label))

        image_size = 84
        if augment and setname == 'train':
            transforms_list = [
                  transforms.RandomResizedCrop(image_size),
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                  transforms.Resize(256),
                  transforms.CenterCrop(224),
                  transforms.Resize(image_size),
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

    def parse_csv(self, csv_path, setname):
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        wnids = []
        
        for l in tqdm(lines, ncols=64):
            name, wnid = l.split(',')
            path = osp.join(split_map[setname], name)
            data.append( path )
            wnids.append( wnid )
        
        labelset = list(sorted(set(wnids)))    
        label_index_map = {k:i for i,k in enumerate(labelset)}
        label = [label_index_map[e] for e in wnids]

        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        if self.use_im_cache:
            image = self.transform(data)
        else:
            image = self.transform(Image.open(data).convert('RGB'))
        
        return image, label

