import os
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm

import torch
from .base import BaseDataset

class CityscapesSegmentation(BaseDataset):
    NUM_CLASS = 19
    #BASE_DIR = 'VOCdevkit/VOC2012'
    BASE_DIR = 'Cityscapes/data'
    def __init__(self, root=os.path.expanduser('~/.encoding/data'), split='train', mode=None, transform=None,
                 target_transform=None):
        super(CityscapesSegmentation, self).__init__(root, split, mode, transform, target_transform,base_size=1024, crop_size=480)
        _cityscapes_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_cityscapes_root, 'gtFine')
        _image_dir = os.path.join(_cityscapes_root, 'leftImg8bit')
        # train/val/test splits are pre-cut
        #_splits_dir = os.path.join(_cityscapes_root, 'ImageSets/Segmentation')
        if self.mode == 'train':
            _split_f = os.path.join(_cityscapes_root, 'train.txt')
        elif self.mode == 'val':
            _split_f = os.path.join(_cityscapes_root, 'val.txt')
        elif self.mode == 'test':
            _split_f = os.path.join(_cityscapes_root, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')
        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in tqdm(lines):
                _image = os.path.join(_image_dir, self.split + '/' + line.rstrip('\n'))
                assert os.path.isfile(_image)
                self.images.append(_image)
                if self.mode != 'test':
                    _mask = os.path.join(_mask_dir, self.split + '/' + line.rstrip('\n')[:-15] + "gtFine_labelIds.png")
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        if self.mode != 'test':
            assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        target = Image.open(self.masks[index])
        #img = img.resize((1024, 512), Image.BILINEAR)
        #target = target.resize((1024, 512), Image.NEAREST)
        # synchrosized transform
        if self.mode == 'train':
            img, target = self._sync_transform( img, target)
        elif self.mode == 'val':
            img, target = self._val_sync_transform( img, target)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            #print("transform for input")
            img = self.transform(img)
        if self.target_transform is not None:
            #print("transform for label")
            target = self.target_transform(target)
        return img, target

    def label_mapping(self, input, mapping):
        output = np.copy(input)
        for ind in range(len(mapping)):
            output[input == mapping[ind][0]] = mapping[ind][1]
        return np.array(output, dtype=np.int32)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        label2train=[[0, 255],[1, 255],[2, 255],[3, 255],[4, 255],[5, 255],[6, 255],[7, 0],[8, 1],[9, 255],[10, 255],[11, 2],[12, 3],
                     [13, 4],[14, 255],[15, 255],[16, 255],[17, 5],[18, 255],[19, 6],[20, 7],[21, 8],[22, 9],[23, 10],[24, 11],[25, 12],
                     [26, 13],[27, 14],[28, 15],[29, 255],[30, 255],[31, 16],[32, 17],[33, 18],[-1, 255]]
        mapping = np.array(label2train, dtype=np.int)
        target = self.label_mapping(target, mapping)

        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)
