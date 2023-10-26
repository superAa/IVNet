
import os
import torch
from Dataset.edge_utils import mask_to_onehot,onehot_to_binary_edges
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
from Dataset.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale, \
    RandomRotation


class MF_dataset0(Dataset):

    def __init__(self, data_dir, split, have_label, input_h=480, input_w=640):

        super(MF_dataset0, self).__init__()

        assert split in ['train_1', 'val', 'test', 'test_day', 'test_night', 'val_test'], 'split must be "train"|"val"|"test"|"test_day"|"test_night"|"val_test"'  # test_day, test_night

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir  = data_dir
        self.split     = split
        self.input_h   = input_h
        self.input_w   = input_w

        self.is_train  = have_label
        self.n_data    = len(self.names)

        self.aug = Compose([
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5),
            RandomHorizontalFlip(0.5),
            RandomScale((0.5,2.0)),
            RandomCrop((480,640), pad_if_needed=True)
        ])

    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, name))
        image     = np.asarray(Image.open(file_path)) # (w,h,c)
        return image

    def get_train_item(self, index):

        name  = self.names[index]
        images = self.read_image(name, 'images')
        label1 = self.read_image(name, 'labels')
        Rgb = images[:,:,:3]
        ir  = images[:,:,3:]
        Themral = ir.repeat(3, axis=2)


        _edgemap0 = mask_to_onehot(label1, 9)
        _Edgemap0 = onehot_to_binary_edges(_edgemap0, 1, 9)
        _Edgemap1 = onehot_to_binary_edges(_edgemap0, 2, 9)

        Rgb = Image.fromarray(np.array(Rgb,dtype=np.uint8))
        Themral = Image.fromarray(np.array(Themral, dtype=np.uint8))
        label1 = Image.fromarray(np.array(label1, dtype=np.uint8))
        _Edgemap1 = Image.fromarray(np.array(_Edgemap1, dtype=np.uint8))
        _Edgemap0 = Image.fromarray(np.array(_Edgemap0, dtype=np.uint8))

        sample = {
            'image': Rgb,
            'depth': Themral,
            'label': label1,
            '_Edgemap1': _Edgemap1,
            '_Edgemap0': _Edgemap0,
        }

        sample = self.aug(sample)

        Rgb     = torch.from_numpy(np.asarray(sample['image'],dtype=np.float32).transpose( 2,0,1)) / 255.
        Themral = torch.from_numpy(np.asarray(sample['depth'],dtype=np.float32).transpose( 2,0,1)) / 255.
        label1  = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64))

        Edge1 = torch.from_numpy(np.asarray(sample['_Edgemap1'], dtype=np.float32))
        Edge1  = Edge1.unsqueeze(0)

        Edge0 = torch.from_numpy(np.asarray(sample['_Edgemap0'], dtype=np.float32))

        return Rgb,Themral,label1,Edge1,Edge0


    def get_test_item(self, index):

        name  = self.names[index]
        images = self.read_image(name, 'images')
        label1 = self.read_image(name, 'labels')

        Rgb = images[:,:,:3]
        ir  = images[:,:,3:]
        Themral = ir.repeat(3, axis=2)

        _edgemap0 = mask_to_onehot(label1, 9)
        _Edgemap0 = onehot_to_binary_edges(_edgemap0, 2, 9)
        _Edgemap1 = onehot_to_binary_edges(_edgemap0, 4, 9)

        Rgb = Image.fromarray(np.array(Rgb,dtype=np.uint8))
        Themral = Image.fromarray(np.array(Themral, dtype=np.uint8))
        label1 = Image.fromarray(np.array(label1, dtype=np.uint8))
        _Edgemap1 = Image.fromarray(np.array(_Edgemap1, dtype=np.uint8))
        _Edgemap0 = Image.fromarray(np.array(_Edgemap0, dtype=np.uint8))

        Rgb     = torch.from_numpy(np.asarray(Rgb,dtype=np.float32).transpose( 2,0,1)) / 255.
        Themral = torch.from_numpy(np.asarray(Themral,dtype=np.float32).transpose( 2,0,1)) / 255.
        label1 = torch.from_numpy(np.asarray(label1, dtype=np.int64))
        Edge1 = torch.from_numpy(np.asarray(_Edgemap1, dtype=np.float32))
        Edge1  = Edge1.unsqueeze(0)

        Edge0 = torch.from_numpy(np.asarray(_Edgemap0, dtype=np.float32))

        return Rgb,Themral,label1,Edge1,Edge0


    def __getitem__(self, index):

        if self.is_train == 'True':
            return self.get_train_item(index)
        else:
            return self.get_test_item (index)

    def __len__(self):
        return self.n_data
