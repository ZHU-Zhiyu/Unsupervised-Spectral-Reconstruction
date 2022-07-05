# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# Referring to the implementation in 
# https://github.com/zhanghang1989/PyTorch-Encoding
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch

from .base_dataset import BaseDataset
import scipy.io as scio
import h5py
import random
import lmdb
import pickle

class PASCALContext(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=59,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=520, 
                 crop_size=(480, 480), 
                 downsample_rate=1,
                 scale_factor=16,
                 center_crop_test=False,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],):
    
        super(PASCALContext, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std)
        
        np_load_old = np.load
        temp_load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        # temp = '/public/zhuzhiyu/segmentation/name.npy'
        temp = '/home/zzy/name.npy'
        name = temp_load(temp)
        name = name.item()
        self.train_name = name['train']
        self.test_name = name['test']
        self.num_width = int(1040/128)
        self.num_hight = int(1390/128)
        self.test_len = 20

        self.max_v = 0.06163118944075685

        # self.reps = scio.loadmat('/public/zhuzhiyu/segmentation/resp0.mat')['resp']
        self.reps = scio.loadmat('/home/zzy/comparing_method/data/resp.mat')['resp']
        self.reps = np.transpose(self.reps,(1,0))

        self.root = os.path.join(root, 'pascal_ctx/VOCdevkit/VOC2010')
        self.split = list_path

        self.num_classes = num_classes
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        self.crop_size = crop_size

        # prepare data
        annots = os.path.join(self.root, 'trainval_merged.json')
        img_path = os.path.join(self.root, 'JPEGImages')
        from detail import Detail
        if 'val' in self.split:
            self.detail = Detail(annots, img_path, 'val')
            mask_file = os.path.join(self.root, 'val.pth')
        elif 'train' in self.split:
            self.mode = 'train'
            self.detail = Detail(annots, img_path, 'train')
            mask_file = os.path.join(self.root, 'train.pth')
        else:
            raise NotImplementedError('only supporting train and val set.')
        self.files = self.detail.getImgs()

        # generate masks
        self._mapping = np.sort(np.array([
            0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22, 
            23, 397, 25, 284, 158, 159, 416, 33, 162, 420, 454, 295, 296, 
            427, 44, 45, 46, 308, 59, 440, 445, 31, 232, 65, 354, 424, 
            68, 326, 72, 458, 34, 207, 80, 355, 85, 347, 220, 349, 360, 
            98, 187, 104, 105, 366, 189, 368, 113, 115]))
            
        self._key = np.array(range(len(self._mapping))).astype('uint8')

        print('mask_file:', mask_file)
        if os.path.exists(mask_file):
            self.masks = torch.load(mask_file)
        else:
            self.masks = self._preprocess(mask_file)
            
        file_name = scio.loadmat('/home/zzy/data/icvl_name.mat')['name']
        self.train_name = file_name[:-20]
        self.test_name = file_name[-20:]

    def _class_to_index(self, mask):
        # assert the values
        values = np.unique(mask)
        for i in range(len(values)):
            assert(values[i] in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def _preprocess(self, mask_file):
        masks = {}
        print("Preprocessing mask, this will take a while." + \
            "But don't worry, it only run once for each split.")
        for i in range(len(self.files)):
            img_id = self.files[i]
            mask = Image.fromarray(self._class_to_index(
                self.detail.getMask(img_id)))
            masks[img_id['image_id']] = mask
        torch.save(masks, mask_file)
        return masks

    def __getitem__(self, index):
        item = self.files[index]
        name = item['file_name']
        img_id = item['image_id']

        image = cv2.imread(os.path.join(self.detail.img_folder,name),
                           cv2.IMREAD_COLOR)
        label = np.asarray(self.masks[img_id],dtype=np.int)
        size = image.shape

        if self.split == 'val':
            image = cv2.resize(image, self.crop_size, 
                               interpolation = cv2.INTER_LINEAR)
            # image = self.input_transform(image)
            image, msi = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            msi = msi.transpose((2, 0, 1))


            label = cv2.resize(label, self.crop_size, 
                               interpolation=cv2.INTER_NEAREST)
            label = self.label_transform(label)
        elif self.split == 'testval':
            # evaluate model on val dataset
            # image = self.input_transform(image)
            image, msi = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            msi = msi.transpose((2, 0, 1))

            label = self.label_transform(label)
        else:
            image, label, msi = self.gen_sample(image, label, 
                                self.multi_scale, self.flip)
        
        index_img = index % len(self.train_name)

        file = h5py.File('/home/zzy/memory/icvl.h5','r')
        hsi_g = file[self.train_name[index_img]][:]
        fac_x = np.random.uniform(low=1/7, high=0.5, size=2)
        fac_y = np.random.uniform(low=1/7, high=0.5, size=2)
        hsi_g = cv2.resize(hsi_g.astype(np.float32), (0,0), fx=fac_x[0], fy=fac_y[0])

        [h,w,c] = hsi_g.shape
        h1 = random.randint(0,h-128)
        w1 = random.randint(0,w-128)
        hsi_g = hsi_g[h1:h1+128,w1:w1+128,:]/self.max_v

        # rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        a = random.randint(1, 800)

        # Random vertical Flip   
        for j in range(vFlip):
            hsi_g = np.flip(hsi_g,axis=1)

        # Random Horizontal Flip
        for j in range(hFlip):
            hsi_g = np.flip(hsi_g,axis=0)

        hsi_g = np.transpose(hsi_g,(2,0,1)).copy()


        # return image.copy(), label.copy(), np.array(size), name
        return image.copy(), label.copy(), np.array(size), name, msi.copy(), hsi_g

    def label_transform(self, label):
        if self.num_classes == 59:
            # background is ignored
            label = np.array(label).astype('int32') - 1
            label[label==-2] = -1
        else:
            label = np.array(label).astype('int32')
        return label
