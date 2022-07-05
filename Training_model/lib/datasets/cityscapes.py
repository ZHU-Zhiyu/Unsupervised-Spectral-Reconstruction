# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

import scipy.io as scio
from .base_dataset import BaseDataset
import h5py
import random
import lmdb
import pickle

class Cityscapes(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=19,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=2048, 
                 crop_size=(512, 1024), 
                 center_crop_test=False,
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):

        super(Cityscapes, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std,)

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

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
                                        1.0166, 0.9969, 0.9754, 1.0489,
                                        0.8786, 1.0023, 0.9539, 0.9843, 
                                        1.1116, 0.9037, 1.0865, 1.0955, 
                                        1.0865, 1.1529, 1.0507]).cuda()

        self.multi_scale = multi_scale
        self.flip = flip
        self.center_crop_test = center_crop_test
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

        self.label_mapping = {-1: ignore_label, 0: ignore_label, 
                              1: ignore_label, 2: ignore_label, 
                              3: ignore_label, 4: ignore_label, 
                              5: ignore_label, 6: ignore_label, 
                              7: 0, 8: 1, 9: ignore_label, 
                              10: ignore_label, 11: 2, 12: 3, 
                              13: 4, 14: ignore_label, 15: ignore_label, 
                              16: ignore_label, 17: 5, 18: ignore_label, 
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15, 
                              29: ignore_label, 30: ignore_label, 
                              31: 16, 32: 17, 33: 18}
        # data = lmdb.Environment('/home/zzy/data/')
        # self.file_txn = data.begin()
        # file_name = [keys for keys,value in self.file_txn.cursor()]
        # self.train_name = file_name[:-20]
        # self.test_name = file_name[-20:]
        file_name = scio.loadmat('/home/zzy/data/icvl_name.mat')['name']
        self.train_name = file_name[:-20]
        self.test_name = file_name[-20:]
    
    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })
        return files
        
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root,'cityscapes',item["img"]),
                           cv2.IMREAD_COLOR)
        size = image.shape

        if 'test' in self.list_path:
            # msi = image.copy().astype(np.float)/255
            image, msi = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            msi = msi.transpose((2, 0, 1))

            return image.copy(), np.array(size), name, msi.copy()

        label = cv2.imread(os.path.join(self.root,'cityscapes',item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)

        image, label, msi = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, 
                                self.center_crop_test)
        index_img = index % len(self.train_name)
        # file=h5py.File('/home/zzy/memory/data.h5','r')

        # hsi_g = file[self.train_name[index_img]][:]
        # hsi_g = self.file_txn.get(self.train_name[index_img])
        # hsi_g = pickle.loads(hsi_g)
        
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

        return image.copy(), label.copy(), np.array(size), name, msi.copy(), hsi_g

    def multi_scale_inference(self, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]
                
            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w - 
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w]).cuda()
                count = torch.zeros([1,1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(model, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]
            preds = F.upsample(preds, (ori_height, ori_width), 
                                   mode='bilinear')
            final_pred += preds
        return final_pred

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
        
