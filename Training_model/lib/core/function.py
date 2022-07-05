# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate
from utils.utils import get_world_size, get_rank
import skimage.measure as skm
import scipy.io as scio
import h5py
import lmdb
import pickle

def cal_SAM(output,GT):
    [B,C,H,W] = output.shape
    output = output.reshape(B,C,H*W)
    GT = GT.reshape(B,C,H*W)

    product = output*GT
    product = product.sum(1)
    len1 = output * output
    len1 = len1.sum(1)
    len2 = GT * GT
    len2 = len2.sum(1)
    len1 = len1.sqrt()
    len2 = len2.sqrt()

    angular = product/(len1*len2+1e-6)

    angular = torch.acos(angular)*180/3.1415926

    return angular.mean().detach().cpu().numpy()
def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp

def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
         trainloader, optimizer, model, writer_dict, device,Logger=None):
    
    # Training
    model.train()
    # model.HRNet.eval()
    # model.hsiG.eval()
    # model.hsiG.eval()
    # model.respG.eval()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    rank = get_rank()
    world_size = get_world_size()
    loss_g = []
    loss_d = []
    GP_List = []
    smooth_List = []
    res_List = []
    for i_iter, batch in enumerate(trainloader):
        images, labels, _, _, MSI, HSI = batch
        # images, labels, _, _,MSI = batch
        model.zero_grad()
        images = images.to(device).float()
        MSI = MSI.to(device).float()
        labels = labels.long().to(device)
        HSI = HSI.to(device).float()
        # loss,_,_ = model(images,MSI,labels)
        loss_d_,GP_loss = model.update_discriminator(MSI = MSI, HSI = HSI, rank = 0)
        loss_g_, smooth_loss, res_loss, gen_loss = model.update_generator(MSI = MSI, HSI = HSI, img = images, rank = 0, seg_label=labels)
        # loss = np.mean(np.array(loss_g_))
        
        loss_g.append(loss_g_.item() - gen_loss.item())
        loss_d.append(loss_d_)
        GP_List.append(GP_loss)
        smooth_List.append(smooth_loss)
        res_List.append(res_loss)

        loss_g_.backward()
        model.gen_optimizer.step()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss_g_.item() - gen_loss.item())

        # gen_lr = adjust_learning_rate(model.gen_optimizer,
        #                           1e-4,
        #                           num_iters,
        #                           0)

        dis_lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            print_loss = ave_loss.average() / world_size
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f} Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters, 
                      batch_time.average(), dis_lr, print_loss)
            logging.info(msg)
            
            writer.add_scalar('train_loss', print_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1
            # break
    Logger['Lossg_logger'].log(epoch,np.mean(np.array(loss_g)))
    Logger['Lossd_logger'].log(epoch,np.mean(np.array(loss_d)))
    Logger['GP_logger'].log(epoch,np.mean(np.array(GP_List)))
    Logger['smooth_logger'].log(epoch,np.mean(np.array(smooth_List)))
    Logger['residual_logger'].log(epoch,np.mean(np.array(res_List)))

def validate(config, testloader, model, writer_dict, device,Logger=None,Epoch =1):
    
    rank = get_rank()
    world_size = get_world_size()
    model.eval()
    ave_loss = AverageMeter()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

    loss_ = []
    loss_t = []
    loss_g = []
    loss_d = []
    ssim_log = []
    SAM_log = []
    psnr_g = []
    with torch.no_grad():
        for num, batch in enumerate(testloader):
            image, label,_, _, MSI, _ = batch
            size = label.size()
            image = image.to(device).float()
            label = label.long().to(device)
            MSI = MSI.to(device).float()
            losses, pred, pred_HSI = model(image, MSI,label)

            # fout = pred_HSI
            # fout_ = fout.detach().cpu().numpy()
            # hsi_g_ = HSI.detach().cpu().numpy()
            # hsi_g = HSI.to(device).float()
            # print('hsi shape:{}, msi shape:{}'.format(hsi_g.shape, fout.shape))
            # if num% 20 == 2:
            #     scio.savemat('./savefile/test_iter{}.mat'.format(Epoch),{'RGB':image.detach().cpu().numpy(),
            #                                                             'GenHSI':fout_,
            #                                                             'HSI':hsi_g_})
            # for i in range(31):
            #     psnr_g.append(skm.compare_psnr(hsi_g_[0,i,:,:]*255,fout_[0,i,:,:]*255,255))
            # fout_0 = np.transpose(fout_,(0,2,3,1))[0,:,:,:]
            # hsi_g_0 = np.transpose(hsi_g_,(0,2,3,1))[0,:,:,:]
            # ssim = skm.compare_ssim(X =fout_0, Y =hsi_g_0,K1 = 0.01, K2 = 0.03,multichannel=True)
            # ssim_log.append(ssim)
            # temp_sam = cal_SAM(fout,hsi_g)
            # SAM_log.append(temp_sam)

            pred = F.upsample(input=pred, size=(
                        size[-2], size[-1]), mode='bilinear')
            loss = losses.mean()
            reduced_loss = reduce_tensor(loss)
            ave_loss.update(reduced_loss.item())

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)
                
        # np_load_old = np.load
        # temp_load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        # # temp = '/public/zhuzhiyu/segmentation/name.npy'
        # temp = '/home/zzy/name.npy'
        # name = temp_load(temp)
        # name = name.item()
        # test_name = name['test']
        reps = scio.loadmat('/home/zzy/comparing_method/data/resp.mat')['resp']
        # reps = scio.loadmat('/public/zhuzhiyu/segmentation/resp0.mat')['resp']
        reps = np.transpose(reps,(1,0))
        max_v = 0.06163118944075685

        # temp_data = lmdb.Environment('/home/zzy/data/')
        # file_txn = temp_data.begin()
        # file_name = [keys for keys,value in file_txn.cursor()]
        # test_name = file_name[-20:]
        file_name = scio.loadmat('/home/zzy/data/icvl_name.mat')['name']
        test_name = file_name[-20:]
        file = h5py.File('/home/zzy/memory/icvl.h5','r')

        for i in range(20):
            # file=h5py.File('/public/zhuzhiyu/data.h5','r')
            # file = h5py.File('/home/zzy/memory/data.h5','r')
            # hsi_g = file_txn.get(test_name[i])
            # hsi_g = pickle.loads(hsi_g)
            hsi_g = file[test_name[i]][:]

            image = np.tensordot(hsi_g,reps,(-1,0))
            image = np.transpose(image,(2,0,1))
            hsi_g = np.transpose(hsi_g,(2,0,1))

            image = torch.from_numpy(image).cuda().float()

            hsi_ = model.gen_HSI(image[None,:,:,:])
            fout_ = hsi_.detach().cpu().numpy()
            if i% 20 == 2:
                scio.savemat('./savefile/test_iter{}.mat'.format(Epoch),{'RGB':image.detach().cpu().numpy(),
                                                                        'GenHSI':fout_,
                                                                        'HSI':hsi_g})
            for j in range(31):
                psnr_g.append(skm.compare_psnr(hsi_g[j,:,:]*255,fout_[0,j,:,:]*255,255))
            fout_0 = np.transpose(fout_,(0,2,3,1))[0,:,:,:]
            hsi_g_0 = np.transpose(hsi_g,(1,2,0))
            ssim = skm.compare_ssim(X =fout_0, Y =hsi_g_0,K1 = 0.01, K2 = 0.03,multichannel=True)
            ssim_log.append(ssim)
            hsi_g = torch.from_numpy(hsi_g).cuda().float()
            temp_sam = cal_SAM(hsi_,hsi_g)
            SAM_log.append(temp_sam)


    ssim_ = np.mean(np.array(ssim_log))
    psnr_ = np.mean(np.array(psnr_g))
    sam_ = np.mean(np.array(SAM_log))
    # Logger.info('SSIM of the image: {}' % (ssim_))
    # Logger.info('PSNR of the image: {}' % (psnr_))
    # Logger.info('SAM  of the image: {}' % (sam_))

    Logger['Lossm_logger'].log(Epoch,ssim_)
    Logger['PSNR_logger'].log(Epoch,psnr_)
    Logger['SAM_logger'].log(Epoch,sam_)
    confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
    reduced_confusion_matrix = reduce_tensor(confusion_matrix)

    confusion_matrix = reduced_confusion_matrix.cpu().numpy()
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    print_loss = ave_loss.average()/world_size

    if rank == 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', print_loss, global_steps)
        writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
    return print_loss, mean_IoU, IoU_array
    

def testval(config, test_dataset, testloader, model, 
        sv_dir='', sv_pred=False,Logger=None):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name = batch
            size = label.size()
            pred = test_dataset.multi_scale_inference(
                        model, 
                        image, 
                        MSI,
                        scales=config.TEST.SCALE_LIST, 
                        flip=config.TEST.FLIP_TEST)
            
            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.upsample(pred, (size[-2], size[-1]), 
                                   mode='bilinear')

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
            
            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc

def test(config, test_dataset, testloader, model, 
        sv_dir='', sv_pred=True,Logger=None):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                        model, 
                        image, 
                        scales=config.TEST.SCALE_LIST, 
                        flip=config.TEST.FLIP_TEST)
            
            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.upsample(pred, (size[-2], size[-1]), 
                                   mode='bilinear')

            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
