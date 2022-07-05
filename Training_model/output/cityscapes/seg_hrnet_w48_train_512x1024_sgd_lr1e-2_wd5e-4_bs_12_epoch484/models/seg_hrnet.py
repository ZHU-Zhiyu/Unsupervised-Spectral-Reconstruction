# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
import sys
import os
print(sys.argv[0])
sys.path.append('./lib/models/')
# print(os.getcwd())
from rgb2hsi import reconnet as hsiG
from SPFNet import SPFNet as respG
# from RGBGenerator import RGBGenerator as MSIG
import itertools

from Discriminator import Discriminator_HSI, Discriminator_RGB, Feature_extractor

from Discriminator import Discriminator_Line as Dis_HSI_Line
from Discriminator import Discriminator_HSI as DisHSI
# from Discriminator import Discriminator_HSI_lite as DisHSIlite
# import Discriminator.Dis_HSI_Line as DisHSILine
from Discriminator import Discriminator_RGB as DisRGB
from Discriminator import Feature_extractor as Fea_extra
from torch.autograd import Variable, Function
import scipy.io as scio
from SPFNet import SPFNet as RespG
from rgb2hsi import reconnet as HSIG
import torch.autograd as ag
# from RGBGenerator import RGBGenerator as RGBG
import math

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class GradNorm(Function):
    @staticmethod
    def forward(ctx, input_x,scale):
        ctx.save_for_backward(scale)
        return input_x
    @staticmethod
    def backward(ctx, grad_output):
        [B,C,H,W] = grad_output.shape
        scale, = ctx.saved_tensors
        gradnrom = (grad_output**2).sum(dim=[1,2,3],keepdim=True).sqrt()
        stdnrom = 1/math.sqrt(C*H*W)
        gradnrom = stdnrom / gradnrom

        gradnrom = torch.clamp(gradnrom,min=0,max=1)

        grad_output = gradnrom * grad_output
        grad_output = scale * grad_output
        # avg = (torch.mean(grad_GAN**2,dim=[1,2,3],keepdim=True)/(H*C*W)).sqrt()
        # grad_GAN = torch.sign(grad_GAN)*avg
        # avg2 = (torch.mean(grad_line**2,dim=[1,2,3],keepdim=True)/(H*C*W)).sqrt()
        # grad_line = torch.sign(grad_line)*avg2

        return grad_output,None

# class GradNorm(Function):
#     @staticmethod
#     def forward(ctx, X_GAN, X_line, X_smooth,X_bp):

#         return torch.clone(X_GAN), torch.clone(X_line), torch.clone(X_smooth), torch.clone(X_bp)
#     @staticmethod
#     def backward(ctx, grad_GAN, grad_line, grad_smooth,grad_bp):
#         [B,C,H,W] = grad_GAN.shape
#         avg = (torch.mean(grad_GAN**2,dim=[1,2,3],keepdim=True)/(H*C*W)).sqrt()
#         grad_GAN = torch.sign(grad_GAN)*avg
#         avg2 = (torch.mean(grad_line**2,dim=[1,2,3],keepdim=True)/(H*C*W)).sqrt()
#         grad_line = torch.sign(grad_line)*avg2
#         # grad_smooth = torch.sign(grad_smooth)/(C*H*W)
#         # grad_bp = torch.sign(grad_bp)/(C*H*W)

#         return grad_GAN, grad_line, grad_smooth, grad_bp

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3, 
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, config, **kwargs):
        extra = config.MODEL.EXTRA
        super(HighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        
        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)
        
        self.last_inp_channels = np.int(np.sum(pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.last_inp_channels,
                out_channels=self.last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(self.last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.last_inp_channels,
                out_channels=config.DATASET.NUM_CLASSES,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        )

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.upsample(x[1], size=(x0_h, x0_w), mode='bilinear')
        x2 = F.upsample(x[2], size=(x0_h, x0_w), mode='bilinear')
        x3 = F.upsample(x[3], size=(x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], 1)

        x_ = self.last_layer(x)

        return x, x_

    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # print(pretrained)
        # print('----------------------- initilization of weights:{}'.format(pretrained))
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            # print('----------------------- initilization of weights:{}'.format(pretrained_dict))
            pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                               if k[6:] in model_dict.keys()}
            print('len of dictionary:{}'.format(len(pretrained_dict)))
            #for k, _ in pretrained_dict.items():
            #    logger.info(
            #        '=> loading {} pretrained model {}'.format(k, pretrained))
            # print('----------------------- initilization of weights:{}'.format(pretrained_dict))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

class Network(nn.Module):
    def __init__(self, criterion, cfg, **kwargs):
        super(Network, self).__init__()
        self.HRNet = HighResolutionNet(cfg, **kwargs)
        self.HRNet.init_weights('/home/zzy/comparing_method/segementation/Train_seg/Joint_training/nonegative_loss_GP_001/pretrain/hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth')
        self.CoutHR = self.HRNet.last_inp_channels
        self.dishsi = DisHSI()

        self.respG = RespG()
        self.hsiG = HSIG()
        # self.RgbG = RGBG()
        self.iter = 1

        self.extractor = Fea_extra(self.CoutHR+31,cfg.DATASET.NUM_CLASSES)

        # self.gen_optimizer = torch.optim.Adam(itertools.chain(self.hsiG.parameters(), self.respG.parameters(), self.extractor.parameters()),lr=1e-4)
        self.gen_optimizer = torch.optim.Adam(itertools.chain(self.hsiG.parameters(), self.respG.parameters()),lr=1e-4)
        # self.dis_optimizer = torch.optim.Adam(itertools.chain(self.dishsi.parameters(), self.dishsi_line.parameters()),lr=1e-3)
        self.dis_optimizer = torch.optim.Adam(itertools.chain(self.dishsi.parameters()),lr=2e-4)

        self.criterion_class = criterion
        self.criterion = nn.BCELoss()

        self.mean = np.array([0.485, 0.456, 0.406])

        self.std = np.array([0.229, 0.224, 0.225])

        self.gradnorm = GradNorm().apply

    def MSI2img(self,MSI):
        
        img = MSI - torch.from_numpy(self.mean).to(MSI.device)[None,:,None,None]
        img = img / torch.from_numpy(self.std).to(MSI.device)[None,:,None,None]
        return img.float()
        
    def calc_gradient_penalty(self, netD, real_data, fake_data, center=0, alpha=None, LAMBDA=.5, device=None):
        # from zero-centered
        if alpha is not None:
            alpha = torch.tensor(alpha, device=device)  # torch.rand(real_data.size(0), 1, device=device)
        else:
            alpha = torch.rand(real_data.size(0), 1, device=device)
        alpha = torch.reshape(alpha,[real_data.size(0), 1, 1, 1])
        alpha = alpha.expand(real_data.size())

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates.requires_grad_(True)

        disc_interpolates = netD(interpolates)

        gradients = ag.grad(outputs=disc_interpolates, inputs=interpolates,
                            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        [B,C,H,W] = gradients.shape
        gradients = torch.reshape(gradients,[B,C*H*W])
        # targets = torch.sign(gradients).detach()/(C*H*W)
        # gradient_penalty = ((gradients - targets) ** 2).sum(dim=1).mean(dim=0) * LAMBDA
        # gradient_penalty = torch.abs(gradients - targets).mean() * LAMBDA
        gradient_penalty = ((gradients.norm(2, dim=1) - center) ** 2).mean() * LAMBDA
        # gradient_penalty = ((gradients.norm(2, dim=1) - center) ** 2)

        # ratio = 
        # ratio = (gradient_penalty.detach()/0.01)**2 * 200
        # ratio = ratio.clip(0, 5000)

        # gradient_penalty = gradient_penalty * ratio
        # gradient_penalty = gradient_penalty.mean()
        return gradient_penalty
        
    def Cal_generatorloss(self,MSI,HSI,rank):
        real_label1 = 1.0
        fake_label1 = 0.0
        one = torch.FloatTensor([1]).to(MSI.device)
        mone = one * -1
        mone.to(one.device)
        b_size, c, h, w = HSI.shape
        # length = 
        real_label = torch.full((b_size,), real_label1, dtype=torch.float, device=HSI.device)
        fake_label = torch.full((b_size,), fake_label1, dtype=torch.float, device=HSI.device)
        real_label2 = torch.full((b_size*h*w,), real_label1, dtype=torch.float, device=HSI.device)
        fake_label2 = torch.full((b_size*h*w,), fake_label1, dtype=torch.float, device=HSI.device)

        # MSI --> Resp
        resp_msi =  self.respG(MSI)

        # MSI + resp --> HSI
        fake_HSI,res_1 = self.hsiG(MSI,resp_msi)

        res_loss = torch.mean(res_1**2)
        # res_loss = torch.mean(torch.abs(res_1))

        # domain loss(2) HSI loss:
        [B,C,H,W] = fake_HSI.shape

        fea = fake_HSI.reshape([B,C,H*W])
        pos = torch.abs(fake_HSI) - fake_HSI
        pos = torch.sum(pos,dim=[1,2,3]).mean()
        smooth = self.first_order(fea)
        smooth = torch.mean(smooth**2)
        # smooth = torch.mean(torch.abs(smooth))
        # max_value = torch.amax(fake_HSI,dim=[1,2,3],keepdim=True)
        mean_vlaue = torch.mean(torch.abs(fake_HSI),dim=[1,2,3],keepdim=True)
        mean_vlaue1 = mean_vlaue.detach()+1e-6
        HSI_pred = self.dishsi( self.gradnorm(fake_HSI, torch.ones(1, device = fake_HSI.device))/mean_vlaue1)
        HSI_pred = torch.squeeze(HSI_pred)
        HSI_dis_loss = self.criterion(HSI_pred,real_label)

        loss =  HSI_dis_loss + res_loss*1e2 + smooth*1e1 #+ pos*1e-2

        if self.iter %20 ==0:
            if rank == 0:
                # print('[iter: {}/991][Gen loss:{:.4f}], [HSI adversarial loss:{:.4f}], [res loss:{:.4f}] [smooth loss :{:.4f}]'.format(self.iter%991,loss.item(), HSI_dis_loss.mean().item(), res_loss.mean().item(),smooth.mean().item()))
                logging.info('[iter: {}/371][Gen loss:{:.4f}], [HSI adversarial loss:{:.4f}], [res loss:{:.4f}] [smooth loss :{:.4f}] [pos loss:{:.4f}]'.format(self.iter%371,loss.item(), HSI_dis_loss.mean().item(), res_loss.mean().item(),smooth.mean().item(),pos.item()))
        return loss, fake_HSI , smooth.item(), res_loss.item() #, MSI_pred , HSI_pred.mean().reshape(1)
    def gen_HSI(self, MSI):

        # MSI --> Resp
        resp_msi =  self.respG(MSI)

        # MSI + resp --> HSI
        fake_HSI,res_1 = self.hsiG(MSI,resp_msi)

        return fake_HSI

    def Third_order(self, feautre):

        for _ in range(3):
            feautre = self.first_order(feautre)
        return feautre

    def second_order(self, feautre):

        for _ in range(2):
            feautre = self.first_order(feautre)
        return feautre

    def first_order(self, feautre):
        input_1 = feautre[:,1:,:]
        input_2 = feautre[:,:-1,:]
        forder = input_1 - input_2
        return forder

    def Cal_discriminatorloss(self,MSI,HSI,rank):
        real_label1 = 1.0
        fake_label1 = 0.0
        one = torch.FloatTensor([1]).to(MSI.device)
        mone = one * -1
        mone.to(one.device)
        b_size = HSI.size(0)

        # MSI --> Resp
        resp_msi =  self.respG(MSI)

        # MSI + resp --> HSI
        fake_HSI,_ = self.hsiG(MSI,resp_msi)

        [B,C,H,W] = fake_HSI.shape
        length = int(B*W*H)
        real_label = torch.full((B,), real_label1, dtype=torch.float, device=HSI.device)
        fake_label = torch.full((B,), fake_label1, dtype=torch.float, device=HSI.device)
        real_label1 = torch.full((length,), real_label1, dtype=torch.float, device=HSI.device)
        fake_label1 = torch.full((length,), fake_label1, dtype=torch.float, device=HSI.device)
        
        # # domain loss(2) HSI loss:
        # max_value = torch.amax(fake_HSI,dim=[1,2,3],keepdim=True)
        # print('shape of the fakeHSI:{}'.format(fake_HSI.shape))
        mean_vlaue = torch.mean(torch.abs(fake_HSI),dim=[1,2,3],keepdim=True)
        mean_vlaue1 = mean_vlaue.detach()+1e-6
        fake_HSI_pred = self.dishsi( self.gradnorm(fake_HSI, torch.ones(1, device = fake_HSI.device))/mean_vlaue1)
        fake_HSI_pred = torch.squeeze(fake_HSI_pred)
        fake_HSI_loss = self.criterion(fake_HSI_pred,fake_label)

        # max_value = torch.amax(HSI,dim=[1,2,3],keepdim=True)
        mean_vlaue = torch.mean(HSI,dim=[1,2,3],keepdim=True)
        mean_vlaue2= mean_vlaue.detach()+1e-6
        # print('shape of the real:{}'.format(HSI.shape))
        real_HSI_pred = self.dishsi( self.gradnorm(HSI, torch.ones(1, device = HSI.device))/mean_vlaue2)
        real_HSI_pred = torch.squeeze(real_HSI_pred)
        real_HSI_loss = self.criterion(real_HSI_pred,real_label)

        # GP_loss = self.calc_gradient_penalty(self.dishsi, HSI.detach()/mean_vlaue2, fake_HSI.detach()/mean_vlaue1, center=0, alpha=None, LAMBDA=10, device=real_HSI_pred.device)
        GP_loss = fake_HSI_loss*0

        # GP_loss = 0
        loss = fake_HSI_loss + real_HSI_loss #+ GP_loss
        # loss = loss_MSI + fake_MSI_loss + real_MSI_loss + fake_HSIU_pred + HSIU_pred + loss_HSIU + fake_HSI_loss + real_HSI_loss
        self.iter +=1
        if self.iter %20 ==0:
            if rank == 0:
                # print('[iter: {}/106][Dis loss:{:.4f}], [HSI  fake loss:{:.2f}, real loss:{:.2f}, GP loss:{:.2f}]'.format(self.iter%106,loss.item(),fake_HSI_loss.item(),real_HSI_loss.item(),GP_loss.item()))
                logging.info('[iter: {}/371][Dis loss:{:.4f}], [HSI  fake loss:{:.2f}, real loss:{:.2f}, GP loss:{:.2f}]'.format(self.iter%371,loss.item(),fake_HSI_loss.item(),real_HSI_loss.item(),GP_loss.item()))
                # print('[iter: {}/106][Dis loss:{:.4f}], [HSI  fake loss:{:.2f}, real loss:{:.2f}]'.format(self.iter%106,loss.item(),fake_HSI_loss.item(),real_HSI_loss.item()))
        return  loss, GP_loss

    def update_generator(self,MSI,HSI,img,rank, seg_label):
        MSI1 = torch.clone(MSI)
        # img = self.MSI2img(MSI)
        [B,C,H,W] = HSI.shape
            
        # spectral feature generation
        self.gen_optimizer.zero_grad()
        MSI1 = torch.nn.functional.interpolate(MSI,size=(int(H),int(W)))
        gen_loss, fake_HSI, smooth, res_loss = self.Cal_generatorloss(MSI1,HSI,rank)

        # spatial feature generation
        with torch.no_grad():
            _,fea = self.HRNet(img)
        [B,C,H,W] = fea.shape

        # spectral feature generation
        fake_HSI = torch.nn.functional.interpolate(self.gradnorm(fake_HSI, torch.ones(1,device = fake_HSI.device)*2),size=(int(H),int(W)))

        pred = self.extractor(Afea = fea, Efea = fake_HSI)

        loss = self.criterion_class(pred,seg_label)
        
        if self.iter% 400 == 0:
            scio.savemat('./savefile/train_iter{}.mat'.format(self.iter),{'RGB':MSI.detach().cpu().numpy(),
                                                                        'GenHSI':fake_HSI.detach().cpu().numpy(),
                                                                        'HSI':HSI.detach().cpu().numpy()})

        return gen_loss + loss, smooth, res_loss

    def update_discriminator(self,MSI,HSI,rank):

        self.dis_optimizer.zero_grad()
        MSI = torch.nn.functional.interpolate(MSI,size=(int(128),int(128)))
        dis_loss, GP_loss = self.Cal_discriminatorloss(MSI,HSI,rank)
        dis_loss.backward()
        dis_loss1 = dis_loss.item()

        self.dis_optimizer.step()
        return dis_loss1, GP_loss.item()

    def forward(self,img,MSI,label=None):
        # MSI --> Resp
        with torch.no_grad():
            img1 = img
            _,fea = self.HRNet(img1)
            [B,C,H,W] = fea.shape

            MSI = torch.nn.functional.interpolate(MSI,size=(int(H),int(W)))
            fea = torch.nn.functional.interpolate(fea,size=(int(H),int(W)))

            # img = self.MSI2img(MSI)
            # MSI1 = torch.nn.functional.interpolate(MSI,size=(int(H/4),int(W/4)))
            resp_msi =  self.respG(MSI)

            # MSI + resp --> HSI
            fake_HSI,res_ = self.hsiG(MSI,resp_msi)
            # fake_HSI = torch.nn.functional.interpolate(fake_HSI,size=(int(H),int(W)))

        pred = self.extractor(Afea = fea, Efea = fake_HSI)

        if label is not None:
            loss = self.criterion_class(pred,label)
            return loss , pred, fake_HSI
        else:

            return 0 , pred, fake_HSI
    # def forward(self,img,MSI,label=None):
    #     # MSI --> Resp
    #     img1 = img
    #     fea,pred = self.HRNet(img1)

    #     if label is not None:
    #         loss = self.criterion_class(pred,label)
    #         return loss , pred, 0
    #     else:

    #         return 0 , pred, 0


def get_seg_model(criterion, cfg, **kwargs):
    model = Network(criterion, cfg, **kwargs)
    # model.HRNet.init_weights('/home/zzy/comparing_method/segementation/Train_seg/Joint_training/nonegative_loss_GP_001/pretrain/hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth')

    return model
