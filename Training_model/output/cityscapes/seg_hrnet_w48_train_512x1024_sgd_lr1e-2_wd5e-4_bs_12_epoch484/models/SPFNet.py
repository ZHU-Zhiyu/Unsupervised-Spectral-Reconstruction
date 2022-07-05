import torch
import torch.nn as nn
# from SpaNet import BCR, denselayer
import numpy as np
import torch.nn.functional as f
import scipy.io as scio
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
        
class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class My_Bn_1(nn.Module):
    def __init__(self):
        super(My_Bn_1,self).__init__()
    def forward(self,x):
        # print(x.shape)
        # _,C,_,_ = x.shape
        # x1,x2 = torch.split(x,C//2,dim=1)
        # x1 = x1 - nn.AdaptiveAvgPool2d(1)(x1)
        # x1 = torch.cat([x1,x2],dim=1)
        # x = 
        return x - torch.mean(x,dim = 1,keepdim=True)

class My_Bn_2(nn.Module):
    def __init__(self):
        super(My_Bn_2,self).__init__()
    def forward(self,x):
        # print(x.shape)
        # _,C,_,_ = x.shape
        # x1,x2 = torch.split(x,C//2,dim=1)
        # x1 = x1 - nn.AdaptiveAvgPool2d(1)(x1)
        # x1 = torch.cat([x1,x2],dim=1)
        # x = 
        return x - nn.AdaptiveAvgPool2d(1)(x)

class BCR(nn.Module):
    def __init__(self,kernel,cin,cout,group=1,stride=1,RELU=True,padding = 0,BN=False,spatial_norm = False):
        super(BCR,self).__init__()
        if stride > 0:
            self.conv = nn.Conv2d(in_channels=cin, out_channels=cout,kernel_size=kernel,groups=group,stride=stride,padding= padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels=cin, out_channels=cout,kernel_size=kernel,groups=group,stride=int(abs(stride)),padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.Swish = MemoryEfficientSwish()
        
        if RELU:
            if BN:
                if spatial_norm:
                    self.Bn = My_Bn_2()
                    self.Module = nn.Sequential(
                        self.conv,
                        self.Bn,
                        self.relu,
                    )
                else:
                    self.Bn = My_Bn_1()
                    self.Module = nn.Sequential(
                        self.conv,
                        self.Bn,
                        self.relu,
                    )
            else:
                self.Module = nn.Sequential(
                    self.conv,
                    self.relu
                )
        else:
            if BN:
                if spatial_norm:
                    self.Bn = My_Bn_2()
                    self.Module = nn.Sequential(
                        self.Bn,
                        self.conv,
                    )
                else:
                    self.Bn = My_Bn_1()
                    self.Module = nn.Sequential(
                        self.Bn,
                        self.conv,
                        self.relu,
                    )
            else:
                self.Module = nn.Sequential(
                    self.conv,
                )

    def forward(self, x):
        output = self.Module(x)
        return output

class denselayer(nn.Module):
    def __init__(self,cin,cout=31,RELU=True,BN=True,kernel_size=3):
        super(denselayer, self).__init__()
        self.compressLayer = BCR(kernel=1,cin=cin,cout=cout,RELU=RELU,BN=BN,spatial_norm=False)
        self.actlayer = BCR(kernel=kernel_size,cin=cout,cout=cout,group=cout,RELU=RELU,padding=(kernel_size-1)//2,BN=BN,spatial_norm=False)
    def forward(self, x):
        output = self.compressLayer(x)
        output = self.actlayer(output)

        return output

class stage(nn.Module):
    def __init__(self,cin,cout,Bn=False,kernel_size = 5):
        super(stage, self).__init__()
        assert(len(cout)==2)
        self.conv = nn.ModuleList([])
        self.conv.append(denselayer(cin = cin, cout= cout[0],RELU=True,BN=Bn,kernel_size=kernel_size))
        self.conv.append(denselayer(cin = cin+cout[0], cout= cout[1],RELU=True,BN=Bn,kernel_size=kernel_size))
        # self.conv1 = 
        # self.conv2 = denselayer(cin = cin )

    def forward(self,MSI):
        feature = [MSI]
        [B,C,H,W] = MSI.shape
        for layer in self.conv:
            feature_ = layer(torch.cat(feature,1))
            # feature = layer(MSI)
            feature.append(feature_)
        output = nn.AdaptiveMaxPool2d(H//2)(feature[-1])
        return output

class Fstage(nn.Module):
    def __init__(self,cin,cout,Bn=False,kernel_size = 3):
        super(Fstage, self).__init__()
        self.conv = nn.ModuleList([])
        self.conv.append(denselayer(cin = 64, cout= 128,RELU=True,BN=Bn,kernel_size=kernel_size))
        self.conv.append(denselayer(cin = 64+128, cout= 128,RELU=True,BN=Bn,kernel_size=kernel_size))
        self.conv.append(denselayer(cin = 64+128*2, cout= 128,RELU=True,BN=Bn,kernel_size=kernel_size))
        self.final_conv = denselayer(cin=128,cout=cout,kernel_size=3,RELU=False,BN=Bn)
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self,Feature):
        feature = [Feature]
        [B,C,H,W] = Feature.shape
        for layer in self.conv:
            feature_ = layer(torch.cat(feature,1))
            # feature = layer(MSI)
            feature.append(feature_)
        feature = self.final_conv(feature[-1])
        feature = self.pool(feature)
        return feature

class SPFNet(nn.Module):
    def __init__(self, cin=0,cout=0,final=False,extra=0,BN = False):
        super(SPFNet, self).__init__()
        self.stages = nn.Sequential(
            stage(cin=3,cout=[16,32],kernel_size=5),
            stage(cin=32,cout=[64,64],kernel_size=5),
            stage(cin=64,cout=[64,64],kernel_size=3),
            stage(cin=64,cout=[64,64],kernel_size=3),
            )
        self.fstage = Fstage(cin=64,cout=28*3)
        # 28 * 3 * 33
        self.resp = torch.from_numpy(scio.loadmat('/home/zzy/comparing_method/unsupervisedSR/SingleGAN/resp.mat')['resp'])[:,:,:-2]
        # self.resp = torch.from_numpy(scio.loadmat('/public/zhuzhiyu/segmentation/resp.mat')['resp'])[:,:,2:]
        self.resp = self.resp.float()
        mean_value = self.resp.sum(dim=2,keepdim=True)
        self.resp = self.resp / mean_value
        # print('Shape of Resp:{}'.format(self.resp.shape))
    def forward(self,MSI):
        feature = MSI
        feature = self.stages(feature)
        feature = self.fstage(feature)
        [B,C,H,W] = feature.shape
        feature = feature.reshape([B,28,3])
        # feature B 28*3
        
        Amp = torch.sigmoid(feature)
        Amp = torch.mean(Amp,dim=1,keepdim=True)
        Amp = Amp / torch.mean(Amp,dim=2,keepdim=True)

        feature = torch.nn.functional.softmax(feature,dim=1)
        feature = feature*Amp

        feature = feature.reshape([B,28*3])
        # feature = feature/torch.sum(feature,dim=1,keepdim=True)

        resp = torch.reshape(self.resp.detach().to(feature.device),[3*28,31])[None,:,:]*feature[:,:,None]
        resp = torch.sum(torch.reshape(resp,[B,28,3,31]),dim=1)
        return resp