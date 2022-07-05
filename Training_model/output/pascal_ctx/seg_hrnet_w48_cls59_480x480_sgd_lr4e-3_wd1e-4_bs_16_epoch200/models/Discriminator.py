import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as f
import math

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

        return x - torch.mean(x,dim = 1,keepdim=True)

class My_Bn_2(nn.Module):
    def __init__(self):
        super(My_Bn_2,self).__init__()
    def forward(self,x):

        return x - nn.AdaptiveAvgPool2d(1)(x)
        
####################################################################
#--------------------- Spectral Normalization ---------------------
#  This part of code is copied from pytorch master branch (0.5.0)
####################################################################
class SpectralNorm(object):
    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                       'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps
    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        weight_mat = weight
        if self.dim != 0:
        # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        weight_mat = weight_mat.reshape(height, -1)
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
                sigma = torch.dot(u, torch.matmul(weight_mat, v))
                weight = weight / sigma
            return weight, u
    def remove(self, module):
        weight = getattr(module, self.name)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight))
    def __call__(self, module, inputs):
        if module.training:
            weight, u = self.compute_weight(module)
            setattr(module, self.name, weight)
            setattr(module, self.name + '_u', u)
        else:
            r_g = getattr(module, self.name + '_orig').requires_grad
            getattr(module, self.name).detach_().requires_grad_(r_g)
    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        height = weight.size(dim)
        u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        module.register_buffer(fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_forward_pre_hook(fn)
        return fn
def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                           torch.nn.ConvTranspose2d,
                           torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module

class BCR(nn.Module):
    def __init__(self,kernel,cin,cout,group=1,stride=1,RELU=True,padding = 0,BN=False,spatial_norm = False,sn=False):
        super(BCR,self).__init__()
        if stride > 0:
            if sn:
                self.conv = spectral_norm(nn.Conv2d(in_channels=cin, out_channels=cout,kernel_size=kernel,groups=group,stride=stride,padding= padding))
            else:
                self.conv = nn.Conv2d(in_channels=cin, out_channels=cout,kernel_size=kernel,groups=group,stride=stride,padding= padding)
        else:
            if sn:
                self.conv = spectral_norm(nn.ConvTranspose2d(in_channels=cin, out_channels=cout,kernel_size=kernel,groups=group,stride=int(abs(stride)),padding=padding))
            else:
                self.conv = nn.ConvTranspose2d(in_channels=cin, out_channels=cout,kernel_size=kernel,groups=group,stride=int(abs(stride)),padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.Swish = MemoryEfficientSwish()
        
        if RELU:
            if BN:
                if spatial_norm:
                    self.Bn = nn.BatchNorm2d(num_features=cout)
                    # self.Bn = nn.InstanceNorm2d(num_features=cout)
                    self.Module = nn.Sequential(
                        self.conv,
                        self.Bn,
                        self.Swish,
                    )
                else:
                    self.Bn = nn.BatchNorm2d(num_features=cout)
                    # self.Bn = nn.InstanceNorm2d(num_features=cout)
                    self.Module = nn.Sequential(
                        self.conv,
                        self.Bn,
                        self.Swish,
                    )
            else:
                self.Module = nn.Sequential(
                    self.conv,
                    self.Swish
                )
        else:
            if BN:
                if spatial_norm:
                    self.Bn = nn.BatchNorm2d(num_features=cout)
                    # self.Bn = nn.InstanceNorm2d(num_features=cout)
                    self.Module = nn.Sequential(
                        self.Bn,
                        self.conv,
                    )
                else:
                    self.Bn = nn.BatchNorm2d(num_features=cout)
                    # self.Bn = nn.InstanceNorm2d(num_features=cout)
                    self.Module = nn.Sequential(
                        self.Bn,
                        self.conv,
                    )
            else:
                self.Module = nn.Sequential(
                    self.conv,
                )

    def forward(self, x):
        output = self.Module(x)
        return output

class B3CR(nn.Module):
    def __init__(self,kernel,cin,cout,group=1,stride=1,RELU=True,padding = 0,BN=False,spatial_norm = False,sn=False):
        super(B3CR,self).__init__()
        if stride > 0:
            if sn:
                self.conv = spectral_norm(nn.Conv1d(in_channels=cin, out_channels=cout,kernel_size=kernel,groups=group,stride=stride,padding= padding))
            else:
                self.conv = nn.Conv1d(in_channels=cin, out_channels=cout,kernel_size=kernel,groups=group,stride=stride,padding= padding)
        else:
            if sn:
                self.conv = spectral_norm(nn.ConvTranspose2d(in_channels=cin, out_channels=cout,kernel_size=kernel,groups=group,stride=int(abs(stride)),padding=padding))
            else:
                self.conv = nn.ConvTranspose2d(in_channels=cin, out_channels=cout,kernel_size=kernel,groups=group,stride=int(abs(stride)),padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.Swish = MemoryEfficientSwish()
        
        if RELU:
            if BN:
                if spatial_norm:
                    self.Bn = nn.BatchNorm1d(num_features=cout)
                    # self.Bn = nn.InstanceNorm2d(num_features=cout)
                    self.Module = nn.Sequential(
                        self.conv,
                        self.Bn,
                        self.Swish,
                    )
                else:
                    self.Bn = nn.BatchNorm1d(num_features=cout)
                    # self.Bn = nn.InstanceNorm2d(num_features=cout)
                    self.Module = nn.Sequential(
                        self.conv,
                        self.Bn,
                        self.Swish,
                    )
            else:
                self.Module = nn.Sequential(
                    self.conv,
                    self.Swish
                )
        else:
            if BN:
                if spatial_norm:
                    self.Bn = nn.BatchNorm1d(num_features=cout)
                    # self.Bn = nn.InstanceNorm2d(num_features=cout)
                    self.Module = nn.Sequential(
                        self.Bn,
                        self.conv,
                    )
                else:
                    self.Bn = nn.BatchNorm1d(num_features=cout)
                    # self.Bn = nn.InstanceNorm2d(num_features=cout)
                    self.Module = nn.Sequential(
                        self.conv,
                        self.Bn,
                        self.Swish,
                    )
            else:
                self.Module = nn.Sequential(
                    self.conv,
                )

    def forward(self, x):
        output = self.Module(x)
        return output

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class reshape1(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1,1,1)

class denselayer(nn.Module):
    def __init__(self,cin,cout=31,RELU=True,BN=True,kernel_size=3,stride=1,act=True,dropout = False):
        super(denselayer, self).__init__()
        self.compressLayer = BCR(kernel=1,cin=cin,cout=cout,RELU=RELU,BN=BN,spatial_norm=True,stride=1)
        self.act = act
        self.actlayer = BCR(kernel=kernel_size,cin=cout,cout=cout,group=cout,RELU=RELU,padding=(kernel_size-1)//2,BN=BN,spatial_norm=True,stride=stride)
        if dropout == True:
            self.dropout = nn.Dropout2d(0.1)
        self.drop = dropout
    def forward(self, x):
        if self.drop:
            [B,C,H,W] = x.shape
            x = x.permute([0,2,3,1]).reshape([B*H*W,C,1,1])
            x = self.dropout(x)
            x = x.reshape([B,H,W,C]).permute([0,3,1,2])
        output = self.compressLayer(x)
        if self.act == True:
            output = self.actlayer(output)

        return output

# class denselayer(nn.Module):
#     def __init__(self,cin,cout=31,RELU=True,BN=True,kernel_size=3,stride=1,act=True):
#         super(denselayer, self).__init__()
#         self.compressLayer = BCR(kernel=1,cin=cin,cout=cout,RELU=RELU,BN=BN,spatial_norm=True,stride=1)
#         self.act = act
#         self.actlayer = BCR(kernel=kernel_size,cin=cout,cout=cout,group=cout,RELU=RELU,padding=(kernel_size-1)//2,BN=BN,spatial_norm=True,stride=stride)
#     def forward(self, x):
#         output = self.compressLayer(x)
#         if self.act == True:
#             output = self.actlayer(output)

#         return output

class denselayer1(nn.Module):
    def __init__(self,cin,cout=31,RELU=True,BN=True,kernel_size=3,stride=1,act=True):
        super(denselayer1, self).__init__()
        self.compressLayer = B3CR(kernel=3,cin=cin,cout=cout,RELU=RELU,BN=BN,spatial_norm=True,stride=stride,padding=(kernel_size-1)//2)
        self.act = act
        self.actlayer = BCR(kernel=kernel_size,cin=cout,cout=cout,group=cout,RELU=RELU,padding=(kernel_size-1)//2,BN=BN,spatial_norm=True,stride=stride)
    def forward(self, x):
        output = self.compressLayer(x)

        return output

class Dis_stage_Line(nn.Module):
    def __init__(self,cin=31,cout=64,down=True):
        super(Dis_stage_Line, self).__init__()

        self.denseconv = nn.Sequential(
            denselayer(cin = cin , cout= cout, RELU=True,kernel_size= 3,stride=1,BN=False,act=False),
            denselayer(cin = cin + cout*1, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=False,act=False),
            denselayer(cin = cin + cout*2, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=False,act=False),
            denselayer(cin = cin + cout*3, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=False,act=False),
            # denselayer(cin = cin + cout*4, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=False,act=False),
            # denselayer(cin = cin + cout*5, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=False,act=False),
            # denselayer(cin = cin + cout*6, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=False,act=False),
        )

    def forward(self,HSI):

        feature = [HSI]

        for conv in self.denseconv:
            feature.append(conv(torch.cat(feature,dim=1)))

        return feature[-1]

class Dis_stage(nn.Module):
    def __init__(self,cin=31,cout=64,down=True):
        super(Dis_stage, self).__init__()
        self.down = down
        if down:
            self.downsample = nn.Sequential(
                denselayer(cin=cin,cout=cout,RELU=True,kernel_size=3,stride=2,BN=True),
                denselayer(cin=cout,cout=cout,RELU=True,kernel_size=3,stride=2,BN=True),
            )
        else:
            self.downsample = nn.Sequential(
                denselayer(cin=cin,cout=cout,RELU=True,kernel_size=3,stride=1,BN=True))
        self.denseconv = nn.Sequential(
            denselayer(cin = cout, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=True),
            denselayer(cin = cout*2, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=True),
            denselayer(cin = cout*3, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=True),
            denselayer(cin = cout*4, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=True),
            denselayer(cin = cout*5, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=True),
            denselayer(cin = cout*6, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=True),
            denselayer(cin = cout*7, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=True),
        )

    def forward(self,MSI):
        if self.down:
            dfeature = self.downsample(MSI)
        else:
            dfeature = self.downsample(MSI)

        feature = [dfeature]

        for conv in self.denseconv:
            feature.append(conv(torch.cat(feature,dim=1)))

        return feature[-1] + dfeature

class Dis_RGB_stage(nn.Module):
    def __init__(self,cin=3,cout=32):
        super(Dis_RGB_stage, self).__init__()
        self.downsample = nn.Sequential(
            BCR(cin=cin,cout=cout,RELU=True,kernel=5,stride=2,BN=True),
            BCR(cin=cout,cout=cout,RELU=True,kernel=5,stride=2,BN=True),
        )
        self.denseconv = nn.Sequential(
            BCR(cin = cout, cout= cout, RELU=True,kernel= 3,stride=1,BN=True,padding=1),
            BCR(cin = cout*2, cout= cout, RELU=True,kernel= 3,stride=1,BN=True,padding=1),
            BCR(cin = cout*3, cout= cout, RELU=True,kernel= 3,stride=1,BN=True,padding=1),
            BCR(cin = cout*4, cout= cout, RELU=True,kernel= 3,stride=1,BN=True,padding=1),
            BCR(cin = cout*5, cout= cout, RELU=True,kernel= 3,stride=1,BN=True,padding=1),
            BCR(cin = cout*6, cout= cout, RELU=True,kernel= 3,stride=1,BN=True,padding=1),
            BCR(cin = cout*7, cout= cout, RELU=True,kernel= 3,stride=1,BN=True,padding=1),
        )

    def forward(self,MSI):

        dfeature = self.downsample(MSI)

        feature = [dfeature]

        for conv in self.denseconv:
            feature.append(conv(torch.cat(feature,dim=1)))

        return feature[-1] + dfeature

class Discriminator_Line(nn.Module):
    def __init__(self):
        super(Discriminator_Line, self).__init__()

        self.dis_stage = Dis_stage_Line(cin=31, cout= 128)
        self.classifier = nn.Sequential(
            reshape1(),
            denselayer(cin = 128,cout=1,RELU=False,BN=False,kernel_size=1,stride=1,act=False),
            Flatten(),
            nn.Sigmoid())
    def forward(self,HSI):

        feature = self.dis_stage(HSI)
        # prod = self.classifier(HSI)
        prod = self.classifier(feature)
    
        return prod
# class Discriminator1(nn.Module):
#     def __init__(self):
#         super(Discriminator1, self).__init__()
#         width = 16
#         self.stage1 = Dis_RGB_stage(cin=6,cout=32)
#         self.stage2 = Dis_RGB_stage(cin=32,cout=64)
#         self.classifier = nn.Sequential(
#             nn.AdaptiveAvgPool2d(2),
#             reshape1(),
#             BCR(cin = 256,cout=1,RELU=False,BN=False,kernel=1,stride=1),
#             Flatten())

#     def forward(self,MSI):

#         feature = self.stage1(MSI)
#         feature = self.stage2(feature)
#         prod = self.classifier(feature)
    
#         return prod

class Discriminator_RGB(nn.Module):
    def __init__(self):
        super(Discriminator_RGB, self).__init__()
        width = 16
        self.stage1 = Dis_RGB_stage(cin=6,cout=32)
        self.stage2 = Dis_RGB_stage(cin=32,cout=64)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            reshape1(),
            BCR(cin = 256,cout=1,RELU=False,BN=False,kernel=1,stride=1),
            Flatten())

    def forward(self,MSI):

        feature = self.stage1(MSI)
        feature = self.stage2(feature)
        prod = self.classifier(feature)
    
        return prod

class Discriminator_HSI(nn.Module):
    def __init__(self):
        super(Discriminator_HSI, self).__init__()
        width = 62
        self.stage1 = Dis_stage(cin=31,cout=64)
        self.stage2 = Dis_stage(cin=64,cout=128)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            reshape1(),
            denselayer(cin = 512,cout=1,RELU=False,BN=False,kernel_size=1,stride=1),
            Flatten(),
            nn.Sigmoid())
    def forward(self,HSI):
        # HSI = GradNorm.apply(HSI)
        feature = self.stage1(HSI)
        feature = self.stage2(feature)
        prod = self.classifier(feature)
    
        return prod

class Discriminator_HSI_lite(nn.Module):
    def __init__(self):
        super(Discriminator_HSI_lite, self).__init__()
        width = 62
        self.stage1 = Dis_stage(cin=31,cout=64,down=True)
        self.stage2 = Dis_stage(cin=64,cout=256,down=False)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            reshape1(),
            denselayer(cin = 256,cout=1,RELU=False,BN=False,kernel_size=1,stride=1),
            Flatten())
    def forward(self,HSI):

        feature = self.stage1(HSI)
        feature = self.stage2(feature)
        prod = self.classifier(feature)
    
        return prod

# class Feature_extractor(nn.Module):
#     def __init__(self,cin=31,cout=64):
#         super(Feature_extractor, self).__init__()
#         ecin = 31
#         ecout = 64
#         self.adenseconv = denselayer(cin = cin-31, cout= ecout, RELU=True,kernel_size= 3,stride=1,BN=True,act=False)
#         self.fianl_denseconv = nn.Sequential(
#             denselayer(cin = ecout+31, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=True,act=True),
#             denselayer(cin = ecout+31 + cout*1, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=True,act=True),
#             denselayer(cin = ecout+31 + cout*2, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=True,act=True),
#             denselayer(cin = ecout+31 + cout*3, cout= cout, RELU=True,kernel_size= 3,stride=1,BN=True,act=True),
#         )

#     def forward(self,Afea,Efea):

#         Afea1 = self.adenseconv(Afea)
        

#         feature2 = torch.cat([Afea1,Efea],dim=1)
#         feature2 = [feature2]
#         for conv in self.fianl_denseconv:
#             feature2.append(conv(torch.cat(feature2,dim=1)))

#         return feature2[-1]

class Feature_extractor(nn.Module):
    def __init__(self,cin=31,cout=64):
        super(Feature_extractor, self).__init__()
        ecin = 31
        ecout = 64
        # self.spectraldown = denselayer1(cin = 1, cout= 4, RELU=True,kernel_size= 3,stride=2,BN=True)
        # self.edenseconv = nn.Sequential(
        #     denselayer1(cin = 4, cout= 4, RELU=True,kernel_size= 3,stride=1,BN=True),
        #     denselayer1(cin = 8, cout= 4, RELU=True,kernel_size= 3,stride=1,BN=True),
        #     denselayer1(cin = 12, cout= 4, RELU=True,kernel_size= 3,stride=1,BN=True),
        #     denselayer1(cin = 16, cout= 4, RELU=True,kernel_size= 3,stride=1,BN=True),
        #     denselayer1(cin = 20, cout= 4, RELU=True,kernel_size= 3,stride=1,BN=True,act=False),
        # )
        # self.spectraldown = denselayer1(cin = 1, cout= 4, RELU=True,kernel_size= 3,stride=2,BN=True)
        self.edenseconv = nn.Sequential(
            denselayer(cin = ecin, cout= ecout, RELU=True,kernel_size= 3,stride=1,BN=True,act=True),
            denselayer(cin = ecin + ecout*1, cout= ecout, RELU=True,kernel_size= 3,stride=1,BN=True,act=True),
            denselayer(cin = ecin + ecout*2, cout= ecout, RELU=True,kernel_size= 3,stride=1,BN=True,act=True),
            denselayer(cin = ecin + ecout*3, cout= ecout, RELU=True,kernel_size= 3,stride=1,BN=True,act=True),
        )
        self.adenseconv = denselayer(cin = cin-31, cout= ecout, RELU=True,kernel_size= 3,stride=1,BN=True,act=False)
        self.fianl_denseconv = nn.Sequential(
            denselayer(cin = ecout*1 + cout*1, cout= cout*2, RELU=True,kernel_size= 3,stride=1,BN=True,act=True),
            denselayer(cin = ecout*1 + cout*3, cout= cout*2, RELU=True,kernel_size= 3,stride=1,BN=True,act=True),
            denselayer(cin = ecout*1 + cout*5, cout= cout*2, RELU=True,kernel_size= 3,stride=1,BN=True,act=True),
            denselayer(cin = ecout*1 + cout*7, cout= cout*2, RELU=True,kernel_size= 3,stride=1,BN=True,act=True),
            denselayer(cin = ecout*1 + cout*9, cout= cout, RELU=False,kernel_size= 3,stride=1,BN=False,act=False,dropout = True),
        )
        self.dropout = nn.Dropout2d(0.2)
    def forward(self,Afea,Efea):

        # Afea1 = self.adenseconv(Afea)

        # [B,C,H,W] = Efea.shape
        [B,C,H,W] = Efea.shape
        Efea = Efea.permute([0,2,3,1]).reshape([B*H*W,C,1,1])
        Efea = self.dropout(Efea)
        Efea = Efea.reshape([B,H,W,C]).permute([0,3,1,2])

        # Efea = Efea.permute([0,2,3,1])
        # Efea = Efea.reshape([B*H*W,1,C])
        # Efea = self.spectraldown(Efea)

        feature = [Efea]
        for conv in self.edenseconv:
            feature.append(conv(torch.cat(feature,dim=1)))
        # temp = feature[-1]
        # temp = temp.reshape([B,H,W,64]).permute([0,3,1,2])
        feature2 = torch.cat([Afea,feature[-1]],dim=1)
        feature2 = [feature2]
        for conv in self.fianl_denseconv:
            feature2.append(conv(torch.cat(feature2,dim=1)))

        return feature2[-1]
        
