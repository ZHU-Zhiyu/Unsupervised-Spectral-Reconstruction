import torch
import torch.nn as nn
# from SpaNet import BCR, denselayer
import numpy as np
import torch.nn.functional as f
from rgb2hsi import reconnet as hsiG
from SPFNet import SPFNet as respG
from RGBGenerator import RGBGenerator as MSIG
import itertools
from Discriminator import Discriminator_HSI as DisHSI
from Discriminator import Discriminator_HSI_lite as DisHSIlite
from Discriminator import Discriminator_RGB as DisRGB
from SPFNet import SPFNet as RespG
from rgb2hsi import reconnet as HSIG
from RGBGenerator import RGBGenerator as RGBG
import torch.nn.functional as F
import torch.autograd as autograd
from utils import get_gaussian_kernel as gaussian

# from utils
def load_model(model,mode_dict):
    # state_dict = torch.load(model_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in mode_dict.items():
        # name = k[7:] # remove `module.`
        name = k # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

class Network(nn.Module):
    def __init__(self, lr=1e-3, iteration=100000):
        super(Network, self).__init__()

        self.dishsi = DisHSI()
        self.dishsilite = DisHSIlite()
        self.respG = RespG()
        # state_dict = torch.load('/home/zzy/comparing_method/unsupervised/SPFWeights/state_dicr_499.pkl')
        # self.respG = load_model(self.respG,state_dict['spenet'])
        self.hsiG = HSIG()
        self.RgbG = RGBG()
        self.iter = 1
        # self.gen_optimizer1 = torch.optim.Adam(itertools.chain(self.respG.parameters(),self.RgbG.parameters()),lr=lr)
        self.gen_optimizer = torch.optim.Adam(itertools.chain(self.hsiG.parameters(),self.respG.parameters()),lr=1e-4)
        self.dis_optimizer = torch.optim.Adam(itertools.chain(self.dishsi.parameters(),self.dishsilite.parameters()),lr=2e-4)
        self.blurlayer = gaussian(kernel_size=5, sigma=2)
        self.criterion = nn.BCELoss()
    # def Cal_gradient_penalty(self, netD, real_data, fake_data):
    #     [B,C,H,W] = real_data.shape
    #     alpha = torch.rand(B, 1)
    #     alpha = alpha.expand(B, real_data.nelement()//B).contiguous().view(B, C, H, W)
    #     alpha = alpha.to(real_data.device)

    #     interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    #     interpolates = interpolates.to(real_data.device)
    #     interpolates = autograd.Variable(interpolates, requires_grad=True)

    #     disc_interpolates = netD(interpolates)

    #     gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
    #                             grad_outputs=torch.ones(disc_interpolates.size()).to(real_data.device),
    #                             create_graph=True, retain_graph=True, only_inputs=True)[0]
    #     gradients = gradients.view(gradients.size(0), -1)

    #     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()*10
    #     return gradient_penalty
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

        gradient_penalty = ((gradients.norm(2, dim=1) - center) ** 2).mean() * LAMBDA

        return gradient_penalty
    # def Cal_Respgeneratorloss(self,MSI,MSI2,HSI):
    #     one = torch.FloatTensor([1]).to(MSI.device)
    #     mone = one * -1
    #     mone.to(one.device)

    #     # MSI --> Resp
    #     resp_msi =  self.respG(MSI)

    #     # MSI + resp --> HSI
    #     fake_HSI,res_1 = self.hsiG(MSI,resp_msi)

    #     # HSI + Resp --> fake_MSI
    #     fake_MSI = self.RgbG(HSI,resp_msi)

    #     # fake_MSI --> fake_resp
    #     fake_resp = self.respG(fake_MSI)

    #     # fake_HSI + fake_resp --> recon_HSI
    #     recon_MSI = self.RgbG(fake_HSI,fake_resp)
    #     # fake_HSI + fake_resp --> recon_HSI
    #     recon_HSI,res_2 = self.hsiG(fake_MSI,resp_msi)

    #     # HSI recon loss, MSI recon loss
    #     # recon_HSI_loss = torch.sum(torch.abs(recon_HSI-HSI))*100
    #     # recon_MSI_loss = torch.sum(torch.abs(recon_MSI-MSI))
    #     recon_HSI_loss = torch.mean(torch.abs(recon_HSI-HSI),dim=0)*0.1
    #     recon_MSI_loss = torch.mean(torch.abs(recon_MSI-MSI),dim=0)*0.1
    #     recon_HSI_loss = torch.sum(recon_HSI_loss)
    #     recon_MSI_loss = torch.sum(recon_MSI_loss)
    #     # res_loss = torch.mean(res_1**2)+ torch.mean(res_2**2)

    #     # resp similarity loss:
    #     recon_resp_loss = torch.mean(torch.abs(fake_resp-resp_msi))

    #     # blurring
    #     # fake_MSI1 = self.blurlayer(fake_MSI)
    #     # MSI2 = self.blurlayer(MSI2)
    #     fake_MSI1 = fake_MSI
    #     # domain loss(1) MSI loss:
    #     fakefeain = torch.cat([fake_MSI1,MSI2],dim=1)
    #     MSI_pred = self.disrgb(fakefeain)
    #     MSI_dis_loss = MSI_pred.mean().reshape(1)*(-1)
    #     # All1 = torch.ones_like(MSI_pred,requires_grad=False).cuda()
    #     # loss_MSI = torch.mean(F.binary_cross_entropy(F.sigmoid(MSI_pred), All1))

    #     # loss_MSI = 

    #     # domain loss(2) HSI loss:
    #     HSI_pred = self.dishsi(fake_HSI)
    #     fake_HSIU = fake_HSI.unfold(2,8,8).unfold(3,8,8)
    #     fake_HSIU = fake_HSIU.permute([0,4,5,1,2,3])
    #     [B,s1,s2,c,h,w] = fake_HSIU.shape
    #     fake_HSIU = fake_HSIU.reshape([B*s1*s2,c,h,w])

    #     HSI_fake_pred = self.dishsilite(fake_HSIU)
    #     HSI_fake_pred = HSI_fake_pred.mean().reshape(1)*(-1)
    #     # HSI_dis_loss = HSI_pred.mean().reshape(1)*(-1)
    #     # All1 = torch.ones_like(HSI_pred,requires_grad=False).cuda()
    #     # loss_HSI = torch.mean(F.binary_cross_entropy(F.sigmoid(HSI_pred), All1))
    #     smooth1 = self.Third_order(fake_resp)
    #     smooth2 = self.Third_order(resp_msi)
    #     reps_loss1 = self.first_order(fake_resp)
    #     reps_loss2 = self.first_order(resp_msi)
    #     # print('Gen Loss: [HSI recon loss:{}, MSI recon loss:{}, resp recon loss:{}], [MSI self loss:{}, HSI self loss:{}]'.format(recon_HSI_loss,
    #     #                                                              recon_MSI_loss, recon_resp_loss, loss_MSI, loss_HSI))
    #     # loss = recon_HSI_loss + recon_MSI_loss + recon_resp_loss + torch.mean(torch.abs(reps_loss1)) + torch.mean(torch.abs(reps_loss2)) + HSI_dis_loss + MSI_dis_loss + HSI_fake_pred
    #     loss = recon_resp_loss + torch.mean(torch.abs(reps_loss1))*0.01 + torch.mean(torch.abs(reps_loss2))*0.01  + MSI_dis_loss*10 + torch.mean(smooth1**2)*5 + torch.mean(smooth2**2)*5

    #     # print(loss)
    #     if self.iter %5 ==0:
    #         print('RspGen Loss :[total loss:{:.4f}], [resp recon loss:{:.4f}], [MSI adversarial loss:{:.4f}]'.format(loss.item(),
    #                                     recon_resp_loss.item(), MSI_dis_loss.item()*(-1)))
    #     return loss #, MSI_pred , HSI_pred.mean().reshape(1)

    def Cal_generatorloss(self,MSI,HSI,rank):
        real_label = 1.
        fake_label = 0.
        one = torch.FloatTensor([1]).to(MSI.device)
        mone = one * -1
        mone.to(one.device)
        b_size = HSI.size(0)
        real_label = torch.full((b_size,), real_label, dtype=torch.float, device=HSI.device)
        fake_label = torch.full((b_size,), fake_label, dtype=torch.float, device=HSI.device)

        # MSI --> Resp
        resp_msi =  self.respG(MSI)

        # MSI + resp --> HSI
        fake_HSI,res_1 = self.hsiG(MSI,resp_msi)

        res_loss = torch.mean(res_1**2)

        # domain loss(2) HSI loss:
        [B,C,H,W] = fake_HSI.shape
        # fea = fake_HSI.permute([0,2,3,1])
        fea = fake_HSI.reshape([B,C,H,W])
        smooth = self.second_order(fea)
        # smooth = self.first_order(fea)
        smooth = torch.mean(torch.abs(smooth))
        # smooth = torch.mean(smooth**2)

        temp1 = fake_HSI.reshape([B,C*H*W])
        
        max_vlaue,_ = temp1.max(dim=1,keepdim=True)

        # fake_HSI = fake_HSI/(max_vlaue.detach()[:,:,None,None]+1e-6)
        fake_HSI = fake_HSI


        HSI_pred = self.dishsi(fake_HSI)
        fake_HSIU = fake_HSI.unfold(2,8,8).unfold(3,8,8)
        fake_HSIU = fake_HSIU.permute([0,4,5,1,2,3])
        [B,s1,s2,c,h,w] = fake_HSIU.shape
        fake_HSIU = fake_HSIU.reshape([B*s1*s2,c,h,w])

        # HSI_fake_pred = self.dishsilite(fake_HSIU)
        # HSI_fake_pred = HSI_fake_pred.mean().reshape(1)*(-1)
        HSI_dis_loss = self.criterion(HSI_pred,real_label)
        # HSI_dis_loss = HSI_pred.mean().reshape(1)*(-1)



        # loss =  HSI_fake_pred + HSI_dis_loss + res_loss*10 + smooth
        loss =  HSI_dis_loss + res_loss*10 + smooth
        # loss = recon_HSI_loss + recon_MSI_loss + res_loss + smooth*10
        # print(loss)
        # self.iter +=1
        if self.iter %5 ==0:
            if rank == 0:
                print('HSIGen Loss :[total loss:{:.4f}], [HSI adversarial loss:{:.4f}], [res loss:{:.4f}]'.format(loss.item(), HSI_pred.mean().item(), res_loss.mean().item()))
        return loss #, MSI_pred , HSI_pred.mean().reshape(1)

    def Third_order(self, feautre):
        # input_1 = feautre[:,1:,:]
        # input_2 = feautre[:,:-1,:]
        # forder = input_1 - input_2
        # forder_1 = forder[:,1:,:]
        # forder_2 = forder[:,:-1,:]
        # sorder = forder_1 - forder_2
        for _ in range(3):
            feautre = self.first_order(feautre)
        return feautre

    def second_order(self, feautre):
        # input_1 = feautre[:,1:,:]
        # input_2 = feautre[:,:-1,:]
        # forder = input_1 - input_2
        # forder_1 = forder[:,1:,:]
        # forder_2 = forder[:,:-1,:]
        # sorder = forder_1 - forder_2
        # return sorder
        for _ in range(2):
            feautre = self.first_order(feautre)
        return feautre

    def first_order(self, feautre):
        input_1 = feautre[:,1:,:]
        input_2 = feautre[:,:-1,:]
        forder = input_1 - input_2
        return forder

    def Cal_discriminatorloss(self,MSI,MSI2,HSI,rank):
        real_label = 1.
        fake_label = 0.
        one = torch.FloatTensor([1]).to(MSI.device)
        mone = one * -1
        mone.to(one.device)
        b_size = HSI.size(0)

        # MSI --> Resp
        resp_msi =  self.respG(MSI)

        # MSI + resp --> HSI
        fake_HSI,_ = self.hsiG(MSI,resp_msi)

        # # HSI + Resp --> fake_MSI
        # fake_MSI = self.RgbG(HSI,resp_msi)

        # # fake_MSI --> fake_resp
        # fake_resp = self.respG(fake_MSI)

        # # fake_MSI + fake_resp --> recon_HSI
        # recon_HSI = self.hsiG(fake_MSI,fake_resp)

        # # fake_HSI + fake_resp --> recon_MSI
        # recon_MSI = self.hsiG(fake_MSI,fake_resp)

        # fake_MSI1 = fake_MSI
        # MSI1 = MSI

        # # # domain loss(1) MSI loss:
        # fakefeain = torch.cat([fake_MSI1,MSI2],dim=1)
        # fake_MSI_pred = self.disrgb(fakefeain)
        # fake_MSI_loss = fake_MSI_pred.mean().reshape(1)
        # realfeain = torch.cat([MSI1,MSI2],dim=1)
        # real_MSI_pred = self.disrgb(realfeain)
        # real_MSI_loss = real_MSI_pred.mean().reshape(1)*(-1)
        [B,C,H,W] = fake_HSI.shape
        real_label = torch.full((B,), real_label, dtype=torch.float, device=HSI.device)
        fake_label = torch.full((B,), fake_label, dtype=torch.float, device=HSI.device)
        # temp1 = fake_HSI.reshape([B,C*H*W])
        # max_vlaue1,_ = temp1.max(dim=1,keepdim=True)
        # temp2 = HSI.reshape([B,C*H*W])
        # max_vlaue2,_ = temp2.max(dim=1,keepdim=True)

        # fake_HSI = fake_HSI/(max_vlaue1.detach()[:,:,None,None]+1e-6)
        # HSI = HSI/(max_vlaue2.detach()[:,:,None,None]+1e-6)
        # fake_HSI = fake_HSI
        # HSI = HSI
        # fake_HSIU = fake_HSI.unfold(2,8,8).unfold(3,8,8)
        # fake_HSIU = fake_HSIU.permute([0,4,5,1,2,3])
        # [B,s1,s2,c,h,w] = fake_HSIU.shape
        # fake_HSIU = fake_HSIU.reshape([B*s1*s2,c,h,w])

        # fake_HSIU_pred = self.dishsilite(fake_HSIU)
        # fake_HSIU_pred = fake_HSIU_pred.mean().reshape(1)

        # HSIU = HSI.unfold(2,8,8).unfold(3,8,8)
        # HSIU = HSIU.permute([0,4,5,1,2,3])
        # [B,s1,s2,c,h,w] = HSIU.shape
        # HSIU = HSIU.reshape([B*s1*s2,c,h,w])

        # HSIU_pred = self.dishsilite(HSIU)
        # HSIU_pred = HSIU_pred.mean().reshape(1)*(-1)
        
        # # domain loss(2) HSI loss:
        fake_HSI_pred = self.dishsi(fake_HSI)
        fake_HSI_loss = self.criterion(fake_HSI_pred,fake_label)
        # fake_HSI_pred.mean().reshape(1).backward(one)
        real_HSI_pred = self.dishsi(HSI)
        real_HSI_loss = self.criterion(real_HSI_pred,real_label)
        GP_loss = self.calc_gradient_penalty(self.dishsi, HSI.detach(), fake_HSI.detach(), center=0, alpha=None, LAMBDA=1, device=real_HSI_pred.device)
        # real_HSI_loss = real_HSI_pred.mean().reshape(1)*(-1)
        # real_HSI_pred.mean().reshape(1).backward(mone)

        # loss_MSI =self.Cal_gradient_penalty(self.disrgb,realfeain,fakefeain)
        # loss_MSI =self.Cal_gradient_penalty(self.disrgb,MSI1,fake_MSI1)
        # loss_HSI =self.Cal_gradient_penalty(self.dishsi,HSI,fake_HSI)
        # loss_HSIU =self.Cal_gradient_penalty(self.dishsilite,HSIU,fake_HSIU)

        # loss = loss_MSI + loss_HSI + fake_MSI_loss + real_MSI_loss + fake_HSI_loss + real_HSI_loss + fake_HSIU_pred + HSIU_pred + loss_HSIU
        # loss = loss_MSI + loss_HSI + (fake_MSI_loss + real_MSI_loss) + (fake_HSI_loss + real_HSI_loss)*1e-3 + (fake_HSIU_pred + HSIU_pred)*1e-2 + loss_HSIU
        # loss = loss_HSI + (fake_HSI_loss + real_HSI_loss) + (fake_HSIU_pred + HSIU_pred) + loss_HSIU
        loss = fake_HSI_loss + real_HSI_loss + GP_loss
        # loss = loss_MSI + fake_MSI_loss + real_MSI_loss + fake_HSIU_pred + HSIU_pred + loss_HSIU + fake_HSI_loss + real_HSI_loss
        self.iter +=1
        if self.iter %5 ==0:
            if rank == 0:
                print('Dis Loss :[total loss:{:.4f}], [HSI  fake loss:{:.2f}, real loss:{:.2f}]'.format(loss.item(),fake_HSI_loss.item(),real_HSI_loss.item()))
        return  loss

    def update_generator(self,MSI,MSI2,HSI,rank):
        one = torch.FloatTensor([1]).to(MSI.device)
        mone = one * -1
        mone.to(one.device)
        self.gen_optimizer.zero_grad()
        gen_loss = self.Cal_generatorloss(MSI,HSI,rank)
        gen_loss.backward()
        gen_loss1 = gen_loss.item()
        self.gen_optimizer.step()

        # self.gen_optimizer2.zero_grad()
        # gen_loss = self.Cal_generatorloss(MSI,HSI)
        # gen_loss.backward()
        # gen_loss2 = gen_loss.item()
        # self.gen_optimizer2.step()
        return gen_loss1

    def update_discriminator(self,MSI,MSI2,HSI,rank):
        self.dis_optimizer.zero_grad()
        dis_loss = self.Cal_discriminatorloss(MSI,MSI2,HSI,rank)
        dis_loss.backward()
        dis_loss1 = dis_loss.item()

        # print('shape of MSI loss:{}'.format(real_MSI_pred.shape))

        # real_MSI_pred.backward(mone)
        # real_HSI_pred.backward(mone)
        # fake_MSI_pred.backward(one)
        # fake_HSI_pred.backward(one)
        self.dis_optimizer.step()
        return dis_loss1

    def forward(self,MSI):
        # MSI --> Resp
        resp_msi =  self.respG(MSI)

        # MSI + resp --> HSI
        fake_HSI,res_ = self.hsiG(MSI,resp_msi)

        return fake_HSI , resp_msi, res_
