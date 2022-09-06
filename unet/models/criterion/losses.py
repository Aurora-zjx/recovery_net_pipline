# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from scipy.interpolate import griddata



#恢复后的信号投影到图像域相差一个DC项的偏移
class recovery_loss(nn.Module):
    def __init__(self):
        super(recovery_loss,self).__init__()

    def forward(self, preds, target, Xffp, dXffp_Amp, pointx):
        
        preds = preds.reshape(preds.size()[0],preds.size()[-1])

        #将信号进行ffp速度归一化
        normal_u = torch.div(target,dXffp_Amp)
        normal_u_uifft = torch.div(preds,dXffp_Amp)

        # 因为相位的不同 不能直接拿0:100信号去投影
        # 循环移位调整信号 
        # 同时batch内每个样本信号的需要的移位不同，因此需要for循环来逐个计算
        batch_losses = []
        for i in range(target.shape[0]):
            normal_u_i = normal_u[i].cpu().numpy().tolist()
            normal_u_uifft_i = normal_u_uifft[i].cpu().detach().numpy().tolist()
            Xffp_i = Xffp[i].cpu().numpy().tolist()
        
        
            if normal_u_i[-1] < 0 and normal_u_i[0] < 0:  #循环左移
                while(normal_u_i[len(normal_u_i)-1] < 0):
                    shift_u = normal_u_i[1:] + normal_u_i[:1]
                    normal_u_i = shift_u

                    shift_uifft = normal_u_uifft_i[1:] + normal_u_uifft_i[:1]
                    normal_u_uifft_i = shift_uifft

                    shift_Xffp = Xffp_i[1:] + Xffp_i[:1]
                    Xffp_i = shift_Xffp
            else:                    #循环右移
                while(normal_u_i[len(normal_u_i)-1]>0):
                    shift_u = normal_u_i[len(normal_u_i)-1:] + normal_u_i[:len(normal_u_i)-1]
                    normal_u_i = shift_u

                    shift_uifft = normal_u_uifft_i[len(normal_u_uifft_i)-1:] + normal_u_uifft_i[:len(normal_u_uifft_i)-1]
                    normal_u_uifft_i = shift_uifft

                    shift_Xffp = Xffp_i[len(Xffp_i)-1:] + Xffp_i[:len(Xffp_i)-1]
                    Xffp_i = shift_Xffp

        
            normal_u_i = np.array(normal_u_i)  
            normal_u_uifft_i = np.array(normal_u_uifft_i)  
            Xffp_i = np.array(Xffp_i)
        
            # 得到batch内每个样本投影到图像域的表示
            ImgTan0_normal = griddata(Xffp_i[0:100], normal_u_i[0:100], pointx, method='nearest')
            ImgTan0_normal_uifft = griddata(Xffp_i[0:100], normal_u_uifft_i[0:100], pointx, method='nearest')
            
            diff = ImgTan0_normal -ImgTan0_normal_uifft
            # print(np.var(diff))
            loss = np.var(diff[1:])
            batch_losses.append(loss)
            if loss > 1000:
                import pdb;pdb.set_trace()

        #计算batch内的average_loss
        batch_avg_loss = sum(batch_losses)/len(batch_losses)
        # print('batch_losses:',batch_losses)
        print('batch_avg_loss:',batch_avg_loss)
        return batch_avg_loss



