# -*- coding: utf-8 -*-
# @Time : 2022.9.5
# @Author : zjx
# @Function : generate recovery signal and project image

import numpy as np
import random
import math
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[0]))
from signal_generate.ConstantList import *
from signal_generate.generate_1D import generate_ori_signal_with_FOV, delete_f0, projection_with_FOV, sigle_projection
from signal_generate.noise import Harmonic_interference_fixed_phase, Harmonic_interference_varied_phase, gaussian_noise





def generate_random_concentration():
    Concentration = np.zeros(100) # 粒子分布浓度  初始化
    Concentration_num = random.randint(1,6)
    ori_serial = range(0,100)
    already_have_list = []
    Concentration_i = 0
    while(1):  
        C_center = random.choice(ori_serial)
        if C_center not in already_have_list:
            left = max(0,C_center - random.randint(0,8))
            right = min(100,C_center + random.randint(0,8))
            Concentration[left:right] = random.randint(1,2)
            new_list = range(left,right)
            already_have_list.extend(new_list)
            Concentration_i = Concentration_i + 1
        if Concentration_i == Concentration_num:
            break
    return Concentration


def get_net_input(Concentration,FOV,SNR,drive_field,d_drive_field):
    ## 1.在FOV内生成原始信号
    ori_signal = generate_ori_signal_with_FOV(Concentration,FOV,t,drive_field,d_drive_field,gradient_strength,m,B1,miu0 * m / (k_B * T))
    ori_signal = ori_signal * 1000
    ## 2.给生成信号添加高斯噪声
    ori_signal_with_noise = gaussian_noise(ori_signal, SNR)
    ## 3.丢失基频信息
    u_ifft = delete_f0(ori_signal_with_noise)

    # visual_two_signal(Concentration,ori_signal,ori_signal_with_noise,u_ifft,drive_field,d_drive_field)

    return ori_signal,ori_signal_with_noise,u_ifft

def net_recovery(input_signal):
    input_signal = input_signal.real
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model_path = '/home/zjx/recovery_net/unet/checkpoints/one_loss/best.pth'
    try:
        model = torch.load(model_path)
        model = model.to(device)
        input_signal = input_signal.to(device)
        input_signal = input_signal.unsqueeze(1)
        with torch.no_grad():
            test_result = model(input_signal)
        test_result = test_result.reshape(test_result.size()[1], -1)
        test_result = test_result.cpu().numpy()
        test_result = test_result.tolist()
        result = test_result[0]
    except:
        model = torch.load(model_path,map_location='cpu')
        device = torch.device('cpu')
        model = model.to(device)
        input_signal = torch.from_numpy(input_signal)
        input_signal = input_signal.unsqueeze(0)
        input_signal = input_signal.unsqueeze(1)
        # for name, param in model.named_parameters():
        #     print(name,'-->',param.type(),'-->',param.dtype,'-->',param.shape)

        # import pdb;pdb.set_trace()
        
        input_signal = torch.tensor(input_signal, dtype=torch.float32)
        with torch.no_grad():
            test_result = model(input_signal)
    
    
    test_result = test_result.cpu().numpy()
    test_result = test_result.tolist()
    result = test_result[0][0]

    # import pdb;pdb.set_trace()

    return result

if __name__ == "__main__":

    save_Path = '/home/zjx/recovery_net/unet/outputs/'

    ## 1.生成随机的粒子分布
    Concentration = generate_random_concentration()
    ## 2.生成随机的驱动场
    drive_field = np.zeros(len(t))   #drive field(驱动场)
    d_drive_field = np.zeros(len(t))
    phase = random.randint(0,360)
    # phase = 0
    for k in range(len(t)):
        drive_field[k] = A * math.cos(w * t[k] + phase / 180 * np.pi)
        d_drive_field[k] = -1 * A * w * math.sin(w * t[k] + phase / 180 * np.pi) 

    ## 3.定义FOV大小及pFOV 及噪声强度
    FOV = np.arange(-Xmax, Xmax, step)
    SNR_low = 15
    SNR_high = 20
    SNR = random.randint(SNR_low, SNR_high)
    
    ## 4.生成net的输入信号
    ori_signal,ori_signal_with_noise,input_signal= get_net_input(Concentration,FOV,SNR,drive_field,d_drive_field)
    output_signal = net_recovery(input_signal)

    p2 = plt.figure(figsize=(17,27))
    plt.subplot(4,2,1)
    plt.xlabel("x")
    plt.ylabel("c")
    plt.plot(Concentration, 'r')
    plt.title('concentration distribution')

    plt.subplot(4,2,2)   
    plt.xlabel("t")
    plt.ylabel("signal")
    plt.plot(t,ori_signal,'r')
    plt.plot(t,input_signal,'g')
    plt.plot(t,output_signal,'k')
    plt.title('signal_t')

    

    ## 5.将恢复出的信号投影到图像域进行显示
    ImgTan0_normal, ImgTan0_normal_recovery,normal_u, normal_u_recovery, pointx, dXffp_Amp, before_normal_u, before_normal_uifft = projection_with_FOV(ori_signal,output_signal,FOV,drive_field,d_drive_field,gradient_strength)
    # import pdb;pdb.set_trace()
    # ImgTan0_normal, normal_u, pointx, dXffp_Amp = sigle_projection(ori_signal,FOV,drive_field,d_drive_field,gradient_strength)
    # ImgTan0_normal_noise_withoutf0, normal_u_noise_withoutf0, pointx_noise_withoutf0, dXffp_Amp_noise_withoutf0 = sigle_projection(input_signal.real,FOV,drive_field,d_drive_field,gradient_strength)
    # ImgTan0_normal_net_recovery, normal_u_net_recovery, pointx_net_recovery, dXffp_Amp_net_recovery = sigle_projection(output_signal,FOV,drive_field,d_drive_field,gradient_strength)

    
    plt.subplot(4,2,3)   
    plt.xlabel("t")
    plt.ylabel("drive_field")
    plt.plot(t, drive_field,'r')
    plt.title('drive_field')

    plt.subplot(4,2,4)   
    plt.xlabel("t")
    plt.ylabel("dXffp_Amp")
    plt.plot(t, dXffp_Amp,'r')
    plt.title('dXffp_Amp')

    plt.subplot(4,2,5)   
    plt.xlabel("t")
    plt.ylabel("normal_u")
    plt.plot(t, before_normal_u,'r')
    plt.plot(t, before_normal_uifft,'k')
    # plt.plot(t, normal_u - normal_u_net_recovery,'b')
    plt.title('normal_u_before_shift')

    plt.subplot(4,2,6)   
    plt.xlabel("t")
    plt.ylabel("normal_u")
    plt.plot(t, normal_u,'r')
    plt.plot(t, normal_u_recovery,'k')
    # plt.plot(t, normal_u - normal_u_net_recovery,'b')
    plt.title('normal_u_after_shift')

    plt.subplot(4,2,7)   
    plt.xlabel("x")
    plt.ylabel("ImgTan0_normal")
    plt.plot(pointx[1:], ImgTan0_normal[1:].real,'r')
    # plt.plot(pointx[1:], ImgTan0_normal_noise_withoutf0[1:].real,'g')
    plt.plot(pointx[1:], ImgTan0_normal_recovery[1:].real,'k')
    # plt.plot(pointx[1:], ImgTan0_normal[1:].real - ImgTan0_normal_new[1:].real,'b')
    plt.title('ImgTan0_normal')

    plt.savefig(save_Path +'input_signal.jpg')


    # ## 6.pFOV扫描 加噪 去噪 pFOV恢复
    # pFOV_scan(Concentration,SNR,FOV,t,drive_field,d_drive_field,gradient_strength,m,B1,miu0 * m / (k_B * T))