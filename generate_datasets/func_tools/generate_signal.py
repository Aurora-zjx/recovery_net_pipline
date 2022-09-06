# -*- coding: utf-8 -*-

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.fftpack import fft,ifft




    # drive_field[i] = A * math.sin(w * t[i])
    # d_drive_field[i] = 1 * A * w * math.cos(w * t[i]) 


'''
系统参数
'''
k_B = 1.3806488e-23  # 玻尔兹曼常数
T = 300   # 温度
d = 20e-9  # 粒子直径
miu0 = 4 * math.pi * 1e-7  # 真空渗透率
Ms = 0.6 / miu0   # 球体磁矩
m = Ms * math.pi / 6 * d**3  # 磁粒子磁矩
k =  miu0 * m / (k_B * T)
B1 = -1
gradient_strength = 4/miu0
A = 40e-3/miu0
Xmax = A / gradient_strength
step = 0.2e-3
x = np.arange(-Xmax, Xmax, step)                          
gradient_field = gradient_strength * x 


'''
子函数
'''

def Langevin(H):
    lv = np.zeros(len(H))
    dlv = np.zeros(len(H))
    for i in range(len(H)):
        if abs(H[i]) < 1e-10:
            lv[i] = 0
            dlv[i] = 1/3
        else:
            lv[i] = math.cosh(H[i]) / math.sinh(H[i]) - 1 / H[i]
            dlv[i] = 1 / (H[i]**2) - 1/ (math.sinh(H[i])**2)
    return lv, dlv


#编码产生信号
def generate_ori_signal(Concentration,drive_field,d_drive_field,gradient_field,t,x):
    # print(len(x))
    H = np.zeros((len(x), len(t)))
    dlv = np.zeros((len(x), len(t)))
    lv = np.zeros((len(x), len(t)))
    dM = np.zeros((len(x), len(t)))
    M = np.zeros((len(x), len(t)))
    u = np.zeros(len(t))
    for i in range(len(t)):
        H[:, i] = drive_field[i] - gradient_field
        # print(k * H[:, i])
        lv[:, i], dlv[:, i] = Langevin(k * H[:, i])
        dM[:, i] = m * Concentration * dlv[:, i] * d_drive_field[i]

    u = B1 * np.sum(dM, axis=0) * 1e8
    # print(u)
    return u

#将信号投影到图像域
def projection(u,drive_field,d_drive_field):
   
    Xffp = drive_field[:]/gradient_strength  
    dXffp = d_drive_field[:]/gradient_strength
    dXffp_Amp = []
    for i in range(len(dXffp)):
        dXffp_Amp.append(math.sqrt(dXffp[i]**2))
    normal_u = np.divide(u,dXffp_Amp)     
    Xffp = np.array(Xffp)
    pointx = np.arange(-Xmax, Xmax, step)
    # 未经过速度归一化投影到x轴
    # print(Xffp[0:100])
    # print(normal_u)
    # import pdb;pdb.set_trace()
    ImgTan0 = griddata(Xffp[0:100], u[0:100], pointx, method='nearest')
    ImgTan0_normal = griddata(Xffp[0:100], normal_u[0:100], pointx, method='nearest')
    # if np.max(ImgTan0) != 0:
    #    ImgTan0 = ImgTan0/np.max(ImgTan0) 
    # if np.max(ImgTan0_normal) != 0:
    #    ImgTan0_normal = ImgTan0_normal/np.max(ImgTan0_normal) 

    return ImgTan0, ImgTan0_normal, normal_u, pointx, dXffp_Amp




def delete_f0(ori_signal):
    u_all = np.tile(ori_signal, 5000)
    u_fft = fft(u_all)

    u_fft[5000] = 0
    u_fft[1000000-5000] = 0

    # u_fft[10000] = 0
    # u_fft[1000000-10000] = 0
    # u_fft[15000] = 0
    # u_fft[1000000-15000] = 0

    u_ifft = ifft(u_fft)
    return u_ifft[:200]


