# -*- coding: utf-8 -*-

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from multiprocessing import Pool
from matplotlib.pylab import mpl
from scipy.interpolate import griddata
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[0]))
from func_tools.generate_signal import generate_ori_signal, delete_f0
from func_tools.noise import Harmonic_interference_fixed_phase, Harmonic_interference_varied_phase, gaussian_noise

#region
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


f_drive = 5e3
signal_T = 1/f_drive
w = 2 * math.pi * f_drive


fs = 1e6 # 采样频率 频率越高，周期内点越密集
t = np.arange(1/fs, 1/f_drive, 1/fs) # 采样时间 [1/fs, 1/f], 间隔 1/fs   一个周期
drive_field = np.zeros(len(t))   #drive field(驱动场)
d_drive_field = np.zeros(len(t))
#endregion


def create_task_train(i):
    print('train:',i)
    generate_datasets('train',i)
def create_task_val(i):
    print('val:',i)
    generate_datasets('val',i)
def create_task_test(i):
    print('test:',i)
    generate_datasets('test',i)

def multicore_pool_train():
    pool = Pool(processes=20)
    pool.map(create_task_train,range(train_num))

def multicore_pool_val():
    pool = Pool(processes=20)
    pool.map(create_task_val,range(val_num))

def multicore_pool_test():
    pool = Pool(processes=20)
    pool.map(create_task_test,range(test_num))


def generate_datasets(file_name,i):
    '''
    给定粒子的浓度分布范围
    '''
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

    '''
    产生原始信号+去基频信号
    '''
    #产生驱动场
    phase = random.randint(0,360)
    # phase = 0
    for k in range(len(t)):
        drive_field[k] = A * math.cos(w * t[k] + phase / 180 * np.pi)
        d_drive_field[k] = -1 * A * w * math.sin(w * t[k] + phase / 180 * np.pi) 

    

    #根据给定的驱动场信号+粒子分布获取接收信号
    ori_signal = generate_ori_signal(Concentration,drive_field,d_drive_field,gradient_field,t,x)
    ori_signal = ori_signal * 1000
    

    ## 添加噪声及干扰
    sir_low = 5
    sir_high = 14
    sir = random.randint(sir_low, sir_high)
    snr = random.randint(sir_low, sir_high)

    noise = gaussian_noise(ori_signal, snr)
    interference = Harmonic_interference_varied_phase(ori_signal, sir)

    signal = ori_signal + noise + interference
    # 原信号的单边频率轴，幅值，去掉基频后的信号，去掉基频后的幅值，原信号的相位，去掉基频后的相位
    u_ifft = delete_f0(signal)


    ##生成的信号数据集是归一化之前的信号
    f1 = open(save_Path + file_name + '_ori' +'.txt', 'a')
    f2 = open(save_Path + file_name + '_label' +'.txt', 'a')
    f3 = open(save_Path + file_name + '_drive' +'.txt', 'a')
    f4 = open(save_Path + file_name + '_d_drive' +'.txt', 'a')
    for j in range(len(ori_signal)):
        f1.write(str(u_ifft[j].real) +' ')
        f2.write(str(ori_signal[j]) + ' ')
        f3.write(str(drive_field[j]) + ' ')
        f4.write(str(d_drive_field[j]) + ' ')
    f1.write('\n')
    f2.write('\n')
    f3.write('\n')
    f4.write('\n')
    f1.close()
    f2.close()
    f3.close()
    f4.close()


    #将保存在test中的数据进行可视化
    if file_name == 'test':
        save_file = '/home/zjx/recovery_net/generate_datasets/results_img/'
        plt.figure(figsize=(25,25))
        #绘制原始粒子分布  接收信号+滤除基频后的  ffp的轨迹及amp
        plt.subplot(3,2,1)
        plt.xlabel("x")
        plt.ylabel("c")
        plt.plot(Concentration, 'r')
        plt.title('concentration distribution')
        
        plt.subplot(3,2,2)
        plt.xlabel("t")
        plt.ylabel("ori_signal")
        plt.plot(t[:100],ori_signal[:100], 'r')
        plt.plot(t[:100],u_ifft[:100].real, 'g')
        plt.title('ori_signal')

        Xffp = drive_field[:]/gradient_strength  
        dXffp = d_drive_field[:]/gradient_strength
        dXffp_Amp = []
        for j in range(len(dXffp)):
            dXffp_Amp.append(math.sqrt(dXffp[j]**2))

        Xffp = np.array(Xffp)
        pointx = np.arange(-Xmax, Xmax, step)


        normal_u = np.divide(ori_signal,dXffp_Amp)   #ori_signal 200   normal_u 100
        normal_u_uifft = np.divide(u_ifft,dXffp_Amp)   
        
        # 因为相位的不同 不能直接拿0:100信号去投影
        #循环移位调整信号 
        normal_u = normal_u.tolist()
        normal_u_uifft = normal_u_uifft.tolist()
        Xffp = Xffp.tolist()
        if normal_u[-1] < 0 and normal_u[0] < 0:  #循环左移
            while(normal_u[len(normal_u)-1] < 0):
                shift_u = normal_u[1:] + normal_u[:1]
                normal_u = shift_u

                shift_uifft = normal_u_uifft[1:] + normal_u_uifft[:1]
                normal_u_uifft = shift_uifft

                shift_Xffp = Xffp[1:] + Xffp[:1]
                Xffp = shift_Xffp
        else:                    #循环右移
            while(normal_u[len(normal_u)-1]>0):
                shift_u = normal_u[len(normal_u)-1:] + normal_u[:len(normal_u)-1]
                normal_u = shift_u

                shift_uifft = normal_u_uifft[len(normal_u_uifft)-1:] + normal_u_uifft[:len(normal_u_uifft)-1]
                normal_u_uifft = shift_uifft

                shift_Xffp = Xffp[len(Xffp)-1:] + Xffp[:len(Xffp)-1]
                Xffp = shift_Xffp
        
        normal_u = np.array(normal_u)  
        normal_u_uifft = np.array(normal_u_uifft)  
        Xffp = np.array(Xffp)
        
            
        ImgTan0_normal = griddata(Xffp[0:100], normal_u[0:100], pointx, method='nearest')
        ImgTan0_normal_uifft = griddata(Xffp[0:100], normal_u_uifft[0:100], pointx, method='nearest')

        plt.subplot(3,2,3)
        plt.xlabel("t")
        plt.ylabel("drive_field")   #产生激励信号的驱动场
        plt.plot(t,drive_field, 'g')
        plt.title('drive_field')

        plt.subplot(3,2,4)
        plt.xlabel("t")
        plt.ylabel("dXffp_Amp")    #归一化所除的ffp的速度
        plt.plot(t,dXffp_Amp, 'r')
        plt.title('dXffp_Amp')

        #验证是否可以恢复出原始的粒子分布图

        plt.subplot(3,2,5)
        plt.xlabel("t")
        plt.ylabel("normal_u")    #归一化所除的ffp的速度
        plt.plot(t,normal_u, 'r')
        plt.plot(t,normal_u_uifft, 'g')
        plt.plot(t,normal_u - normal_u_uifft, 'b')
        plt.title('normal_u')

        plt.subplot(3,2,6)
        plt.xlabel("pointx")
        plt.ylabel("ImgTan0_normal")    #归一化所除的ffp的速度
        plt.plot(pointx[1:],ImgTan0_normal[1:], 'r')
        plt.plot(pointx[1:],ImgTan0_normal_uifft[1:], 'g')
        plt.plot(pointx[1:],ImgTan0_normal[1:] - ImgTan0_normal_uifft[1:], 'b')
        plt.title('ImgTan0_normal')


        plt.savefig(save_file + str(i) +'_test.jpg')



if __name__ == "__main__":

    save_Path = '/home/zjx/recovery_net/generate_datasets/datasets/'
    train_num = 10000
    val_num = 1600
    test_num = 20

    multicore_pool_train()
    multicore_pool_val()
    multicore_pool_test()
    
    


   
