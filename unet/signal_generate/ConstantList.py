# -*- coding: utf-8 -*-
import numpy as np
import random
import math


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