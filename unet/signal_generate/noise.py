# -*- coding: utf-8 -*-

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from scipy.fftpack import fft, ifft
# from generate_signal import generate_ori_signal, projection, delete_f0
# from pathlib import Path 
# import sys
# sys.path.append(str(Path(__file__).resolve().parents[1]))
# from func_tools.generate_signal_new import generate_ori_signal, projection, delete_f0

def calculate_snr(signal1, signal2):
    #signal2为target信号
    # signal_1 = np.array(signal1, dtype=np.float64)
    # signal_2 = np.array(signal2, dtype=np.float64)
    diff = signal1 - signal2
    snr = 10 * np.log10((np.sum(signal2**2)) / ((np.sum(diff**2))))
    return snr


def Harmonic_interference_fixed_phase(signal, sir):
    u_all = np.tile(signal, 5000)
    fft_sig = fft(u_all)

    sir = 10 ** (sir/20)
    fft_harmonic = fft_sig / sir
    fft_noise = fft_sig + fft_harmonic
    signal_with_interference = ifft(fft_noise).real
    return signal_with_interference[:200]

def Harmonic_interference_varied_phase(signal, sir):
    fs = 1e6
    u_all = np.tile(signal, 5000)
    u_fft = fft(u_all)
    u_abs = np.abs(u_fft)
    u_normal = u_abs / len(u_all)
    u_half = u_normal[range(int(len(u_all)/ 2))]
    sir = 10 ** (sir/20)
    gamma =  2 * u_half / sir

    noise = np.zeros(200)
    for i in range(0, len(u_half)):
        phase = 2 * np.pi * np.random.random()
        if i % 5000 == 0:      #在倍频处添加噪声
            for j in range(200):
                noise[j] = noise[j] + gamma[i] * math.sin(2 * math.pi * i * (j+1)/fs + phase)
    signal_with_interference = signal + noise
    return signal_with_interference

def gaussian_noise(signal, SNR):
    noise = np.random.randn(len(signal)) 	#产生N(0,1)噪声数据
    noise = noise - np.mean(noise) 								#均值为0
    signal_power = np.linalg.norm( signal - signal.mean() )**2 / signal.size	#此处是信号的std**2
    noise_variance = signal_power/np.power(10,(SNR/10))         #此处是噪声的std**2
    noise = (np.sqrt(noise_variance) / np.std(noise) )*noise    ##此处是噪声的std**2
    signal_noise = noise + signal
    return signal_noise
