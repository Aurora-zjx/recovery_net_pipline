# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    test_path = '/home/zjx/Documents/net/signal_unet/datasets_small/test_ori.txt'
    test_label_path = '/home/zjx/Documents/net/signal_unet/datasets_small/test_label.txt'
    test_drive_path = '/home/zjx/Documents/net/signal_unet/datasets_small/test_drive.txt'
    test_d_drive_path = '/home/zjx/Documents/net/signal_unet/datasets_small/test_d_drive.txt'
    test_output = 'outputs/pred_signal/test_output_one_loss.txt'

    test_data = np.loadtxt(test_path)
    label_data = np.loadtxt(test_label_path)
    test_drive_data = np.loadtxt(test_drive_path)
    test_d_drive_data = np.loadtxt(test_d_drive_path)
    result = np.loadtxt(test_output)
    for i in range(len(test_data)):
        if i == len(test_data) - 2:
            plt.figure(figsize=(20,20))
            plt.subplot(2,2,1)  
            plt.plot(test_data[i], 'y')
            plt.plot(result[i], 'r')
            plt.plot(label_data[i], 'b')
            plt.title('signal')

            plt.subplot(2,2,2)  
            plt.plot(test_drive_data[i], 'r')
            plt.title('drive')

            plt.subplot(2,2,3)  
            plt.plot(test_d_drive_data[i], 'r')
            plt.title('d_drive')
            plt.savefig("outputs/analyse_res/test_result{}.jpg".format(i))




    