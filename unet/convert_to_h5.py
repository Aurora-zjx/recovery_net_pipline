import glob
import h5py
import cv2
import numpy as np


def prepare_data(name):
    ori_dir = '/home/zjx/recovery_net/generate_datasets/datasets/' + name + '_ori.txt'
    GT_dir = '/home/zjx/recovery_net/generate_datasets/datasets/' + name + '_label.txt'
    output_dir = '/home/zjx/recovery_net/unet/datasets_small/' + name + '.h5'

    h5_file = h5py.File(output_dir, 'w')

    ori_signal = []
    label_signal = []

    with open(ori_dir, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            line_list = line.split(' ')
            line_list = [float(num) for num in line_list[:-1]]
            ori_signal.append(line_list)
    
    with open(GT_dir, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            line_list = line.split(' ')
            line_list = [float(num) for num in line_list[:-1]]
            label_signal.append(line_list)

    

    ori_signal = np.array(ori_signal)
    label_signal = np.array(label_signal)


    print(ori_signal.shape)
    print(label_signal.shape)

    h5_file.create_dataset('ori', data=ori_signal)
    h5_file.create_dataset('label', data=label_signal)
    h5_file.close()





if __name__ == '__main__':
    prepare_data('train')
    prepare_data('val')
    





    