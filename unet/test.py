# -*- coding: utf-8 -*-

import argparse
import numpy as np
import torch
from torch.utils import data
from data.dataset import DatasetFromFolder_test
import matplotlib.pyplot as plt
from utils.average_tools import calc_psnr


def tester(args):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    test_data = DatasetFromFolder_test(args.test_path, args.test_label_path)
    test_dataLoader = data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
    model = torch.load(args.model)
    model = model.to(device)
    f = open(args.test_output, 'w')

    snr_list = []
    for step, (test_data, test_label) in enumerate(test_dataLoader):
        import pdb;pdb.set_trace()
        test_data, test_label= test_data.to(device), test_label.to(device)
        test_data = test_data.unsqueeze(1)
        test_label = test_label.unsqueeze(1)
        with torch.no_grad():
            test_result = model(test_data)
        snr = calc_psnr(test_result, test_label)
        snr_list.append(snr)
    # import pdb;pdb.set_trace()
        test_result = test_result.reshape(test_result.size()[1], -1)
        test_result = test_result.cpu().numpy()
        test_result = test_result.tolist()
        result = test_result[0]

        for j in range(len(result)):
            f.write(str(result[j]) + ' ')
        f.write('\n')

    print('SNR_list:',snr_list)
    print('Avg_SNR:',sum(snr_list)/len(snr_list))
    
    f.close()


    


def load_data_visual(args):
    test_data = np.loadtxt(args.test_path)
    label_data = np.loadtxt(args.test_label_path)
    result = np.loadtxt(args.test_output)
    for i in range(len(test_data)):
        # plt.switch_backend('agg')
        plt.figure()
        plt.plot(test_data[i], 'y')
        plt.plot(result[i], 'r')
        plt.plot(label_data[i], 'b')
        plt.savefig("outputs/output_img_one_loss/test_result{}.jpg".format(i))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, required=False, default="/home/zjx/Documents/net/signal_unet/datasets_small/test_ori.txt")
    parser.add_argument('--test_label_path', type=str, required=False, default="/home/zjx/Documents/net/signal_unet/datasets_small/test_label.txt")
    parser.add_argument('--model', type=str, default="checkpoints/one_loss/best.pth")
    parser.add_argument('--test_output', type=str, default="outputs/pred_signal/test_output_one_loss.txt")
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    tester(args)
    load_data_visual(args)

    





