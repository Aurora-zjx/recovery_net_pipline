import torch
import numpy as np


def calc_psnr(ori_signal, noise_signal):
    ori_signal = ori_signal.cpu().numpy()
    noise_signal = noise_signal.cpu().numpy()
    Ps = ( np.linalg.norm(ori_signal - ori_signal.mean()) )**2          # signal power
    Pn = ( np.linalg.norm(ori_signal - noise_signal ) )**2          # noise power
    snr = 10*np.log10(Ps/Pn)
    return snr


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count