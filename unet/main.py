# -*- coding: utf-8 -*-
import argparse
import torch
import torch.backends.cudnn as cudnn
from data.dataset import DatasetFromFolder, DatasetFromH5DY
from models.unet import UNet1d
from models.criterion.losses import recovery_loss
from os.path import join
from torch.utils import data
from os import listdir
import numpy as np
import math
import matplotlib.pyplot as plt
# from math import log10
from tqdm import tqdm
import copy
from utils.logging_config import get_logger
from utils.average_tools import AverageMeter,calc_psnr


class Trainer(object):
    def __init__(self, args, train_data, val_data):
        super(Trainer, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda:3' if self.CUDA else 'cpu')
        self.lr = args.lr
        self.epochs = args.epochs
        self.train_data = train_data
        self.val_data = val_data
        self.train_dataLoader = data.DataLoader(dataset=train_data, batch_size=args.batch_size,num_workers=8,pin_memory=True, shuffle=True) 
        self.val_dataLoader = data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.output_dir = args.output_dir

    def build_model(self):
        self.model = UNet1d(
            in_channels=1,
            out_channels=1,
            num_stages=4,
            initial_num_channels=8,
            norm='bn',
            non_lin='relu',
            kernel_size=3
        ).to(self.device)
        self.criterion0 = torch.nn.L1Loss(size_average=True)
        self.criterion1 = recovery_loss()

        #################MPI参数定义
        self.miu0 = 4 * math.pi * 1e-7  # 真空渗透率
        self.gradient_strength = 4/self.miu0
        self.A = 40e-3/self.miu0
        self.Xmax = self.A / self.gradient_strength
        self.step = 0.2e-3
        self.pointx = np.arange(-self.Xmax, self.Xmax, self.step)
        #################
        
        torch.manual_seed(self.seed)
        
        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion0.cuda()
            self.criterion1.cuda()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        self.best_epoch = 0
        self.best_psnr = 0.0
        self.best_weights = copy.deepcopy(model)

        self.logger = get_logger('signal_train.log')
        self.logger.info('model_init_finished!')
        
 
    def train(self,epoch):
        self.model.train()
        epoch_losses = AverageMeter()
        epoch_losses0 = AverageMeter()
        epoch_losses1 = AverageMeter()
        # print('epoch_losses:',epoch_losses.avg)
        with tqdm(total = (len(self.train_data) - len(self.train_data) % self.batch_size)) as t:
            t.set_description('epoch:{}/{}'.format(epoch,self.epochs ))
            # for batch_num, (data, target, drive, d_drive) in enumerate((self.train_dataLoader)):
            for batch_num, (data, target) in enumerate((self.train_dataLoader)):
                # import pdb;pdb.set_trace()
                # data, target, drive, d_drive = data.to(self.device), target.to(self.device), drive.to(self.device), d_drive.to(self.device)
                data, target = data.to(self.device), target.to(self.device)
                # print(self.device)
                data = data.unsqueeze(1)  

                ####################对相关参数进行求解
                # Xffp = torch.div(drive, self.gradient_strength)
                # dXffp = torch.div(d_drive, self.gradient_strength)
                # dXffp_Amp = dXffp ** 2
                # dXffp_Amp = dXffp_Amp ** 0.5
                
                ####################

                # loss1 = self.criterion1(self.model(data), target, Xffp, dXffp_Amp, self.pointx)
                target = target.unsqueeze(1)
                # print('1')
                data = data.float()
                loss0 = self.criterion0(self.model(data), target)  #计算的是每个batch的average loss
                # print('2')
                # loss = loss0 + loss1 * 1e5
                loss = loss0
                
                # import pdb;pdb.set_trace()

                epoch_losses.update(loss.item(),len(data))
                epoch_losses0.update(loss0.item(),len(data))
                # epoch_losses1.update(loss1.item() * 1e5,len(data))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(data))
                
            print("Average Loss: {:.4f}".format(epoch_losses.avg))
            self.logger.info("Epoch:[{}/{}]\t Average Loss: {:.4f}".format(epoch,self.epochs,epoch_losses.avg))
            self.loss_list.append(epoch_losses.avg)
            self.loss0_list.append(epoch_losses0.avg)
            # self.loss1_list.append(epoch_losses1.avg)

    def val(self,epoch):
        self.model.eval()
        epoch_psnr = AverageMeter()
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.val_dataLoader):
                data, target = data.to(self.device), target.to(self.device)
                data = data.unsqueeze(1)
                target = target.unsqueeze(1)
                # drive = drive.unsqueeze(1)
                # d_drive = d_drive.unsqueeze(1)
                data = data.float()
                
                preds = self.model(data)
                snr = calc_psnr(preds, target)
                epoch_psnr.update(snr,len(data))
        print("  Average SNR: {:.4f} dB".format(epoch_psnr.avg))
        self.logger.info("  Average SNR: {:.4f} dB".format(epoch_psnr.avg))
        self.psnr_list.append(epoch_psnr.avg)

        if epoch_psnr.avg > self.best_psnr:
            self.best_epoch = epoch
            self.best_psnr = epoch_psnr.avg
            self.best_weights = copy.deepcopy(self.model)


    def save_model(self,epoch):
        model_out_path = join(self.output_dir,'epoch_{}.pth'.format(epoch))
        torch.save(self.model, model_out_path)
        print("model saved to {}".format(model_out_path))

        torch.save(self.best_weights, join(self.output_dir, 'best.pth'))
        self.logger.info('best epoch: {}, psnr: {:.2f}'.format(self.best_epoch, self.best_psnr))



    def run(self):
        self.build_model()
        print('==> model_init_finished!')
        self.logger.info('start training!')
        self.loss_list = list()
        self.loss0_list = list()
        # self.loss1_list = list()
        self.psnr_list = list()

        for epoch in range(1, self.epochs + 1):
            print('==> start training!')  #每轮epoch:train->val->保存epoch.pth->保存best.pth->plot
            self.train(epoch)
            self.val(epoch)
            self.save_model(epoch) 

            p1 = plt.figure()
            plt.subplot(1,2,1)  
            plt.xlabel("epoch")
            plt.ylabel("train_loss")
            plt.plot(self.loss_list,'k--')
            # plt.plot(self.loss0_list,'r--')
            # plt.plot(self.loss1_list,'g--')
            plt.title('train loss')

            plt.subplot(1,2,2)  
            plt.xlabel("epoch")
            plt.ylabel("val_psnr")
            plt.plot(self.psnr_list,'k--')
            plt.title('val PSNR')
            plt.savefig('loss_psnr.jpg')
        
        self.logger.info('finish training!')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str, required=False, default="/home/zjx/recovery_net/unet/datasets_small/train.h5")
    parser.add_argument('--val_path', type=str, required=False, default="/home/zjx/recovery_net/unet/datasets_small/val.h5")

    parser.add_argument('--output_dir', type=str, default = '/home/zjx/recovery_net/unet/checkpoints/one_loss/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=126)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    

    train_data = DatasetFromH5DY(args.train_path)
    val_data = DatasetFromH5DY(args.val_path)
    model = Trainer(args, train_data, val_data)
    model.run()





