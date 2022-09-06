import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models.unet import UNet1d
from data.dataset import DatasetFromFolder, DatasetFromH5DY
from utils.logging_config import get_logger
from utils.average_tools import AverageMeter,calc_psnr
import matplotlib.pyplot as plt
import time



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=False, default="/home/zjx/recovery_net/unet/datasets_small/train.h5")
    parser.add_argument('--val_path', type=str, required=False, default="/home/zjx/recovery_net/unet/datasets_small/val.h5")
    # parser.add_argument('--train_path', type=str, required=False, default="/home/zjx/recovery_net/unet/datasets_small/train_ori.txt")
    # parser.add_argument('--train_label_path', type=str, required=False, default="/home/zjx/recovery_net/unet/datasets_small/train_label.txt")
    # parser.add_argument('--train_drive_path', type=str, required=False, default="/home/zjx/recovery_net/unet/datasets_small/train_drive.txt")
    # parser.add_argument('--train_d_drive_path', type=str, required=False, default="/home/zjx/recovery_net/unet/datasets_small/train_d_drive.txt")

    # parser.add_argument('--val_path', type=str, required=False, default="/home/zjx/recovery_net/unet/datasets_small/val_ori.txt")
    # parser.add_argument('--val_label_path', type=str, required=False, default="/home/zjx/recovery_net/unet/datasets_small/val_label.txt")
    # parser.add_argument('--val_drive_path', type=str, required=False, default="/home/zjx/recovery_net/unet/datasets_small/val_drive.txt")
    # parser.add_argument('--val_d_drive_path', type=str, required=False, default="/home/zjx/recovery_net/unet/datasets_small/val_d_drive.txt")

    parser.add_argument('--outputs_dir', type=str, default = '/home/zjx/recovery_net/unet/checkpoints/one_loss/')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()  #解析参数


    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    # device_ids = [0,1,2,3]

    torch.manual_seed(args.seed)

    # model = SRCNN().to(device)
    # model = SRCNN()
    # model = torch.nn.DataParallel(model,device_ids) 
    # model = model.cuda(device=device_ids[0])
    model = UNet1d(
            in_channels=1,
            out_channels=1,
            num_stages=4,
            initial_num_channels=8,
            norm='bn',
            non_lin='relu',
            kernel_size=3
        ).to(device)
    # model = model.double()

    criterion = torch.nn.L1Loss(size_average=True)
    # optimizer = optim.Adam([
    #     {'params': model.conv1.parameters()},
    #     {'params': model.conv2.parameters()},
    #     {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    # ], lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    # train_dataset = DatasetFromFolder(args.train_path, args.train_label_path)
    # eval_dataset = DatasetFromFolder(args.val_path, args.val_label_path)
    
    train_dataset = DatasetFromH5DY(args.train_path)
    eval_dataset = DatasetFromH5DY(args.val_path)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    loss_list = list()
    psnr_list = list()

    
    logger = get_logger('signal_train.log')
    logger.info('start training!')

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                # import pdb;pdb.set_trace()
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs = inputs.unsqueeze(1)
                labels = labels.unsqueeze(1)
                # print(inputs)
                # print(inputs.device)
                inputs = inputs.float()
                labels = labels.float()

                
                preds = model(inputs)
                loss = criterion(preds, labels)
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

                

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        loss_list.append(epoch_losses.avg)

        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                inputs = inputs.unsqueeze(1)
                labels = labels.unsqueeze(1)
                preds = model(inputs)
                snr = calc_psnr(preds, labels)

                # preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        psnr_list.append(epoch_psnr.avg.item())

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

        
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch , args.num_epochs, epoch_losses.avg, epoch_psnr.avg.item() ))

        print('loss:',loss_list)
        print('psnr:',psnr_list)

        try:
            x1 = range(0, len(loss_list))
            plt.subplot(2, 1, 1)
            plt.plot(x1, loss_list, 'k--')
            plt.xlabel('epoches')
            plt.ylabel('Loss')

            plt.subplot(2, 1, 2)
            plt.plot(x1, psnr_list, 'k--')
            plt.xlabel('epoches')
            plt.ylabel('PSNR')

            plt.savefig('loss_psnr.jpg')
        except:
            pass


    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
    
    logger.info('finish training!')

    
    
