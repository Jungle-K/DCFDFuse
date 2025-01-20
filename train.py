import os
import time

import torch
import kornia
import argparse
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
from config import cfg
from core.dataset import data_loader
from core.loss import Fusionloss, cc
from core.net import fuse_1
from core.net import trans_encoder, cnn_encoder
from tools.utils import write_mes

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 5"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-debug', '--if_debug', default=False)
    parser.add_argument('-pc', '--is_pc_working', default=False)
    parser.add_argument('-notes', '--notes', default=' test')
    parser.add_argument('-gpu_id', '--gpu_id', default=0)
    args = parser.parse_args()

    num_epochs = cfg.TRAIN.NUM_EPOCHS
    epoch_gap = cfg.TRAIN.EPOCH_GAP
    train_batchsize = cfg.TRAIN.BATCHSIZE
    data_name = cfg.DATA.DATA_NAME
    is_rgb = cfg.DATA.IS_RGB
    if args.is_pc_working:
        dataset_path = cfg.DATA.PC_DATA_PATH
        log_dir = cfg.TRAIN.PC_LOG_DIR
    else:
        dataset_path = cfg.DATA.SERVER_DATA_PATH
        log_dir = cfg.TRAIN.SERCER_LOG_DIR
    dataset_path = os.path.join(dataset_path, data_name)

    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    log_dir = os.path.join(log_dir, timestamp + args.notes)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    config_record = os.path.join(log_dir, 'config.txt')
    arg_dict = args.__dict__
    msg = ['{}:{}\n'.format(k, v) for k, v in arg_dict.items()]
    write_mes(msg, config_record, mode='w')
    assert data_name != 'TNO', f"{data_name} can not be used in training!"
    assert epoch_gap < num_epochs, f"epoch_gap must be smaller than num_epochs!"

    # writer = SummaryWriter(comment=args.notes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_weight = [[2., 2., 1., 1, 0.001]]

    for l in range(len(loss_weight)):
        print('This is the {} group of loss weight.'.format(l + 1))
        trans_net = nn.DataParallel(trans_encoder())
        fuse_net = nn.DataParallel(fuse_1())

        for param in trans_net.parameters():
            param.requires_grad = False
        optimizer_fuse = torch.optim.Adam(
            fuse_net.parameters(), lr=1e-4, weight_decay=0)
        scheduler_fuse = torch.optim.lr_scheduler.StepLR(optimizer_fuse, step_size=20, gamma=0.5)

        trans_net.to(device)
        fuse_net.to(device)

        criteria_fusion = Fusionloss()
        criteria_fusion = criteria_fusion.to(device)
        MSELoss = nn.MSELoss()
        MSELoss = MSELoss.to(device)
        L1Loss = nn.L1Loss()
        L1Loss = L1Loss.to(device)
        Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')
        Loss_ssim = Loss_ssim.to(device)

        step = 0
        # torch.backends.cudnn.benchmark = True
        prev_time = time.time()
        train_stage_loss = []
        test_stage_loss = []
        min_train_loss = float('inf')
        min_test_loss = float('inf')

        train_loader, test_loader = data_loader(data_path=dataset_path, data_name=data_name, batch_size=train_batchsize,
                                                is_rgb=is_rgb)

        for epoch in range(num_epochs):

            train_epoch_loss = 0.
            pbar = tqdm(total=len(train_loader))
            for i, train_data in enumerate(train_loader):
                trans_net.train()
                fuse_net.train()

                fuse_net.zero_grad()

                optimizer_fuse.zero_grad()

                _, ir_images, vis_images, vis_rgbs = train_data
                ir_images = ir_images.to(device)
                vis_images = vis_images.to(device)
                vis_rgbs = vis_rgbs.to(device)

                ir_base, ir_detail = trans_net(ir_images)
                vis_base, vis_detail = trans_net(vis_images)
                fuse_images = fuse_net(vis_images, ir_base, vis_base, ir_detail, vis_detail)

                fusionloss, _, _ = criteria_fusion(vis_rgbs, ir_images, fuse_images)
                mse_loss_V = loss_weight[l][0] * Loss_ssim(vis_images, fuse_images) + \
                             loss_weight[l][1] * MSELoss(vis_images, fuse_images)
                mse_loss_I = loss_weight[l][0] * Loss_ssim(ir_images, fuse_images) + \
                             loss_weight[l][1] * MSELoss(ir_images, fuse_images)
                loss = loss_weight[l][2] * mse_loss_V + loss_weight[l][3] * mse_loss_I \
                       + loss_weight[l][4] * fusionloss
                loss.backward()
                nn.utils.clip_grad_norm_(
                    fuse_net.parameters(), max_norm=0.01, norm_type=2)
                optimizer_fuse.step()

                train_epoch_loss += loss.item()

                pbar.update(1)
                pbar.set_description('train')

            pbar.close()
            if not epoch < epoch_gap:
                scheduler_fuse.step()

            if optimizer_fuse.param_groups[0]['lr'] <= 1e-6:
                optimizer_fuse.param_groups[0]['lr'] = 1e-6

            train_epoch_loss /= len(train_loader)
            train_stage_loss.append(train_epoch_loss)
            if train_epoch_loss < min_train_loss:
                min_train_loss = train_epoch_loss

            trans_net.eval()
            fuse_net.eval()

            test_epoch_loss = 0.
            with torch.no_grad():
                pbar = tqdm(total=len(test_loader))
                for i, test_data in enumerate(test_loader):
                    _, ir_images, vis_images, vis_rgbs = test_data
                    ir_images = ir_images.to(device)
                    vis_images = vis_images.to(device)
                    vis_rgbs = vis_rgbs.to(device)

                    ir_base, ir_detail = trans_net(ir_images)
                    vis_base, vis_detail = trans_net(vis_images)
                    fuse_images = fuse_net(vis_images, ir_base, vis_base, ir_detail, vis_detail)

                    fusionloss, _, _ = criteria_fusion(vis_rgbs, ir_images, fuse_images)
                    mse_loss_V = loss_weight[l][0] * Loss_ssim(vis_images, fuse_images) + \
                                 loss_weight[l][1] * MSELoss(vis_images, fuse_images)
                    mse_loss_I = loss_weight[l][0] * Loss_ssim(ir_images, fuse_images) + \
                                 loss_weight[l][1] * MSELoss(ir_images, fuse_images)
                    loss = loss_weight[l][2] * mse_loss_V + loss_weight[l][3] * mse_loss_I \
                           + loss_weight[l][4] * fusionloss

                    test_epoch_loss += loss.item()

                    pbar.update(1)
                    pbar.set_description('test')
                pbar.close()
                test_epoch_loss /= len(test_loader)
                test_stage_loss.append(test_epoch_loss)
                if test_epoch_loss < min_test_loss:
                    min_test_loss = test_epoch_loss
                # if epoch == epoch_gap:
                #     min_test_loss = test_epoch_loss
                if (epoch + 1) % 20 == 0:
                    pth_dir = os.path.join(log_dir, 'model_loss_group{}_epoch{}.pth'.format((l + 1), (epoch + 1)))
                    if args.is_pc_working:
                        checkpoint = {
                            'trans_net': trans_net.state_dict(),
                            'fuse_net': fuse_net.state_dict(),
                        }
                    else:
                        checkpoint = {
                            'trans_net': trans_net.module.state_dict(),
                            'fuse_net': fuse_net.module.state_dict(),
                        }
                    torch.save(checkpoint, pth_dir)

            info = "Group{},This is the {}/{} epoch".format(l + 1, epoch + 1, num_epochs) + "." \
                   + "The train loss is {:.5f}, best train loss : {:.5f}, the test loss is {:.5f},best test loss : {:.5f}" \
                       .format(train_epoch_loss, min_train_loss, test_epoch_loss, min_test_loss)
            write_mes(info, config_record, mode='a')

        plt.plot(train_stage_loss, label='group{}_train'.format(l + 1))
        plt.plot(test_stage_loss, label='group{}_test'.format(l + 1))
        plt.legend()
        loss_line_path = os.path.join(log_dir,
                                      'loss_group{}_train{}_test{}.png'.format(l + 1, min_train_loss, min_test_loss))
        plt.savefig(loss_line_path)
