import os

import torch
import argparse
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import cfg
from core.dataset import data_loader
from core.net import trans_encoder
from core.net import fuse_1
from tools.utils import write_mes, img_save, image_read_cv2
from tools.evaluator import Evaluator
from tensorboardX import SummaryWriter

'''
结果保存在test_log
查看命令
tensorboard --logdir=./test_log
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-debug', '--if_debug', default=False)
    parser.add_argument('-pc', '--is_pc_working', default=True)
    parser.add_argument('-notes', '--notes', default='')
    parser.add_argument('-gpu_id', '--gpu_id', default=0)
    parser.add_argument('-model_dir', '--model_dir',
                        default="E:/Programs/Vis-Inf_Fuse/finnal/DCFDFuse/"
                                "models/model_loss_group_epoch100.pth")
    args = parser.parse_args()

    metrics = ['EN', 'SD', 'SF', 'MI', 'SCD', 'VIF', 'Qabf', 'SSIM', 'AG', 'MSE', 'CC', 'PSNR']
    df_combined = pd.DataFrame(index=metrics)

    for i in [0]:
        print('This is group {}'.format(i + 1))
        debug_training = False
        test_batchsize = cfg.TEST.BATCHSIZE
        if args.is_pc_working:
            dataset_path = cfg.DATA.PC_DATA_PATH
            log_dir = cfg.TEST.PC_LOG_DIR
        else:
            dataset_path = cfg.DATA.SERVER_DATA_PATH
            log_dir = cfg.TEST.SERCER_LOG_DIR
        model_dir = args.model_dir
        # model_dir = model_dir.split('group')[0] + 'group' + str(i + 1) + model_dir.split('group')[1][1:]

        note_id = model_dir.split('/')[-2].split()[-1]
        timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
        out_dir = timestamp + ' ' + note_id + ' test_' + str(i + 1)
        if args.notes is not None:
            out_dir = out_dir + args.notes

        note_id_dir = os.path.join(log_dir, note_id)
        if not os.path.isdir(note_id_dir):
            os.mkdir(note_id_dir)
        part_note_id = note_id + '_' + str(i + 1)
        log_dir = os.path.join(note_id_dir, out_dir)
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        data_name = cfg.DATA.DATA_NAME

        image_out_dir = os.path.join(log_dir, data_name)
        if not os.path.isdir(image_out_dir):
            os.mkdir(image_out_dir)

        dataset_path = os.path.join(dataset_path, data_name)
        is_rgb = cfg.DATA.IS_RGB

        # writer = SummaryWriter(log_dir, comment=args.notes)

        train_loader, test_loader = data_loader(data_path=dataset_path, data_name=data_name, batch_size=test_batchsize,
                                                is_rgb=is_rgb)

        config_record = os.path.join(log_dir, 'config.txt')
        evaluator_record = os.path.join(log_dir, 'evaluator.csv')
        arg_dict = args.__dict__
        msg = ['{}:{}\n'.format(k, v) for k, v in arg_dict.items()]
        write_mes(msg, config_record, mode='w')

        device = 'cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu'
        trans_net = trans_encoder().to(device)
        fuse_net = fuse_1().to(device)

        trans_net.load_state_dict(torch.load(model_dir)['trans_net'])

        # ckpt_path = './models/CDDFuse_IVF.pth'
        # state_dict = torch.load(ckpt_path)['DIDF_Encoder']
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     if k.startswith('module.'):
        #         new_state_dict[k[7:]] = v  # 去掉'module.'前缀
        #     else:
        #         new_state_dict[k] = v
        # trans_net.load_state_dict(new_state_dict)

        fuse_net.load_state_dict(torch.load(model_dir)['fuse_net'])

        trans_net.eval()
        # high_net.eval()
        # cnn_net.eval()
        fuse_net.eval()
        test_epoch_loss = 0.
        with torch.no_grad():
            pbar = tqdm(total=len(test_loader))
            metric_result = np.zeros((13))
            for index, test_data in enumerate(test_loader):
                name, ir_images, vis_images, vis_rgbs = test_data
                ir_images = ir_images.to(device)
                vis_images = vis_images.to(device)
                vis_rgbs = vis_rgbs.to(device)
                '''
                data = [ir_images, vis_images, ir_L_images, vis_L_images, ir_H_images, vis_H_images]

                for i in range(len(data)):
                    data[i] = data[i].to(device)

                ir_images, vis_images, ir_L_images, vis_L_images, ir_H_images, vis_H_images = data
                '''

                ir_base, ir_detail = trans_net(ir_images)
                vis_base, vis_detail = trans_net(vis_images)
                fuse_images = fuse_net(vis_images, ir_base, vis_base, ir_detail, vis_detail)

                fuse_images = fuse_images + vis_images

                for k in range(fuse_images.shape[0]):
                    fusion_image = (fuse_images[k] - torch.min(fuse_images[k])) / (
                            torch.max(fuse_images[k]) - torch.min(fuse_images[k]))
                    fi = np.squeeze((fusion_image * 255).cpu().numpy())
                    try:
                        img_save(fi, name[k].split(sep='.')[0], image_out_dir)
                    except:
                        debug_training = True
                        break

                pbar.update(1)
                pbar.set_description('test:{}'.format(i + 1))
                if debug_training:
                    break
            pbar.close()
            if debug_training:
                metric_result = np.zeros(12)
                df = pd.DataFrame(metric_result, index=metrics, columns=pd.Index([part_note_id]))
                df_combined = pd.concat([df_combined, df], axis=1)
                continue
            # exit()

            if data_name == 'MSRS':
                print('The evaluation indicators is calculating...')
                dataset_path = os.path.join(dataset_path, "test")
                metric_result = np.zeros(12)
                for img_name in os.listdir(os.path.join(dataset_path, "Ir")):
                    ir = image_read_cv2(os.path.join(dataset_path, "Ir", img_name), 'GRAY')
                    vi = image_read_cv2(os.path.join(dataset_path, "Vis", img_name), 'GRAY')
                    ir = np.resize(ir, cfg.DATA.DATA_SIZE)
                    vi = np.resize(vi, cfg.DATA.DATA_SIZE)
                    fi = image_read_cv2(os.path.join(image_out_dir, img_name.split('.')[0] + ".png"), 'GRAY')
                    # fi = np.resize(fi, data_size)
                    metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                                  , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                                  , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                                  , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)
                                                  , Evaluator.AG(fi), Evaluator.MSE(fi, ir, vi)
                                                  , Evaluator.CC(fi, ir, vi), Evaluator.PSNR(fi, ir, vi)])

                metric_result /= len(os.listdir(image_out_dir))
                print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
                print("DCFDFuse    " + '\t' + str(np.round(metric_result[0], 2)) + '\t'
                      + str(np.round(metric_result[1], 2)) + '\t'
                      + str(np.round(metric_result[2], 2)) + '\t'
                      + str(np.round(metric_result[3], 2)) + '\t'
                      + str(np.round(metric_result[4], 2)) + '\t'
                      + str(np.round(metric_result[5], 2)) + '\t'
                      + str(np.round(metric_result[6], 2)) + '\t'
                      + str(np.round(metric_result[7], 2))
                      )
                print("=" * 80)
                metric_log = ['{}:{}\n'.format(k, v) for k, v in zip(metrics, metric_result)]
                write_mes(metric_log, config_record, mode='a')
                df = pd.DataFrame(metric_result, index=metrics, columns=pd.Index([part_note_id]))
                df_combined = pd.concat([df_combined, df], axis=1)
                df.to_csv(evaluator_record, index=True, float_format='%.6f', header=True)
            elif data_name == 'M3FD':
                print('The evaluation indicators is calculating...')
                # dataset_path = os.path.join(dataset_path, "test")
                metric_result = np.zeros(12)
                for img_name in os.listdir(image_out_dir):
                    ir = image_read_cv2(os.path.join(dataset_path, "Ir", img_name), 'GRAY')
                    vi = image_read_cv2(os.path.join(dataset_path, "Vis", img_name), 'GRAY')
                    ir = np.resize(ir, cfg.DATA.DATA_SIZE)
                    vi = np.resize(vi, cfg.DATA.DATA_SIZE)
                    fi = image_read_cv2(os.path.join(image_out_dir, img_name.split('.')[0] + ".png"), 'GRAY')
                    # fi = np.resize(fi, data_size)
                    metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                                  , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                                  , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                                  , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)
                                                  , Evaluator.AG(fi), Evaluator.MSE(fi, ir, vi)
                                                  , Evaluator.CC(fi, ir, vi), Evaluator.PSNR(fi, ir, vi)])

                metric_result /= len(os.listdir(image_out_dir))
                print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
                print("DCFDFuse    " + '\t' + str(np.round(metric_result[0], 2)) + '\t'
                      + str(np.round(metric_result[1], 2)) + '\t'
                      + str(np.round(metric_result[2], 2)) + '\t'
                      + str(np.round(metric_result[3], 2)) + '\t'
                      + str(np.round(metric_result[4], 2)) + '\t'
                      + str(np.round(metric_result[5], 2)) + '\t'
                      + str(np.round(metric_result[6], 2)) + '\t'
                      + str(np.round(metric_result[7], 2))
                      )
                print("=" * 80)
                metric_log = ['{}:{}\n'.format(k, v) for k, v in zip(metrics, metric_result)]
                write_mes(metric_log, config_record, mode='a')
                df = pd.DataFrame(metric_result, index=metrics, columns=pd.Index([part_note_id]))
                df_combined = pd.concat([df_combined, df], axis=1)
                df.to_csv(evaluator_record, index=True, float_format='%.6f', header=True)
            elif data_name == 'TNO':
                print('The evaluation indicators is calculating...')
                # dataset_path = os.path.join(dataset_path, "test")
                metric_result = np.zeros(12)
                for img_name in os.listdir(image_out_dir):
                    img_name = img_name.split('.')[0]
                    ir = image_read_cv2(os.path.join(dataset_path, "Ir", img_name + '.bmp'), 'GRAY')
                    vi = image_read_cv2(os.path.join(dataset_path, "Vis", img_name + '.bmp'), 'GRAY')
                    ir = np.resize(ir, cfg.DATA.DATA_SIZE)
                    vi = np.resize(vi, cfg.DATA.DATA_SIZE)
                    fi = image_read_cv2(os.path.join(image_out_dir, img_name.split('.')[0] + ".png"), 'GRAY')
                    # fi = np.resize(fi, data_size)
                    metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                                  , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                                  , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                                  , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)
                                                  , Evaluator.AG(fi), Evaluator.MSE(fi, ir, vi)
                                                  , Evaluator.CC(fi, ir, vi), Evaluator.PSNR(fi, ir, vi)])

                metric_result /= len(os.listdir(image_out_dir))
                print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
                print("DCFDFuse    " + '\t' + str(np.round(metric_result[0], 2)) + '\t'
                      + str(np.round(metric_result[1], 2)) + '\t'
                      + str(np.round(metric_result[2], 2)) + '\t'
                      + str(np.round(metric_result[3], 2)) + '\t'
                      + str(np.round(metric_result[4], 2)) + '\t'
                      + str(np.round(metric_result[5], 2)) + '\t'
                      + str(np.round(metric_result[6], 2)) + '\t'
                      + str(np.round(metric_result[7], 2))
                      )
                print("=" * 80)
                metric_log = ['{}:{}\n'.format(k, v) for k, v in zip(metrics, metric_result)]
                write_mes(metric_log, config_record, mode='a')
                df = pd.DataFrame(metric_result, index=metrics, columns=pd.Index([part_note_id]))
                df_combined = pd.concat([df_combined, df], axis=1)
                df.to_csv(evaluator_record, index=True, float_format='%.6f', header=True)
    eva_total_csv_path = os.path.join(note_id_dir, 'eva_total.csv')
    df_combined.to_csv(eva_total_csv_path, index=True, float_format='%.6f', header=True)
