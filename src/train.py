import argparse
import numpy as np
import torch
import torch.nn as nn
# from tensorboardX import SummaryWriter
from time import time
# from torch.utils.data import DataLoader, Subset

import torch.optim as optim
from tqdm import tqdm
import os
import random
import sys

import logging
import math
import importlib
import datetime
import munch
import yaml
import copy

from train_utils import AverageValueMeter, save_model
from dataset import ModelNet40

import warnings
warnings.filterwarnings("ignore")


def train():
    logging.info(str(args))
    metrics = ['RotE', 'transE', 'MSE', 'RMSE', 'recall']
    best_epoch_losses = {m: (0, 0) if m =='recall' else (0, math.inf) for m in metrics}
    # best_epoch_losses = (0, inf)
    # train_loss_meter = AverageValueMeter()
    val_loss_meters = {m: AverageValueMeter() for m in metrics}
    val_split_loss_meters = []

    num_R_levels = args.num_rot_levels if args.num_rot_levels > 0 else 1
    num_M_levels = args.num_corr_levels if args.num_corr_levels > 0 else 1
    for i in range(num_R_levels):
        row = []
        for j in range(num_M_levels):
            row.append(copy.deepcopy(val_loss_meters))
        val_split_loss_meters.append(row)
    val_split_loss_meters = np.array(val_split_loss_meters)

    if args.benchmark == "MVP":
        dataset = MVP_RG(prefix="train", args=args)
        dataset_test = MVP_RG(prefix="val", args=args)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers))
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))
    elif args.benchmark == "MN40":
        dataset = ModelNet40(prefix='train', args=args)
        dataset_test = ModelNet40(prefix='val', args=args)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)

    logging.info('Length of train dataset:%d', len(dataset))
    logging.info('Length of test dataset:%d', len(dataset_test))

    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)

    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    if hasattr(model_module, 'weights_init'):
        net.module.apply(model_module.weights_init)

    lr = args.lr
    optimizer = getattr(optim, args.optimizer)
    if args.optimizer == 'Adam':
        betas = args.betas.split(',')
        betas = (float(betas[0].strip()), float(betas[1].strip()))
        optimizer = optimizer(net.module.parameters(), lr=lr, weight_decay=args.weight_decay, betas=betas)
    else:
        raise ValueError('must set an optimizer')

    if args.lr_decay:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_decay_rate, min_lr=args.lr_clip)
    
    if args.load_model:
        ckpt = torch.load(args.load_model)
        net.module.load_state_dict(ckpt['net_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)

    for epoch in range(args.start_epoch, args.nepoch):
        # train_loss_meter.reset()
        net.module.train()

        fw_time = []
        bw_time = []
        data_time = []
        
        start = time()
        for i, data in enumerate(dataloader, 0):
            src, tgt, T_gt, _, _ = data

            src = src.float().cuda()
            tgt = tgt.float().cuda()
            T_gt = T_gt.float().cuda()

            data_time.append(time() - start)

            optimizer.zero_grad()
            net_loss, r_err, t_err, rmse, mse = net(src, tgt, T_gt)
            fw_time.append(time() - start)

            # train_loss_meter.update(net_loss.mean().item())
            net_loss.backward()
            optimizer.step()
            bw_time.append(time() - start)

            if i % args.step_interval_to_print == 0:
                logging.info('data time: %f,\t fw time: %f,\t bw time: %f\t' % (np.mean(data_time), np.mean(fw_time), np.mean(bw_time)))
                logging.info(exp_name + ' train [%d: %d/%d] total_loss: %.4f rot_loss: %.4f trans_loss: %.4f rmse_loss: %.4f mse_loss: %.4f lr: %f' %
                    (epoch, i, len(dataset) / args.batch_size, net_loss.mean().item(), r_err.mean().item(), t_err.mean().item(), rmse.mean().item(), mse.mean().item(), lr) )
                fw_time.clear()
                bw_time.clear()
                data_time.clear()
            start = time()
        scheduler.step(net_loss)

        if epoch % args.epoch_interval_to_save == 0:
            save_model('%s/network.pth' % log_dir, net)
            logging.info("Saving net...")

        if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
            val(net, epoch, val_loss_meters, val_split_loss_meters, dataloader_test, best_epoch_losses)


def val(net, curr_epoch_num, val_loss_meters, val_split_loss_meters, dataloader_test, best_epoch_losses, rmse_thresh=0.1):
    logging.info('Testing...')
    for v in val_loss_meters.values():
        v.reset()

    for i in range(val_split_loss_meters.shape[0]):
        for j in range(val_split_loss_meters.shape[1]):
            for v in val_split_loss_meters[i][j].values():
                v.reset()
    
    # num_samples = np.ones(2, 2) * 1.e-6

    num_R_levels, num_M_levels = val_split_loss_meters.shape

    net.module.eval()

    with torch.no_grad():
        for _, data in enumerate(dataloader_test):
            src, tgt, T_gt, match_level, rot_level = data
            curr_batch_size = T_gt.shape[0]

            src = src.float().cuda()
            tgt = tgt.float().cuda()
            T_gt = T_gt.float().cuda()
            match_level = match_level.int().cuda()
            rot_level = rot_level.int().cuda()

            _, r_err, t_err, rmse, mse = net(src, tgt, T_gt, prefix="val")

            val_loss_meters['RotE'].update(r_err.mean().item(), curr_batch_size)
            val_loss_meters['transE'].update(t_err.mean().item(), curr_batch_size)
            val_loss_meters['MSE'].update(mse.mean().item(), curr_batch_size)
            val_loss_meters['RMSE'].update(rmse.mean().item(), curr_batch_size)
            val_loss_meters['recall'].update((rmse < rmse_thresh).to(torch.float32).mean().item(), curr_batch_size)
            
            for i in range(curr_batch_size):
                cur_rot_level = rot_level[i] if num_R_levels > 1 else 0
                cur_match_level = match_level[i] if num_M_levels > 1 else 0
                val_split_loss_meters[cur_rot_level][cur_match_level]['RotE'].update(r_err[i].item())
                val_split_loss_meters[cur_rot_level][cur_match_level]['transE'].update(t_err[i].item())
                val_split_loss_meters[cur_rot_level][cur_match_level]['MSE'].update(mse[i].item())
                val_split_loss_meters[cur_rot_level][cur_match_level]['RMSE'].update(rmse[i].item())
                val_split_loss_meters[cur_rot_level][cur_match_level]['recall'].update((rmse[i] < rmse_thresh).item())
            
        fmt = 'best_%s: %f [epoch %d]; '
        best_log = ''
        for loss_type, (curr_best_epoch, curr_best_loss) in best_epoch_losses.items():
            if (val_loss_meters[loss_type].avg < curr_best_loss and loss_type != 'recall') or \
                    (val_loss_meters[loss_type].avg > curr_best_loss and loss_type == 'recall'):
                best_epoch_losses[loss_type] = (curr_epoch_num, val_loss_meters[loss_type].avg)
                save_model('%s/best_%s_network.pth' % (log_dir, loss_type), net)
                logging.info('Best %s net saved!' % loss_type)
                best_log += fmt % (loss_type, best_epoch_losses[loss_type][1], best_epoch_losses[loss_type][0])
            else:
                best_log += fmt % (loss_type, curr_best_loss, curr_best_epoch)

        curr_log = ''
        for loss_type, meter in val_loss_meters.items():
            curr_log += 'curr_%s: %f; ' % (loss_type, meter.avg)

        logging.info(curr_log)
        logging.info(best_log)

        for i in range(val_split_loss_meters.shape[0]):
            for j in range(val_split_loss_meters.shape[1]):
                curr_val_level = val_split_loss_meters[i][j]
                
                curr_split_log = '[rot_level %d, match_level %d] ' % (i, j)
                for loss_type, meter in curr_val_level.items():
                    curr_split_log += 'curr_%s: %f; ' % (loss_type, meter.avg)

                logging.info(curr_split_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    curr_time = datetime.datetime.now().isoformat()[:19]
    # if args.load_model:
    #     exp_name = os.path.basename(os.path.dirname(args.load_model))
    #     log_dir = os.path.dirname(args.load_model)
    # else:
    exp_name = args.model_name + '_' + args.benchmark + '_' + args.flag + '_' + curr_time + '_' + args.exp_name
    log_dir = os.path.join(args.work_dir, exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                        logging.StreamHandler(sys.stdout)])

    os.system('cp %s %s' % ('./models/' + args.model_name + '.py', log_dir + '/'+args.model_name + '.py.backup'))
    train()


