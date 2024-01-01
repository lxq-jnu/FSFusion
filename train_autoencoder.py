#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/17 10:44
# @Author  : wangjuan
# @File    : train_autoencoder_COCO.py
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from net import NestFuse_autoencoder
from args_fusion import args
import pytorch_msssim
from utils_data_COCO import read_data, input_setup
# from HED.HED_edge import Get_edgeMap
from HED.HED_edge import Get_edgeMap



edge_weight = 500
def main():
    for i in range(2, 3):
        # i = 3
        train(i)


def train(i):
    batch_size = args.batch_size

    # load network model
    # nest_model = FusionNet_gra()
    input_nc = 1
    output_nc = 1
    # true for deeply supervision
    # In our paper, deeply supervision strategy was not used.
    deepsupervision = False
    nb_filter = [64, 112, 160, 208, 256]

    nest_model = NestFuse_autoencoder(nb_filter, input_nc, output_nc, deepsupervision)

    if args.resume is not None:
        print('Resuming, initializing using weight from {}.'.format(args.resume))
        nest_model.load_state_dict(torch.load(args.resume))
    print(nest_model)
    # for n,p in nest_model.named_parameters():
    #     p.requires_grad = False
    # for n,p in nest_model.named_parameters():
    #     No_frozen_layer = ['conv0', 'EB1_0', 'DB1_1', 'conv_out']
    #     for fl in No_frozen_layer:
    #         if fl in n:
    #             p.requires_grad = True
    #             print('No_frozen', n)
    #             break
        # print(n)
        # print(p)
    optimizer = Adam(nest_model.parameters(), args.lr)
    # optimizer = Adam(filter(lambda p: p.requires_grad, nest_model.parameters()), args.lr)
    mse_loss = torch.nn.MSELoss()
    ssim_loss = pytorch_msssim.msssim

    if args.cuda:
        nest_model.cuda()

    # tbar = trange(args.epochs)
    print('\n')
    print('Start training.....')
    print("Data preparation!")
    if not os.path.exists(os.path.join('./checkpoint', "Train")):
        print("Ready")
        input_setup("Train/Train2014")

    train_data_dir = os.path.join('./checkpoint', "Train", "train.h5")
    edge_data_dir = os.path.join('./checkpoint', "edge_train", "train.h5")
    print("Data preparation over!")
    print("Reading data!")
    train_data = read_data(train_data_dir)
    print(len(train_data))
    edge_data = read_data(edge_data_dir)
    print(len(edge_data))

    Loss_pixel = []
    Loss_ssim = []                                                                                     
    Loss_egde = []
    Loss_all = []
    count_loss = 0
    all_ssim_loss = 0.
    all_pixel_loss = 0.
    all_egde_loss = 0.
    print("ready")
    for e in trange(args.epochs):
        print('\nEpoch %d.....' % e)
        # load training database
        if args.cuda:
            nest_model.cuda()

        batches = len(train_data) // batch_size
        nest_model.train()
        count = 0
        for batch in range(batches):
            img = train_data[batch * batch_size: (batch + 1) * batch_size]
            img = torch.from_numpy(img).float()
            edge_img = edge_data[batch * batch_size: (batch + 1) * batch_size]
            edge_img = torch.from_numpy(edge_img).float()
            count += 1

            # 把梯度置零，也就是把loss关于weight的导数变成0
            optimizer.zero_grad()

            img = Variable(img, requires_grad=False)
            edge_img = Variable(edge_img, requires_grad=False)
            if args.cuda:
                img = img.cuda()
                edge_img = edge_img.cuda()
            # get fusion image
            # encoder
            en = nest_model.encoder(img)
            # # edge
            edge_feature = nest_model.edge_feature(edge_img * 255.0 + img)
            x1_0 = nest_model.edge_guidance1(edge_feature, en[0])
            # x2_0 = nest_model.edge_guidance2(edge_feature, en[1])
            # x3_0 = nest_model.edge_guidance3(edge_feature, en[2])
            # DFM
            DFM = nest_model.non_local(en)
            # decoder
            outputs = nest_model.decoder_train([x1_0,en[1],en[2]], DFM)
            # resolution loss: between fusion image and visible image
            x = Variable(img.data.clone(), requires_grad=False)
            x_edge = Variable(edge_img.data.clone(), requires_grad=False)
            #
            outputs_tensor = outputs[0]
            outputs_tensor = Variable(outputs_tensor.data.clone(), requires_grad=False)
            outputs_edge_map_sequence = []
            # #
            # HED生成图像
            for j in range(args.batch_size):
                # print(outputs_tensor[j].shape)
                outputs_edgeMap_temp = Get_edgeMap(outputs_tensor[j].squeeze().cpu().numpy())

                outputs_edgeMap_temp = torch.from_numpy(outputs_edgeMap_temp)
                outputs_edge_map_sequence.append(outputs_edgeMap_temp)
            # # print(outputs_edge_map_sequence.shape)
            outputs_edge_map_result = torch.stack(outputs_edge_map_sequence, dim=0)
            # print(outputs_edge_map_result)
            # # print(outputs_edge_map_result.shape)
            # # print(type(outputs_edge_map_result))
            outputs_edge_map_result = outputs_edge_map_result.cuda()

            # # HED_main生成图像
            # for j in range(args.batch_size):
            #     # print(outputs_tensor[j].shape)
            #     outputs_edgeMap_temp = Get_edgeMap(outputs_tensor[j].squeeze().cpu().numpy())
            #
            #     outputs_edgeMap_temp = torch.from_numpy(outputs_edgeMap_temp)
            #     outputs_edge_map_sequence.append(outputs_edgeMap_temp)
            # # # print(outputs_edge_map_sequence.shape)
            # outputs_edge_map_result = torch.stack(outputs_edge_map_sequence, dim=0)
            # # print(outputs_edge_map_result)
            # # # print(outputs_edge_map_result.shape)
            # # # print(type(outputs_edge_map_result))
            # outputs_edge_map_result = outputs_edge_map_result.cuda()

            ssim_loss_value = 0.
            pixel_loss_value = 0.
            edge_loss_value = 0.

            edge_loss_temp = mse_loss(outputs_edge_map_result, x_edge)
            edge_loss_value += edge_loss_temp

            for output in outputs:
                pixel_loss_temp = mse_loss(output, x)
                ssim_loss_temp = ssim_loss(output, x, normalize=True)
                ssim_loss_value += (1 - ssim_loss_temp)
                pixel_loss_value += pixel_loss_temp
            ssim_loss_value /= len(outputs)
            pixel_loss_value /= len(outputs)
            edge_loss_value /= len(outputs)

            # total loss
            total_loss = pixel_loss_value + args.ssim_weight[i] * ssim_loss_value + edge_weight * edge_loss_value
            total_loss.backward()
            optimizer.step()

            all_ssim_loss += ssim_loss_value.item()
            all_pixel_loss += pixel_loss_value.item()
            all_egde_loss += edge_loss_value.item()
            if (batch + 1) % args.log_interval == 0:
                print(
                    "{}\t SSIM weight {}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t ssim loss: {:.6f}\t edge loss: {:.6f}\t total: {:.6f}".format(
                        time.ctime(), i, e + 1, count, batches,
                                         all_pixel_loss / args.log_interval,
                                         (args.ssim_weight[i] * all_ssim_loss) / args.log_interval,
                                         (edge_weight * all_egde_loss) / args.log_interval,
                                         (args.ssim_weight[i] * all_ssim_loss + all_pixel_loss + edge_weight * all_egde_loss) / args.log_interval
                    ))
                Loss_pixel.append(all_pixel_loss / args.log_interval)
                Loss_ssim.append(all_ssim_loss / args.log_interval)
                Loss_egde.append(all_egde_loss / args.log_interval)
                Loss_all.append((args.ssim_weight[i] * all_ssim_loss + edge_weight * all_egde_loss) / args.log_interval)
                count_loss = count_loss + 1
                all_ssim_loss = 0.
                all_pixel_loss = 0.
                all_egde_loss = 0.

            if (batch + 1) % (20 * args.log_interval) == 0:
                # save model
                nest_model.eval()
                nest_model.cpu()
                save_model_filename = args.ssim_path[i] + '/' + "Epoch_" + str(e) + "_iters_" + str(count) + "_" + \
                                      str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[
                                          i] + ".model"
                save_model_path = os.path.join(args.save_model_dir_autoencoder, save_model_filename)
                print(os.path.join(args.save_model_dir_autoencoder, args.ssim_path[i]))
                check_paths(os.path.join(args.save_model_dir_autoencoder, args.ssim_path[i]))
                torch.save(nest_model.state_dict(), save_model_path)
                # save loss data
                # pixel loss
                loss_data_pixel = Loss_pixel
                loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_pixel_epoch_" + str(
                    e) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
                                     args.ssim_path[i] + ".txt"
                check_paths(args.save_loss_dir + args.ssim_path[i])
                scio.savemat(loss_filename_path, {'loss_pixel': loss_data_pixel})
                # # SSIM loss
                loss_data_ssim = Loss_ssim
                loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_ssim_epoch_" + str(
                    e) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
                                     args.ssim_path[i] + ".txt"
                check_paths(args.save_loss_dir + args.ssim_path[i])
                scio.savemat(loss_filename_path, {'loss_ssim': loss_data_ssim})
                # # Edge loss
                loss_data_edge = Loss_egde
                loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + 'Edge_loss' + '/' + "loss_edge_epoch_" + str(
                    e) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
                                     args.ssim_path[i] + ".txt"
                print(loss_filename_path)
                check_paths(args.save_loss_dir + args.ssim_path[i] + '/' + 'Edge_loss')
                scio.savemat(loss_filename_path, {'loss_edge': loss_data_edge})
                # # all loss
                loss_data = Loss_all
                loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_all_epoch_" + str(
                    e) + "_iters_" + \
                                     str(count) + "-" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
                                     args.ssim_path[i] + ".txt"
                check_paths(args.save_loss_dir + args.ssim_path[i])
                scio.savemat(loss_filename_path, {'loss_all': loss_data})

                nest_model.train()
                nest_model.cuda()
                print("\nCheckpoint, trained model saved at")

    # pixel loss
    # loss_data_pixel = Loss_pixel
    # loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_pixel_epoch_" + str(
    # 	args.epochs) + "_" + str(
    # 	time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
    #
    # scio.savemat(loss_filename_path, {'final_loss_pixel': loss_data_pixel})
    # loss_data_ssim = Loss_ssim
    # loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_ssim_epoch_" + str(
    # 	args.epochs) + "_" + str(
    # 	time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
    # scio.savemat(loss_filename_path, {'final_loss_ssim': loss_data_ssim})
    # # SSIM loss
    # loss_data = Loss_all
    # loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_all_epoch_" + str(
    # 	args.epochs) + "_" + str(
    # 	time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
    # scio.savemat(loss_filename_path, {'final_loss_all': loss_data})
    # save model
    nest_model.eval()
    nest_model.cpu()
    save_model_filename = args.ssim_path[i] + '/' "Final_epoch_" + str(args.epochs) + "_" + \
                          str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".model"
    save_model_path = os.path.join(args.save_model_dir_autoencoder, save_model_filename)
    check_paths(os.path.join(args.save_model_dir_autoencoder, args.ssim_path[i]))
    torch.save(nest_model.state_dict(), save_model_path)

    print("\nDone, trained model saved at")


def check_paths(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    main()
