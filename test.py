#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/18 10:35
# @Author  : wangjuan
# @File    : test.py

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from torch.autograd import Variable
from net import FSFusion_autoencoder
import utils
from args_fusion import args
import numpy as np
import FSFusion_strategy


def load_model(path, deepsupervision):
    input_nc = 1
    output_nc = 1
    nb_filter = [64, 112, 160, 208, 256]

    FSFusion_model = FSFusion_autoencoder(nb_filter, input_nc, output_nc, deepsupervision)
    # for var_name in nest_model.state_dict():
    #     print(var_name, "\t", nest_model.state_dict()[var_name].size())
    FSFusion_model.load_state_dict(torch.load(path))

    para = sum([np.prod(list(p.size())) for p in FSFusion_model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(FSFusion_model._get_name(), para * type_size / 1000 / 1000))

    FSFusion_model.eval()
    FSFusion_model.cuda()

    return FSFusion_model


def run_demo(FSFusion_model, infrared_path, visible_path, output_path_root, index, f_type):
    edge_img_ir, img_ir, h, w, c = utils.get_test_image(infrared_path)
    edge_img_vi, img_vi, h, w, c = utils.get_test_image(visible_path)

    # dim = img_ir.shape
    if c is 1:
        if args.cuda:
            img_ir = img_ir.cuda()
            edge_img_ir = edge_img_ir.cuda()
            img_vi = img_vi.cuda()
            edge_img_vi = edge_img_vi.cuda()
        img_ir = Variable(img_ir, requires_grad=False)
        edge_img_ir = Variable(edge_img_ir, requires_grad=False)
        img_vi = Variable(img_vi, requires_grad=False)
        edge_img_vi = Variable(edge_img_vi, requires_grad=False)
        # encoder
        fusion_function = FSFusion_strategy.attention_fusion_weight

        en_r = FSFusion_model.encoder(img_ir)
        en_v = FSFusion_model.encoder(img_vi)
        # x_ir = en_r[3]
        # x_vi = en_v[3]
        # # save features
        # file_name_ir = str(index) + '_ir_4_deep' + '.png'
        # output_path = output_path_root + file_name_ir
        # utils.save_image_test(torch.unsqueeze(x_ir[:, 32, :, :], 1), output_path)
        # print(output_path)
        #
        # file_name_vi = str(index) + '_vi_4_deep' + '.png'
        # output_path = output_path_root + file_name_vi
        # utils.save_image_test(torch.unsqueeze(x_vi[:, 32, :, :], 1), output_path)
        # print(output_path)
        # fusion
        # print(np.array(en_v).size())
        # max
        # en_0 = torch.max(en_v[0], en_r[0])
        # en_1 = torch.max(en_v[1], en_r[1])
        # en_2 = torch.max(en_v[2], en_r[2])
        # attention
        en_0 = fusion_function(en_v[0], en_r[0])
        en_1 = fusion_function(en_v[1], en_r[1])
        en_2 = fusion_function(en_v[2], en_r[2])

        # en_3 = fusion_function(en_v[3],en_r[3],'attention_max')
        img = (img_ir + img_vi) * 0.5
        # img_norm = img / 255.0
        edge_img = torch.max(edge_img_ir, edge_img_vi)
        # edge_feature_vi = FSFusion_model.edge_feature(edge_img_vi * 255.0 + img_vi)
        # edge_feature_ir = FSFusion_model.edge_feature(edge_img_ir * 255.0 + img_ir)
        edge_feature = FSFusion_model.edge_feature(edge_img * 255.0 + img)
        # edge_feature = fusion_function(edge_feature_vi, edge_feature_ir, 'attention_max')
        x1_0 = FSFusion_model.edge_guidance1(edge_feature, en_0)
        x2_0 = FSFusion_model.edge_guidance2(edge_feature, en_1)
        x3_0 = FSFusion_model.edge_guidance3(edge_feature, en_2)

        #
        DFM_vi = FSFusion_model.non_local(en_v)
        DFM_ir = FSFusion_model.non_local(en_r)
        #
        DFM = fusion_function(DFM_vi, DFM_ir)

        # decoder
        img_fusion_list = FSFusion_model.decoder_eval([x1_0, x2_0, x3_0], DFM)
    else:
        # fusion each block
        img_fusion_blocks = []
        for i in range(c):
            # encoder
            img_vi_temp = img_vi[i]
            img_ir_temp = img_ir[i]
            if args.cuda:
                img_vi_temp = img_vi_temp.cuda()
                img_ir_temp = img_ir_temp.cuda()
            img_vi_temp = Variable(img_vi_temp, requires_grad=False)
            img_ir_temp = Variable(img_ir_temp, requires_grad=False)

            en_r = FSFusion_model.encoder(img_ir)
            en_v = FSFusion_model.encoder(img_vi)

            # fusion
            f = FSFusion_model.fusion(en_r, en_v, f_type)
            # decoder
            img_fusion_temp = FSFusion_model.decoder_eval(f)
            img_fusion_blocks.append(img_fusion_temp)
        img_fusion_list = utils.recons_fusion_images(img_fusion_blocks, h, w)

    ############################ multi outputs ##############################################

    for img_fusion in img_fusion_list:
        # file_name = str(index) + '.png'
        if index < 10:
            file_name = '0' + str(index) + '.png'
        else:
            file_name = str(index) + '.png'
        output_path = output_path_root + file_name
        # save images
        utils.save_image_test(img_fusion, output_path)
        print(output_path)


def main():
    # run demo
    test_path_ir = "images/Test_ir/"
    test_path_vi = "images/Test_vi/"
    # test_path_ir = "images/RoadScene43/ir/"
    # test_path_vi = "images/RoadScene43/vi/"
    # test_path = "images/IV_images/"
    deepsupervision = False  # true for deeply supervision
    fusion_dataset = 'TNO'

    with torch.no_grad():
        if deepsupervision:
            model_path = args.model_deepsuperssh
        else:
            model_path = args.model_default
        model = load_model(model_path, deepsupervision)

        output_path = './outputs/' + fusion_dataset
        if os.path.exists(output_path) is False:
            os.mkdir(output_path)
        output_path = output_path + '/'

        f_type = fusion_dataset
        print('Processing......  ' + f_type)

        for i in range(42):
            index = i + 1
            if index < 10:
                infrared_path = test_path_ir + '0' + str(index) + '.png'
                visible_path = test_path_vi + '0' + str(index) + '.png'
            else:
                infrared_path = test_path_ir + str(index) + '.png'
                visible_path = test_path_vi + str(index) + '.png'
            # path = test_path + 'VIS' +str(index) + '.png'
            # infrared_path = test_path_ir + str(index) + '.png'
            # visible_path = test_path_vi + str(index) + '.png'
            run_demo(model, infrared_path, visible_path, output_path, index, f_type)
        # run_test(model,infrared_path, output_path, index, f_type)
    print('Done......')


if __name__ == '__main__':
    main()
