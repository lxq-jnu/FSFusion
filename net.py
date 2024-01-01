#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 10:44
# @Author  : wangjuan
# @File    : net.py


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import FSFusion_strategy
from args_fusion import args
from torch.autograd import Variable

class UpsampleReshape_eval(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right%2 is 0.0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 is 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2


# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = F.relu(out, inplace=True)
        return out

# DeConvolution operation
class DeConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, destride,padding,outpadding,is_last=False):
        super(DeConvLayer, self).__init__()
        self.deconv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, destride,padding,outpadding)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.deconv2d(x)
        if self.is_last is False:
            out = F.relu(out, inplace=True)
        return out

# light version
class DenseBlock_light(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseBlock_light, self).__init__()
        out_channels_def = int(in_channels / 2)
        # out_channels_def = out_channels
        denseblock = []

        denseblock += [ConvLayer(in_channels, out_channels_def, kernel_size, stride),
                       ConvLayer(out_channels_def, out_channels, 1, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


# FSFusion network - light, no desnse
# Pytorch的数据格式为NCHW
class FSFusion_autoencoder(nn.Module):
    def __init__(self, nb_filter, input_nc=1, output_nc=1, deepsupervision=True):
        super(FSFusion_autoencoder, self).__init__()
        self.deepsupervision = deepsupervision
        block = DenseBlock_light
        output_filter = 16
        kernel_size = 3
        stride = 1
        destride = 2
        padding = 1
        outpadding = 1

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)
        self.up_eval = UpsampleReshape_eval()

        # encoder
        self.conv0 = ConvLayer(input_nc, output_filter, 1, stride)
        self.EB1_0 = block(output_filter, nb_filter[0], kernel_size, 1)
        self.EB2_0 = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        self.EB3_0 = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        self.EB4_0 = block(nb_filter[2], nb_filter[3], kernel_size, 1)

        # decoder
        self.DB3_1 = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1)
        self.DB2_1 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)

        # Non local attention
        self.NL1_1 = ConvLayer(nb_filter[0],nb_filter[0],kernel_size,stride)
        self.NL1_2 = ConvLayer(nb_filter[0],nb_filter[0],kernel_size,stride)
        self.NL2_1 = ConvLayer(nb_filter[1],nb_filter[0],kernel_size,stride)
        self.NL3_1 = ConvLayer(nb_filter[2],nb_filter[0],kernel_size,stride)
        self.NL3_2 = ConvLayer(nb_filter[2],nb_filter[3], kernel_size, stride)
        self.NL4_1 = ConvLayer(nb_filter[3],nb_filter[0],kernel_size,stride)
        self.NL4_2 = ConvLayer(nb_filter[3],nb_filter[3], kernel_size, stride)
        self.NL_G = ConvLayer(nb_filter[0]*4,nb_filter[0], kernel_size, stride)
        self.NL_F = ConvLayer(nb_filter[3]*2,nb_filter[3],kernel_size,stride)
        self.NL_R = ConvLayer(nb_filter[3],nb_filter[3],1,stride)


        # edge
        self.edge1 = ConvLayer(input_nc,output_filter,kernel_size, stride)
        self.edge2 = ConvLayer(output_filter,output_filter*2,kernel_size,stride)
        self.edge3 = ConvLayer(output_filter*2,nb_filter[0],kernel_size,stride)
        self.edge4 = ConvLayer(nb_filter[0],nb_filter[0],kernel_size,stride)

        #edge guidance1
        self.guidance1_conv1 = ConvLayer(nb_filter[0],nb_filter[0],kernel_size,stride)
        self.guidance1_conv2 = ConvLayer(nb_filter[0],nb_filter[0],kernel_size,stride)
        self.guidance1_conv3 = ConvLayer(nb_filter[0],nb_filter[0],kernel_size,stride)
        self.guidance1_conv4 = ConvLayer(nb_filter[0],nb_filter[0],kernel_size,stride)
        self.guidance1_conv5 = ConvLayer(2,nb_filter[0],kernel_size, stride)

        # edge guidance2
        self.guidance2_conv1 = ConvLayer(nb_filter[0], nb_filter[0],kernel_size,stride)
        self.guidance2_conv2 = ConvLayer(nb_filter[0], nb_filter[1],kernel_size,stride)
        self.guidance2_conv3 = ConvLayer(nb_filter[0], nb_filter[0], kernel_size, stride)
        self.guidance2_conv4 = ConvLayer(nb_filter[0], nb_filter[1], kernel_size, stride)
        self.guidance2_conv5 = ConvLayer(2, nb_filter[1], kernel_size, stride)

        # edge guidance3
        self.guidance3_conv1 = ConvLayer(nb_filter[0], nb_filter[1], kernel_size, stride)
        self.guidance3_conv2 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.guidance3_conv3 = ConvLayer(nb_filter[0], nb_filter[1], kernel_size, stride)
        self.guidance3_conv4 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.guidance3_conv5 = ConvLayer(2, nb_filter[2], kernel_size, stride)


        if self.deepsupervision:
            self.conv1 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv2 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv3 = ConvLayer(nb_filter[0], output_nc, 1, stride)
        else:
            self.conv_out = ConvLayer(nb_filter[0], output_nc, 1, stride)

    def encoder(self, input):
        x = self.conv0(input)
        x1_0 = self.EB1_0(x)
        x2_0 = self.EB2_0(self.pool(x1_0))
        x3_0 = self.EB3_0(self.pool(x2_0))
        x4_0 = self.EB4_0(self.pool(x3_0))
        return [x1_0, x2_0, x3_0, x4_0]


    def edge_feature(self,edge_img):
        x1_0 = self.edge1(edge_img)
        x2_0 = self.edge2(x1_0)
        x3_0 = self.edge3(x2_0)
        x4_0 = self.edge4(x3_0)
        return x4_0

    def edge_guidance1(self,edge_feature,enx1_0):
        conv1 = self.guidance1_conv1(edge_feature)
        edge_1 = self.guidance1_conv2(conv1)
        edge_feature_1 = edge_1 * enx1_0
        edge_feature_1_avgpool = torch.mean(edge_feature_1,dim=1,keepdim=True)
        edge_feature_1_maxpool,_ = torch.max(edge_feature_1,dim=1,keepdim=True)
        edge_enhance_1 = self.guidance1_conv5(torch.cat([edge_feature_1_avgpool,edge_feature_1_maxpool],1))
        return edge_enhance_1 * enx1_0


    def edge_guidance2(self,edge_feature,enx2_0):
        conv1 = self.guidance2_conv1(edge_feature)
        edge_2 = self.guidance2_conv2(self.pool(conv1))
        edge_feature_2 = edge_2 * enx2_0
        edge_feature_2_avgpool = torch.mean(edge_feature_2, dim=1, keepdim=True)
        edge_feature_2_maxpool, _ = torch.max(edge_feature_2, dim=1, keepdim=True)
        edge_enhance_2 = self.guidance2_conv5(torch.cat([edge_feature_2_avgpool, edge_feature_2_maxpool],1))
        return edge_enhance_2 * enx2_0


    def edge_guidance3(self,edge_feature,enx3_0):
        conv1 = self.guidance3_conv1(self.pool(edge_feature))
        edge_3 = self.guidance3_conv2(self.pool(conv1))
        edge_feature_3 = edge_3 * enx3_0
        edge_feature_3_avgpool = torch.mean(edge_feature_3, dim=1, keepdim=True)
        edge_feature_3_maxpool, _ = torch.max(edge_feature_3, dim=1, keepdim=True)
        edge_enhance_3 = self.guidance3_conv5(torch.cat([edge_feature_3_avgpool, edge_feature_3_maxpool],1))
        return edge_enhance_3 * enx3_0




    def non_local(self, f_en):
        x1_0 = self.NL1_1(self.pool(f_en[0]))
        x1_1 = self.NL1_2(self.pool(x1_0))
        x2_0 = self.NL2_1(self.pool(f_en[1]))
        x3_0 = self.NL3_1(f_en[2])
        x3_1 = self.NL3_2(f_en[2])
        x4_0 = self.NL4_1(self.up(f_en[3]))
        x4_1 = self.NL4_2(self.up(f_en[3]))
        x_G = self.NL_G(torch.cat([x1_1, x2_0, x3_0, x4_0], dim=1))
        x_F = self.NL_F(torch.cat([x3_1, x4_1], dim=1))
        shape_G = x_G.size()
        shape_F = x_F.size()
        x_G = torch.reshape(x_G, [shape_G[0], shape_G[1], shape_G[2] * shape_G[3]])
        x_F = torch.reshape(x_F, [shape_F[0], shape_F[1], shape_F[2] * shape_F[3]])
        DFM = []
        G_L_attention = torch.softmax(torch.matmul(torch.transpose(x_G, 1, 2), x_G), dim=1)
        DFM = torch.matmul(G_L_attention, torch.transpose(x_F, 1, 2))
        DFM_transpose = torch.transpose(DFM, 1, 2)
        attention = torch.reshape(DFM_transpose, [shape_F[0], shape_F[1], shape_F[2], shape_F[3]])
        result = self.NL_R(self.pool(attention))
        return result + f_en[3]


    def fusion(self, en1, en2, p_type):
        # attention weight
        fusion_function = FSFusion_strategy.attention_fusion_weight

        f1_0 = fusion_function(en1[0], en2[0], p_type)
        f2_0 = fusion_function(en1[1], en2[1], p_type)
        f3_0 = fusion_function(en1[2], en2[2], p_type)
        f4_0 = fusion_function(en1[3], en2[3], p_type)
        return [f1_0, f2_0, f3_0, f4_0]


    def decoder_train(self, f_en,dfm):

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up(dfm)], 1))
        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up(x3_1)], 1))
        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up(x2_1)], 1))


        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x2_1)
            output3 = self.conv3(x3_1)
            # output4 = self.conv4(x1_4)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_1)
            # print(output.shape)
            return [output]

    def decoder_eval(self, f_en,dfm):

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up(dfm)], 1))
        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up(x3_1)], 1))
        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up(x2_1)], 1))

        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x2_1)
            output3 = self.conv3(x3_1)
            # output4 = self.conv4(x1_4)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_1)
            return [output]