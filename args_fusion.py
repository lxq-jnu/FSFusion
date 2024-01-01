#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/18 10:42
# @Author  : wangjuan
# @File    : arg_fusion.py
class args():
	# training args
	epochs = 2  #"number of training epochs, default is 2"
	batch_size = 4 #"batch size for training, default is 4"
	# the COCO dataset path in your computer
	# URL: http://images.cocodataset.org/zips/train2014.zip
	dataset = "./data/Train2014/"
	HEIGHT = 256
	WIDTH = 256

	save_model_dir_autoencoder = "./models"
	save_loss_dir = './models/loss/'

	cuda = 1
	ssim_weight = [1,10,100,1000,10000]
	ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']

	lr = 1e-4  #"learning rate, default is 0.0001"
	lr_light = 1e-4  # "learning rate, default is 0.001"
	log_interval = 100  #"number of images after which the training loss is logged, default is 500"
	resume = None   # default


	# for test, model_default is the model used in paper
	model_default = './models/Final.model'


