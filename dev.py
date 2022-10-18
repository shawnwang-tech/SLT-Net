import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from PIL import Image
from lib import VideoModel_pvtv2 as Network
from dataloaders import test_dataloader
import imageio
import pdb

import torch.onnx


import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  type=str, default='MoCA')
parser.add_argument('--testsplit',  type=str, default='TestDataset_per_sq')
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--trainsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshot/COD10K/Net_epoch_best.pth')
parser.add_argument('--pretrained_cod10k', default=None,
                        help='path to the pretrained Resnet')
opt = parser.parse_args()

model = Network(opt)

load_pram = torch.load('/Users/mac/Downloads/snapshot/Net_epoch_MoCA_short_term_pseudo.pth', map_location=torch.device('cpu'))

onnx_fp = '/Users/mac/Downloads/sltnet.onnx'

model.load_state_dict(load_pram)

input_names = ["image"]  
output_names = ["class"]  
dynamic_axes = {'image': {0: '-1'}, 'class': {0: '-1'}} 
dummy_input = [torch.randn(1, 3, 352, 352), torch.randn(1, 3, 352, 352), torch.randn(1, 3, 352, 352)]

torch.onnx.export(model, dummy_input, onnx_fp, input_names = input_names, dynamic_axes = dynamic_axes, output_names = output_names, opset_version=11, verbose=True) 

print('debug')

