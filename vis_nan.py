import pickle
import os.path
import time

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.encodings import STE_binary, STE_multistep
from custom.encodings import STE_binary_with_ratio
from custom.model import entropy_skipping
from custom.recorder import record
from custom.recorder import record, init_recorder, get_logger, init_tb_writer, tb_writer, tb
from typing import *
from custom.prediction import K_neighbor_extraction, feat_collection, K_neighbor_extraction_batch

import os
import struct
import time
from functools import reduce

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
from torch_scatter import scatter_max

from utils.general_utils import (build_scaling_rotation, get_expon_lr_func,
                                 inverse_sigmoid, strip_symmetric)
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p
from utils.entropy_models import Entropy_bernoulli, Entropy_gaussian, Entropy_factorized

from utils.encodings import \
    STE_binary, STE_multistep, Quantize_anchor, \
    GridEncoder, Q_anchor, \
    anchor_round_digits, \
    get_binary_vxl_size

from utils.encodings_cuda import \
    encoder, decoder, \
    encoder_gaussian_chunk, decoder_gaussian_chunk
    
from custom.encodings import STE_binary_with_ratio
from custom.model import entropy_skipping
from custom.recorder import record
from custom.prediction import K_neighbor_extraction, feat_collection, K_neighbor_extraction_batch

entropy_gaussian = Entropy_gaussian(Q=1).cuda()

with open(f'/home/ethan/Project/ESGS/playground/higs_nan_record/step_5260_data.pkl', 'rb') as f:
    feat_chosen, grid_scaling_chosen, grid_offsets_chosen, mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat, Q_scaling, Q_offsets, binary_grid_masks_chosen = pickle.load(f)

# conduct entropy encoding
bit_feat = entropy_gaussian.forward(feat_chosen, mean, scale, Q_feat)
bit_scaling = entropy_gaussian.forward(grid_scaling_chosen, mean_scaling, scale_scaling, Q_scaling, )
bit_offsets = entropy_gaussian.forward(grid_offsets_chosen, mean_offsets, scale_offsets, Q_offsets)
    
    
    
    
    
    
    
    
    
with open(f'/home/ethan/Project/ESGS/playground/higs_nan_record/step_nan_feat_data.pkl', 'rb') as f:
    feat_chosen, grid_scaling_chosen, grid_offsets_chosen, mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat, Q_scaling, Q_offsets, binary_grid_masks_chosen = pickle.load(f)
    
# conduct entropy encoding
bit_feat = entropy_gaussian.forward(feat_chosen, mean, scale, Q_feat)
bit_scaling = entropy_gaussian.forward(grid_scaling_chosen, mean_scaling, scale_scaling, Q_scaling, )
bit_offsets = entropy_gaussian.forward(grid_offsets_chosen, mean_offsets, scale_offsets, Q_offsets)
bit_per_feat_param = torch.sum(bit_feat) / bit_feat.numel() * 1
if torch.isnan(bit_feat):
    print('bit_feat has nan')
    
    