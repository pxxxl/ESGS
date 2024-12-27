#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np
import subprocess
import torch
import torchvision
import json
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, get_project_path
from utils.encodings import get_binary_vxl_size
from lpipsPyTorch import lpips
from custom.encodings import STE_binary_with_ratio
from custom.model import entropy_skipping
from custom.recorder import record, init_recorder, get_logger, init_tb_writer, tb_writer, tb
import pickle
from torch.utils.tensorboard import SummaryWriter

bit2MB_scale = 8 * 1024 * 1024

def training(args_param, dataset, opt, pipe, testing_iterations, saving_iterations, debug_from, logger=None, ply_path=None):
    first_iter = 0
    gaussians = GaussianModel(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        dataset.use_feat_bank,
        n_features_per_level=args_param.n_features,
        log2_hashmap_size=args_param.log2,
        log2_hashmap_size_2D=args_param.log2_2D,
    )
    scene = Scene(dataset, gaussians, ply_path=ply_path)
    gaussians.update_anchor_bound()
    gaussians.set_steps(args_param)

    gaussians.training_setup(opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    torch.cuda.synchronize(); t_start = time.time()
    log_time_sub = 0
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, step=iteration)
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

        bit_per_param = render_pkg["bit_per_param"]
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        scaling_reg = scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg
        tb().add_scalar("train/loss/fidelity_loss", loss.item(), iteration)

        if bit_per_param is not None:
            _, bit_hash_grid, MB_hash_grid, _ = get_binary_vxl_size((gaussians.get_encoding_params()+1)/2)
            denom = gaussians._anchor.shape[0]*(gaussians.feat_dim+6+3*gaussians.n_offsets)
            bit_loss = args_param.lmbda * (bit_per_param + bit_hash_grid / denom)
            loss = loss + bit_loss
            tb().add_scalar("train/loss/bit_loss", bit_loss.item(), iteration)

            mask_loss = 5e-4 * torch.mean(torch.sigmoid(gaussians._mask))
            loss = loss + mask_loss
            tb().add_scalar("train/loss/mask_loss", mask_loss.item(), iteration)
            tb().add_scalar("train/loss/loss", bit_per_param, iteration)
            

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            torch.cuda.synchronize(); t_start_log = time.time()
            if iteration in testing_iterations:
                apply_testing(scene, (pipe, background), logger)
            if iteration == testing_iterations[-1]:
                apply_coding(gaussians, logger, args_param.model_path)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            torch.cuda.synchronize(); t_end_log = time.time()
            t_log = t_end_log - t_start_log
            log_time_sub += t_log

            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                if iteration not in range(3000, 4000):  # let the model get fit to quantization
                    # densification
                    if iteration > opt.update_from and iteration % opt.update_interval == 0:
                        gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
            tb().add_scalar("train/model/anchor_size", gaussians._anchor.shape[0], iteration)
            

    torch.cuda.synchronize(); t_end = time.time()
    logger.info("\n Total Training time: {}".format(t_end-t_start-log_time_sub))
    return


def apply_coding(gaussians: GaussianModel, logger=None, pre_path_name=''):
    log_info_1 = gaussians.estimate_final_bits()
    bit_stream_path = os.path.join(pre_path_name, 'bitstreams')
    os.makedirs(bit_stream_path, exist_ok=True)
    log_info_2 = gaussians.conduct_encoding(pre_path_name=bit_stream_path)
    log_info_3 = gaussians.conduct_decoding(pre_path_name=bit_stream_path)
    if logger:
        logger.info(log_info_1)
        logger.info(log_info_2)
        logger.info(log_info_3)
    

def apply_testing(scene : Scene, renderArgs, logger=None):
    scene.gaussians.eval()
    torch.cuda.empty_cache()
    validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0

            t_list = []

            for idx, viewpoint in enumerate(config['cameras']):
                torch.cuda.synchronize(); t_start = time.time()
                voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                render_output = render(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)
                image = torch.clamp(render_output["render"], 0.0, 1.0)
                time_sub = render_output["time_sub"]
                torch.cuda.synchronize(); t_end = time.time()
                t_list.append(t_end - t_start - time_sub)

                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
                ssim_test += ssim(image, gt_image).mean().double()
                lpips_test += lpips(image, gt_image, net_type='vgg').detach().mean().double()

            psnr_test /= len(config['cameras'])
            ssim_test /= len(config['cameras'])
            lpips_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])
            logger.info("\nEvaluating {}: L1 {} PSNR {} ssim {} lpips {}".format(config['name'], l1_test, psnr_test, ssim_test, lpips_test))
            test_fps = 1.0 / torch.tensor(t_list[0:]).mean()
            logger.info(f'Test FPS: {test_fps.item():.5f}')
    torch.cuda.empty_cache()

    scene.gaussians.train()


def resume_gaussians_from_bitstream(model_path):
    parser = ArgumentParser(description="Resume script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = load_args(model_path)
    dataset = lp.extract(args)
    args_param = args
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.feat_dim,
            dataset.n_offsets,
            dataset.voxel_size,
            dataset.update_depth,
            dataset.update_init_factor,
            dataset.update_hierachy_factor,
            dataset.use_feat_bank,
            n_features_per_level=args_param.n_features,
            log2_hashmap_size=args_param.log2,
            log2_hashmap_size_2D=args_param.log2_2D,
            decoded_version=True,
        )
        scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
        gaussians.eval()
        gaussians.conduct_encoding(os.path.join(args_param.model_path, 'bitstreams'))
    return scene, gaussians

def save_args(args, model_path):
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(os.path.join(model_path, 'bitstreams'), exist_ok=True)
    with open(os.path.join(model_path, 'bitstreams', "args.pkl"), 'wb') as f:
        pickle.dump(args, f)
        
def load_args(model_path):
    with open(os.path.join(model_path, 'bitstreams', "args.pkl"), 'rb') as f:
        args = pickle.load(f)
    return args

def main():
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log2", type=int, default = 13)
    parser.add_argument("--log2_2D", type=int, default = 15)
    parser.add_argument("--n_features", type=int, default = 4)
    parser.add_argument("--lmbda", type=float, default = 0.001)
    
    parser.add_argument("--step_begin_quantization", type=int, default=3000)
    parser.add_argument("--step_begin_RD_training", type=int, default=10000)
    # adjust the following op parameters to control densification
    # start_stat = 500
    # update_from = 1500
    # update_interval = 100
    # update_until = 15_000 
    args = parser.parse_args(sys.argv[1:])
    
    save_args(args, args.model_path)

    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)
    init_recorder("record.txt", model_path)
    init_tb_writer(os.path.join(model_path, 'runs'))

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(args, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.debug_from, logger)

if __name__ == "__main__":
    main()