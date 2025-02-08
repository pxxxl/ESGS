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

def generate_neural_gaussians_hirachical(viewpoint_camera, pc_list : GaussianModel, visible_mask_list=None, is_training=False, step=0, active_gaussian=1):
    ## view frustum filtering for acceleration

    time_sub = 0

    if visible_mask_list is None:
        visible_mask_list = [torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device) for pc in pc_list]
    
    low_anchor = torch.zeros(0, 3, device=pc_list[0].get_anchor.device)
    low_feat = torch.zeros(0, pc_list[0].feat_dim, device=pc_list[0].get_anchor.device)
    low_grid_offsets = torch.zeros(0, pc_list[0].n_offsets, 3, device=pc_list[0].get_anchor.device)
    low_grid_scaling = torch.zeros(0, 6, device=pc_list[0].get_anchor.device)
    low_binary_grid_masks = torch.zeros(0, pc_list[0].n_offsets, 1, device=pc_list[0].get_anchor.device)
    low_mask_anchor = torch.zeros(0, device=pc_list[0].get_anchor.device)
    low_visible_mask = torch.zeros(0, dtype=torch.bool, device=pc_list[0].get_anchor.device)

    for i in range(active_gaussian):
        low_pc = pc_list[i]
        low_visible_mask = torch.cat([low_visible_mask, visible_mask_list[i]], dim=0)
        low_anchor = torch.cat([low_anchor, low_pc.get_anchor], dim=0)
        low_feat = torch.cat([low_feat, low_pc._anchor_feat], dim=0)
        low_grid_offsets = torch.cat([low_grid_offsets, low_pc._offset], dim=0)
        low_grid_scaling = torch.cat([low_grid_scaling, low_pc.get_scaling], dim=0)
        
        low_binary_grid_masks = torch.cat([low_binary_grid_masks, low_pc.get_mask], dim=0)
        low_mask_anchor = torch.cat([low_mask_anchor, low_pc.get_mask_anchor], dim=0)
        
    coord_max = low_anchor.max(dim=0)[0]
    coord_min = low_anchor.min(dim=0)[0]
            
    pc = pc_list[active_gaussian]
    visible_mask = visible_mask_list[active_gaussian]

    anchor = pc.get_anchor[visible_mask]
    #
    feat = pc._anchor_feat[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    binary_grid_masks = pc.get_mask[visible_mask]
    mask_anchor = pc.get_mask_anchor[visible_mask]
    mask_anchor_bool = mask_anchor.to(torch.bool)
    mask_anchor_rate = (mask_anchor.sum() / mask_anchor.numel()).detach()

    bit_per_param = None
    bit_per_feat_param = None
    bit_per_scaling_param = None
    bit_per_offsets_param = None
    Q_feat = 1
    Q_scaling = 0.001
    Q_offsets = 0.2
    if is_training:
        if step > pc.step_begin_quantization and step <= pc.step_full_RD:
            # quantization
            feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
            grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
            grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets

        if step == pc.step_full_RD:
            pc.update_anchor_bound()
            pc.build_anchor_neighbor_index_cache(low_anchor)

        if step > pc.step_full_RD:
            anchor_neighbor_index = pc.get_anchor_neighbor_index()
            neighbor_indices = anchor_neighbor_index[visible_mask]
            n_feat, n_anchor, n_offsets, n_scaling = feat_collection(low_anchor, low_feat, low_grid_offsets, low_grid_scaling, neighbor_indices)
            n_offsets = torch.reshape(n_offsets, (-1, pc_list[0].n_offsets, 3 * pc_list[0].n_offsets))
            feat_all = torch.cat([n_feat, n_offsets, n_scaling], dim=2)
            hir_feat_context = pc.hir_entropy_prediction(feat_all, n_anchor, coord_max, coord_min)
            
            mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(hir_feat_context, split_size_or_sections=[pc.feat_dim, pc.feat_dim, 6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1, 1], dim=-1)

            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))
            feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
            grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
            grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets.unsqueeze(1)

            choose_idx = torch.rand_like(anchor[:, 0]) <= 0.05
            choose_idx = choose_idx & mask_anchor_bool
            feat_chosen = feat[choose_idx]
            grid_scaling_chosen = grid_scaling[choose_idx]
            grid_offsets_chosen = grid_offsets[choose_idx].view(-1, 3*pc.n_offsets)
            mean = mean[choose_idx]
            scale = scale[choose_idx]
            mean_scaling = mean_scaling[choose_idx]
            scale_scaling = scale_scaling[choose_idx]
            mean_offsets = mean_offsets[choose_idx]
            scale_offsets = scale_offsets[choose_idx]
            Q_feat = Q_feat[choose_idx]
            Q_scaling = Q_scaling[choose_idx]
            Q_offsets = Q_offsets[choose_idx]
            binary_grid_masks_chosen = binary_grid_masks[choose_idx].repeat(1, 1, 3).view(-1, 3*pc.n_offsets)
            bit_feat = pc.entropy_gaussian.forward(feat_chosen, mean, scale, Q_feat, pc._anchor_feat.mean())
            bit_scaling = pc.entropy_gaussian.forward(grid_scaling_chosen, mean_scaling, scale_scaling, Q_scaling, pc.get_scaling.mean())
            bit_offsets = pc.entropy_gaussian.forward(grid_offsets_chosen, mean_offsets, scale_offsets, Q_offsets, pc._offset.mean())
            bit_offsets = bit_offsets * binary_grid_masks_chosen
            bit_per_feat_param = torch.sum(bit_feat) / bit_feat.numel() * mask_anchor_rate
            bit_per_scaling_param = torch.sum(bit_scaling) / bit_scaling.numel() * mask_anchor_rate
            bit_per_offsets_param = torch.sum(bit_offsets) / bit_offsets.numel() * mask_anchor_rate
            bit_per_param = (torch.sum(bit_feat) + torch.sum(bit_scaling) + torch.sum(bit_offsets)) / \
                            (bit_feat.numel() + bit_scaling.numel() + bit_offsets.numel()) * mask_anchor_rate
            tb().add_scalar(f"train/hir_{active_gaussian}/bit/feat", bit_per_feat_param, step)
            tb().add_scalar(f"train/hir_{active_gaussian}/bit/scaling", bit_per_scaling_param, step)
            tb().add_scalar(f"train/hir_{active_gaussian}/bit/offset", bit_per_offsets_param, step)
            tb().add_scalar(f"train/hir_{active_gaussian}/bit/param", bit_per_param, step)
            
            if step == 5260:
                with open("step_5260_data.pkl", "wb") as f:
                    import pickle
                    pickle.dump((feat_chosen, grid_scaling_chosen, grid_offsets_chosen, mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat, Q_scaling, Q_offsets, binary_grid_masks_chosen, step, mask_anchor_rate), f)
                    
            if torch.isnan(bit_per_param) and step < 10000:
                with open("step_nan_feat_data.pkl", "wb") as f:
                    import pickle
                    pickle.dump((feat_chosen, grid_scaling_chosen, grid_offsets_chosen, mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat, Q_scaling, Q_offsets, binary_grid_masks_chosen, step, mask_anchor_rate), f)
           
    elif not pc.decoded_version:
        torch.cuda.synchronize(); t1 = time.time()
        anchor_neighbor_index = pc.get_anchor_neighbor_index()
        neighbor_indices = anchor_neighbor_index[visible_mask]
        n_feat, n_anchor, n_offsets, n_scaling = feat_collection(low_anchor, low_feat, low_grid_offsets, low_grid_scaling, neighbor_indices)
        n_offsets = torch.reshape(n_offsets, (-1, pc_list[0].n_offsets, 3 * pc_list[0].n_offsets))
        feat_all = torch.cat([n_feat, n_offsets, n_scaling], dim=2)
        hir_feat_context = pc.hir_entropy_prediction(feat_all, n_anchor, coord_max, coord_min)
        
        mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            torch.split(hir_feat_context, split_size_or_sections=[pc.feat_dim, pc.feat_dim, 6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1, 1], dim=-1)

        Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
        Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
        Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))  # [N_visible_anchor, 1]
        feat = (STE_multistep.apply(feat, Q_feat, pc._anchor_feat.mean())).detach()
        grid_scaling = (STE_multistep.apply(grid_scaling, Q_scaling, pc.get_scaling.mean())).detach()
        grid_offsets = (STE_multistep.apply(grid_offsets, Q_offsets.unsqueeze(1), pc._offset.mean())).detach()
        torch.cuda.synchronize(); time_sub = time.time() - t1

    else:
        pass

    ob_view = anchor - viewpoint_camera.camera_center
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)  # [3+1]

        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)  # [N_visible_anchor, 1, 3]

        feat = feat.unsqueeze(dim=-1)  # feat: [N_visible_anchor, 32]
        feat = \
            feat[:, ::4, :1].repeat([1, 4, 1])*bank_weight[:, :, :1] + \
            feat[:, ::2, :1].repeat([1, 2, 1])*bank_weight[:, :, 1:2] + \
            feat[:, ::1, :1]*bank_weight[:, :, 2:]
        feat = feat.squeeze(dim=-1)  # [N_visible_anchor, 32]

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N_visible_anchor, 32+3+1]

    neural_opacity = pc.get_opacity_mlp(cat_local_view)  # [N_visible_anchor, K]
    neural_opacity = neural_opacity.reshape([-1, 1])  # [N_visible_anchor*K, 1]
    neural_opacity = neural_opacity * binary_grid_masks.view(-1, 1)
    mask = (neural_opacity > 0.0)
    mask = mask.view(-1)  # [N_visible_anchor*K]

    # select opacity
    opacity = neural_opacity[mask]  # [N_opacity_pos_gaussian, 1]
    # get offset's color
    color = pc.get_color_mlp(cat_local_view)  # [N_visible_anchor, K*3]
    color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])  # [N_visible_anchor*K, 3]

    # get offset's cov
    scale_rot = pc.get_cov_mlp(cat_local_view)  # [N_visible_anchor, K*7]
    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])  # [N_visible_anchor*K, 7]

    offsets = grid_offsets.view([-1, 3])  # [N_visible_anchor*K, 3]

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)  # [N_visible_anchor, 6+3]
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)  # [N_visible_anchor*K, 6+3]
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets],
                                 dim=-1)  # [N_visible_anchor*K, (6+3)+3+7+3]
    masked = concatenated_all[mask]  # [N_opacity_pos_gaussian, (6+3)+3+7+3]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

    # post-process cov
    scaling = scaling_repeat[:, 3:] * torch.sigmoid(
        scale_rot[:, :3])
    rot = pc.rotation_activation(scale_rot[:, 3:7])  # [N_opacity_pos_gaussian, 4]

    offsets = offsets * scaling_repeat[:, :3]  # [N_opacity_pos_gaussian, 3]
    xyz = repeat_anchor + offsets  # [N_opacity_pos_gaussian, 3]
    
    

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param
    else:
        return xyz, color, opacity, scaling, rot, time_sub

def generate_neural_gaussians_base(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False, step=0):
    ## view frustum filtering for acceleration

    time_sub = 0

    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    anchor = pc.get_anchor[visible_mask]
    #
    feat = pc._anchor_feat[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    binary_grid_masks = pc.get_mask[visible_mask]
    mask_anchor = pc.get_mask_anchor[visible_mask]
    mask_anchor_bool = mask_anchor.to(torch.bool)
    mask_anchor_rate = (mask_anchor.sum() / mask_anchor.numel()).detach()

    bit_per_param = None
    bit_per_feat_param = None
    bit_per_scaling_param = None
    bit_per_offsets_param = None
    Q_feat = 1
    Q_scaling = 0.001
    Q_offsets = 0.2
    if is_training:
        if step > pc.step_begin_quantization and step <= pc.step_full_RD:
            # quantization
            feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
            grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
            grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets

        if step == pc.step_full_RD:
            pc.update_anchor_bound()

        if step > pc.step_full_RD:
            feat_context = pc.calc_interp_feat(anchor)
            feat_context = pc.get_grid_mlp(feat_context)
            mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(feat_context, split_size_or_sections=[pc.feat_dim, pc.feat_dim, 6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1, 1], dim=-1)

            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))
            feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
            grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
            grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets.unsqueeze(1)

            choose_idx = torch.rand_like(anchor[:, 0]) <= 0.05
            choose_idx = choose_idx & mask_anchor_bool
            feat_chosen = feat[choose_idx]
            grid_scaling_chosen = grid_scaling[choose_idx]
            grid_offsets_chosen = grid_offsets[choose_idx].view(-1, 3*pc.n_offsets)
            mean = mean[choose_idx]
            scale = scale[choose_idx]
            mean_scaling = mean_scaling[choose_idx]
            scale_scaling = scale_scaling[choose_idx]
            mean_offsets = mean_offsets[choose_idx]
            scale_offsets = scale_offsets[choose_idx]
            Q_feat = Q_feat[choose_idx]
            Q_scaling = Q_scaling[choose_idx]
            Q_offsets = Q_offsets[choose_idx]
            binary_grid_masks_chosen = binary_grid_masks[choose_idx].repeat(1, 1, 3).view(-1, 3*pc.n_offsets)
            bit_feat = pc.entropy_gaussian.forward(feat_chosen, mean, scale, Q_feat, pc._anchor_feat.mean())
            bit_scaling = pc.entropy_gaussian.forward(grid_scaling_chosen, mean_scaling, scale_scaling, Q_scaling, pc.get_scaling.mean())
            bit_offsets = pc.entropy_gaussian.forward(grid_offsets_chosen, mean_offsets, scale_offsets, Q_offsets, pc._offset.mean())
            bit_offsets = bit_offsets * binary_grid_masks_chosen
            bit_per_feat_param = torch.sum(bit_feat) / bit_feat.numel() * mask_anchor_rate
            bit_per_scaling_param = torch.sum(bit_scaling) / bit_scaling.numel() * mask_anchor_rate
            bit_per_offsets_param = torch.sum(bit_offsets) / bit_offsets.numel() * mask_anchor_rate
            bit_per_param = (torch.sum(bit_feat) + torch.sum(bit_scaling) + torch.sum(bit_offsets)) / \
                            (bit_feat.numel() + bit_scaling.numel() + bit_offsets.numel()) * mask_anchor_rate
            tb().add_scalar("train/base/bit/feat", bit_per_feat_param, step)
            tb().add_scalar("train/base/bit/scaling", bit_per_scaling_param, step)
            tb().add_scalar("train/base/bit/offsets", bit_per_offsets_param, step)
            tb().add_scalar("train/base/bit/param", bit_per_param, step)
           
    elif not pc.decoded_version:
        torch.cuda.synchronize(); t1 = time.time()
        feat_context = pc.calc_interp_feat(anchor)
        mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            torch.split(pc.get_grid_mlp(feat_context), split_size_or_sections=[pc.feat_dim, pc.feat_dim, 6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1, 1], dim=-1)

        Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
        Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
        Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))  # [N_visible_anchor, 1]
        feat = (STE_multistep.apply(feat, Q_feat, pc._anchor_feat.mean())).detach()
        grid_scaling = (STE_multistep.apply(grid_scaling, Q_scaling, pc.get_scaling.mean())).detach()
        grid_offsets = (STE_multistep.apply(grid_offsets, Q_offsets.unsqueeze(1), pc._offset.mean())).detach()
        torch.cuda.synchronize(); time_sub = time.time() - t1

    else:
        pass

    ob_view = anchor - viewpoint_camera.camera_center
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)  # [3+1]

        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)  # [N_visible_anchor, 1, 3]

        feat = feat.unsqueeze(dim=-1)  # feat: [N_visible_anchor, 32]
        feat = \
            feat[:, ::4, :1].repeat([1, 4, 1])*bank_weight[:, :, :1] + \
            feat[:, ::2, :1].repeat([1, 2, 1])*bank_weight[:, :, 1:2] + \
            feat[:, ::1, :1]*bank_weight[:, :, 2:]
        feat = feat.squeeze(dim=-1)  # [N_visible_anchor, 32]

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N_visible_anchor, 32+3+1]

    neural_opacity = pc.get_opacity_mlp(cat_local_view)  # [N_visible_anchor, K]
    neural_opacity = neural_opacity.reshape([-1, 1])  # [N_visible_anchor*K, 1]
    neural_opacity = neural_opacity * binary_grid_masks.view(-1, 1)
    mask = (neural_opacity > 0.0)
    mask = mask.view(-1)  # [N_visible_anchor*K]

    # select opacity
    opacity = neural_opacity[mask]  # [N_opacity_pos_gaussian, 1]

    # get offset's color
    color = pc.get_color_mlp(cat_local_view)  # [N_visible_anchor, K*3]
    color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])  # [N_visible_anchor*K, 3]

    # get offset's cov
    scale_rot = pc.get_cov_mlp(cat_local_view)  # [N_visible_anchor, K*7]
    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])  # [N_visible_anchor*K, 7]

    offsets = grid_offsets.view([-1, 3])  # [N_visible_anchor*K, 3]

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)  # [N_visible_anchor, 6+3]
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)  # [N_visible_anchor*K, 6+3]
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets],
                                 dim=-1)  # [N_visible_anchor*K, (6+3)+3+7+3]
    masked = concatenated_all[mask]  # [N_opacity_pos_gaussian, (6+3)+3+7+3]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

    # post-process cov
    scaling = scaling_repeat[:, 3:] * torch.sigmoid(
        scale_rot[:, :3])
    rot = pc.rotation_activation(scale_rot[:, 3:7])  # [N_opacity_pos_gaussian, 4]

    offsets = offsets * scaling_repeat[:, :3]  # [N_opacity_pos_gaussian, 3]
    xyz = repeat_anchor + offsets  # [N_opacity_pos_gaussian, 3]

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param
    else:
        return xyz, color, opacity, scaling, rot, time_sub


def render(viewpoint_camera, pc_list : List[GaussianModel], pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask_list=None, retain_grad=False, step=0, active_gaussians=0, is_training=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    is_training = is_training
    xyz_list = []
    color_list = []
    opacity_list = []
    scaling_list = []
    rot_list = []
    bit_per_param = None
    bit_per_feat_param = None
    bit_per_scaling_param = None
    bit_per_offsets_param = None
    time_sub_total = 0
    xyz_len_list = []
    
    if active_gaussians == 0:
        if is_training:
            xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param = generate_neural_gaussians_base(viewpoint_camera, pc_list[0], visible_mask_list[0], is_training=is_training, step=step)
        else:
            xyz, color, opacity, scaling, rot, time_sub = generate_neural_gaussians_base(viewpoint_camera, pc_list[0], visible_mask_list[0], is_training=is_training, step=step)
            time_sub_total += time_sub
        xyz_len_list.append(xyz.shape[0])
    else:
        for i in range(active_gaussians):
            if i == 0:
                xyz, color, opacity, scaling, rot, time_sub = generate_neural_gaussians_base(viewpoint_camera, pc_list[0], visible_mask_list[0], is_training=False, step=step)
                xyz_list.append(xyz)
                color_list.append(color)
                opacity_list.append(opacity)
                scaling_list.append(scaling)
                rot_list.append(rot)
                time_sub_total += time_sub
            else:
                xyz, color, opacity, scaling, rot, time_sub = generate_neural_gaussians_hirachical(viewpoint_camera, pc_list, visible_mask_list, is_training=False, step=step, active_gaussian=i)
                xyz_list.append(xyz)
                color_list.append(color)
                opacity_list.append(opacity)
                scaling_list.append(scaling)
                rot_list.append(rot)
                time_sub_total += time_sub
            xyz_len_list.append(xyz.shape[0])
        if is_training:
            xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param = generate_neural_gaussians_hirachical(viewpoint_camera, pc_list, visible_mask_list, is_training=is_training, step=step, active_gaussian=active_gaussians)
        else:
            xyz, color, opacity, scaling, rot, time_sub = generate_neural_gaussians_hirachical(viewpoint_camera, pc_list, visible_mask_list, is_training=is_training, step=step, active_gaussian=active_gaussians)
            time_sub_total += time_sub
        xyz_len_list.append(xyz.shape[0])
        xyz_list.append(xyz)
        color_list.append(color)
        opacity_list.append(opacity)
        scaling_list.append(scaling)
        rot_list.append(rot)
        
        # concat the list
        xyz = torch.cat(xyz_list, dim=0)
        color = torch.cat(color_list, dim=0)
        opacity = torch.cat(opacity_list, dim=0)
        scaling = torch.cat(scaling_list, dim=0)
        rot = torch.cat(rot_list, dim=0)
        

    screenspace_points = torch.zeros_like(xyz, dtype=pc_list[0].get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "xyz_len_list": xyz_len_list,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "bit_per_param": bit_per_param,
                "bit_per_feat_param": bit_per_feat_param,
                "bit_per_scaling_param": bit_per_scaling_param,
                "bit_per_offsets_param": bit_per_offsets_param,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "time_sub": time_sub_total,
                }


def prefilter_voxel(viewpoint_camera, pc_list: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None, active_gaussians:int=0):
    """
    Render the scene. 

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    anchor_list = [pc.get_anchor for i, pc in enumerate(pc_list) if i <= active_gaussians]
    scaling_list = [pc.get_scaling for i, pc in enumerate(pc_list) if i <= active_gaussians]
    rotations_list = [pc.get_rotation for i, pc in enumerate(pc_list) if i <= active_gaussians]
    concated_anchor = torch.cat(anchor_list, dim=0)
    concated_scaling = torch.cat(scaling_list, dim=0)
    concated_rotations = torch.cat(rotations_list, dim=0)
    screenspace_points = torch.zeros_like(concated_anchor, dtype=concated_anchor.dtype, requires_grad=True,
                                          device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = concated_anchor

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = concated_scaling
    rotations = concated_rotations

    radii_pure = rasterizer.visible_filter(
        means3D=means3D,
        scales=scales[:, :3],
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,  # None
    )
    
    mask = radii_pure > 0
    anchor_len_list = [len(anchor_list[i]) for i in range(len(anchor_list))]
    split_mask = torch.split(mask, anchor_len_list, dim=0)

    return split_mask
