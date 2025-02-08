import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.general_utils import (build_scaling_rotation, get_expon_lr_func,
                                 inverse_sigmoid, strip_symmetric)


def neighbor_index_build(anchor, interval=0.001, K=10):
    """
    anchor: (N, 3) 目标点
    interval: float 目标点间隔
    K: int 临近点数量
    
    输出：
        neighbor_index: (X, Y, Z, K) 临近点索引
    """
    # 生成网格
    x = torch.arange(anchor[:, 0].min(), anchor[:, 0].max(), interval)
    y = torch.arange(anchor[:, 1].min(), anchor[:, 1].max(), interval)
    z = torch.arange(anchor[:, 2].min(), anchor[:, 2].max(), interval)
    X, Y, Z = torch.meshgrid(x, y, z)
    grid = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
    
    # 生成临近点索引
    dist = torch.cdist(anchor, grid, p=2) # (N, X*Y*Z)
    neighbor_index = torch.topk(dist, K, largest=False, dim=0)[1] # (K, X*Y*Z)
    
    return neighbor_index.reshape(x.shape[0], y.shape[0], z.shape[0], K)

def K_neighbor_extraction(anchor_1, anchor_2, K=10):
    """
    anchor_1: (N, 3) 要提取的临近点
    anchor_2: (M, 3) 目标点
    
    输出：
        neighbor_indices: (M, K) 临近点索引
    """
    dist = torch.cdist(anchor_1, anchor_2, p=2) # (N, M)
    neighbor_indices = torch.topk(dist, K, largest=False, dim=0)[1] # (K, M)

    return neighbor_indices

def K_neighbor_extraction_batch(anchor_1, anchor_2, K=10, batch_size=1000):
    """
    anchor_1: (N, 3) 源点云
    anchor_2: (M, 3) 目标点云
    batch_size: 每批次处理的目标点数（根据显存调整）
    
    输出：
        neighbor_indices: (M, K) 每个目标点的最近邻索引
    """
    M = anchor_2.shape[0]
    device = anchor_2.device
    neighbor_indices = []
    
    # 确保anchor_1在正确的设备上
    anchor_1 = anchor_1.to(device)
    
    for i in range(0, M, batch_size):
        end_idx = min(i + batch_size, M)
        current_batch = anchor_2[i:end_idx]
        
        # 计算当前批次的距离矩阵 (N, current_batch_size)
        dist = torch.cdist(anchor_1, current_batch, p=2)
        
        # 获取topk索引并转置 (current_batch_size, K)
        indices = torch.topk(dist, K, dim=0, largest=False)[1].T
        neighbor_indices.append(indices)
    
    # 合并所有批次结果
    neighbor_indices = torch.cat(neighbor_indices, dim=0)
    return neighbor_indices

def feat_collection(anchor_1, feat_1, offset_1, scale_1, neighbor_indices):
    """
    anchor_1: (N, 3) 要提取的临近点
    feat_1: (N, 50) 临近点特征
    offset_1: (N, 10, 3) 临近点偏移
    scale_1: (N, 3) 临近点尺度
    neighbor_indices: (M, K) 临近点索引
    
    输出：
        neighbor_feats: (M, K, 50) 临近点特征
        neighbor_anchor: (M, K, 3) 临近点坐标
        offset_feats: (M, K, 10, 3) 临近点偏移
        scale_feats: (M, K, 3) 临近点尺度
    """
    
    neighbor_feats = feat_1[neighbor_indices] # (K, M, 50)
    neighbor_anchor = anchor_1[neighbor_indices] # (K, M, 3)
    offset_feats = offset_1[neighbor_indices] # (K, M, 3)
    scale_feats = scale_1[neighbor_indices] # (K, M, 3)
    
    return neighbor_feats, neighbor_anchor, offset_feats, scale_feats
