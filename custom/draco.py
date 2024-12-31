import os
import torch
import numpy as np
import open3d as o3d

def attribute_reform(anchor, attribute_list):
    """
    anchor : [N, 3], torch
    attribute_list : [[N, M], ...], torch
    
    Sort the anchor by x, y, z from small to large, apply the same permutation to the attribute_list.
    """
    if anchor.shape[1] != 3:
        raise ValueError("Input anchor must have shape (N, 3)")
    if len(attribute_list) == 0:
        raise ValueError("attribute_list must have at least one element")
    if anchor.shape[0] != attribute_list[0].shape[0]:
        raise ValueError("The first dimension of anchor and the first dimension of the first element of attribute_list must be the same")
    
    # Sort the anchor by x from small to large, if x is the same, sort by y, if y is the same, sort by z
    sorted_indices = torch.argsort(anchor[:, 0] + anchor[:, 1] * 1e-6 + anchor[:, 2] * 1e-12)
    anchor = anchor[sorted_indices]
    attribute_list = [attribute[sorted_indices] for attribute in attribute_list]
    
    return anchor, attribute_list

def write_anchor_to_ply(anchor, ply_file_path):
    """
    anchor : [N, 3], torch
    ply_file_path : str, path to the output ply file
    """
    if anchor.shape[1] != 3:
        raise ValueError("Input anchor must have shape (N, 3)")
    
    num_points = anchor.shape[0]
    with open(ply_file_path, "w") as f:
        # Write the PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        for point in anchor:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

def estimate_anchor_bits_using_draco(_quantized_v, temp_folder_path='./temp/', qp=16):
    anchor_np = _quantized_v.cpu().numpy().astype(np.int16)
    write_anchor_to_ply(anchor_np, os.path.join(temp_folder_path, "_temp_quantized_v.ply"))
    
    draco_encode_cmd = f"draco_encoder -i {os.path.join(temp_folder_path, '_temp_quantized_v.ply')} -o {os.path.join(temp_folder_path, '_temp_quantized_v.drc')} -point_cloud -qp {qp} -point_cloud"
    os.system(draco_encode_cmd)
    
    compressed_size = os.path.getsize(os.path.join(temp_folder_path, "_temp_quantized_v.drc")) * 8
    os.remove(os.path.join(temp_folder_path, "_temp_quantized_v.ply"))
    os.remove(os.path.join(temp_folder_path, "_temp_quantized_v.drc"))
    return compressed_size


def encode_anchor_using_draco(_quantized_v, _quantized_v_ply_file_path, qp=16):
    anchor_np = _quantized_v
    write_anchor_to_ply(anchor_np, _quantized_v_ply_file_path)
    draco_encode_cmd = f"draco_encoder -i {_quantized_v_ply_file_path} -o {_quantized_v_ply_file_path[:-4]}.drc -point_cloud -qp {qp} -point_cloud"
    os.system(draco_encode_cmd)
    compressed_size = os.path.getsize(_quantized_v_ply_file_path[:-4] + ".drc")
    return compressed_size


def decode_anchor_using_draco(_quantized_v_ply_file_path):
    _compressed_v_drc_file_path = _quantized_v_ply_file_path[:-4] + ".drc"
    decoded_path = _compressed_v_drc_file_path[:-4] + "_decoded.ply"
    draco_decode_cmd = f"draco_decoder -i {_compressed_v_drc_file_path} -o {decoded_path} -point_cloud"
    os.system(draco_decode_cmd)
    # read the decoded ply file
    points = o3d.io.read_point_cloud(decoded_path)
    array = np.asarray(points.points)
    return array
    
    