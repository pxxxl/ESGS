import os
import torch
import numpy as np
import open3d as o3d

def npy_to_ply(npy_file, ply_file):
    """
    将 npy 文件转换为 PLY 文件，数据类型为 uint16。
    """
    array = np.load(npy_file).astype(np.int32)
    if array.shape[1] != 3:
        raise ValueError("Input npy array must have shape (N, 3)")

    num_points = array.shape[0]

    with open(ply_file, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property int32 x\n")
        f.write("property int32 y\n")
        f.write("property int32 z\n")
        f.write("end_header\n")
        
        for point in array:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")
    
    print(f"Saved PLY file: {ply_file}")

def compress_ply_with_draco(input_ply, output_ply, quantization_bits=16):
    cmd = f"draco_encoder -i {input_ply} -o {output_ply} -qp {quantization_bits} -point_cloud"
    os.system(cmd)
    print(f"Compressed PLY file: {output_ply}")
    
def decompress_ply_with_draco(input_ply, output_ply):
    cmd = f"draco_decoder -i {input_ply} -o {output_ply}"
    os.system(cmd)
    print(f"Decompressed PLY file: {output_ply}")
    
def ply_to_npy(ply_file, npy_file):
    points = o3d.io.read_point_cloud(ply_file)
    array = np.asarray(points.points).astype(np.uint16)
    array2, _ = attribute_reform_np(array, [])
    np.save(npy_file, array2)
    print(f"Saved npy file: {npy_file}")

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

def attribute_reform_np(anchor, attribute_list):
    """
    anchor : [N, 3], numpy
    attribute_list : [[N, M], ...], numpy
    
    Sort the anchor by x, y, z from small to large, apply the same permutation to the attribute_list.
    """
    if anchor.shape[1] != 3:
        raise ValueError("Input anchor must have shape (N, 3)")
    
    # Sort the anchor by x from small to large, if x is the same, sort by y, if y is the same, sort by z
    sorted_indices = np.argsort(anchor[:, 0] + anchor[:, 1] * 1e-6 + anchor[:, 2] * 1e-12)
    anchor = anchor[sorted_indices]
    attribute_list = [attribute[sorted_indices] for attribute in attribute_list]
    
    return anchor, attribute_list

def write_anchor_to_ply(anchor, ply_file_path):
    """
    anchor : [N, 3] int, numpy
    ply_file_path : str, path to the output ply file
    """
    if not isinstance(anchor, np.ndarray):
        raise ValueError("Input anchor must be a numpy array")
    if anchor.shape[1] != 3:
        raise ValueError("Input anchor must have shape (N, 3)")
    
    num_points = anchor.shape[0]
    with open(ply_file_path, "w") as f:
        # Write the PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property int x\n")
        f.write("property int y\n")
        f.write("property int z\n")
        f.write("end_header\n")
        
        for point in anchor:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

def estimate_anchor_bits_using_draco(_quantized_v, temp_folder_path='./temp/', qp=16):
    anchor_np = _quantized_v.cpu().numpy().astype(np.int32)
    write_anchor_to_ply(anchor_np, os.path.join(temp_folder_path, "_temp_quantized_v.ply"))
    
    draco_encode_cmd = f"draco_encoder -i {os.path.join(temp_folder_path, '_temp_quantized_v.ply')} -o {os.path.join(temp_folder_path, '_temp_quantized_v.drc')} -point_cloud -qp {qp} -point_cloud"
    os.system(draco_encode_cmd)
    
    compressed_size = os.path.getsize(os.path.join(temp_folder_path, "_temp_quantized_v.drc")) * 8
    os.remove(os.path.join(temp_folder_path, "_temp_quantized_v.ply"))
    os.remove(os.path.join(temp_folder_path, "_temp_quantized_v.drc"))
    return compressed_size

def draco_encode(npy_path, drc_path, qp=16):
    ply_path = npy_path[:-4] + ".ply"
    npy_to_ply(npy_path, ply_path)
    compress_ply_with_draco(ply_path, drc_path, qp)

def draco_decode(drc_path, npy_path):
    ply_path = drc_path[:-4] + ".ply"
    decompress_ply_with_draco(drc_path, ply_path)
    ply_to_npy(ply_path, npy_path)
    
    









































def encode_anchor_using_draco(_quantized_v, _quantized_v_ply_file_path, qp=16):
    anchor_np = _quantized_v.cpu().numpy().astype(np.int32)
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
    sorted_indices = np.argsort(array[:, 0] + array[:, 1] * 1e-6 + array[:, 2] * 1e-12)
    _decoded_quantized_v = array[sorted_indices]
    return _decoded_quantized_v
    
def encode_decoder_simulate(_quantized_v, temp_folder_path='./temp/', qp=16):
    # input: reformed tensor _quantized_v, >0, float, but all uint
    anchor_np = _quantized_v.cpu().numpy().astype(np.int32)
    write_anchor_to_ply(anchor_np, os.path.join(temp_folder_path, "_temp_quantized_v.ply"))
    
    draco_encode_cmd = f"draco_encoder -i {os.path.join(temp_folder_path, '_temp_quantized_v.ply')} -o {os.path.join(temp_folder_path, '_temp_quantized_v.drc')} -point_cloud -qp {qp} -point_cloud"
    os.system(draco_encode_cmd)
    
    draco_decode_cmd = f"draco_decoder -i {os.path.join(temp_folder_path, '_temp_quantized_v.drc')} -o {os.path.join(temp_folder_path, '_temp_quantized_v_decoded.ply')} -point_cloud"
    os.system(draco_decode_cmd)
    
    points = o3d.io.read_point_cloud(os.path.join(temp_folder_path, "_temp_quantized_v_decoded.ply"))
    decoded_quantized_v = np.asarray(points.points, dtype=np.int32)
    sorted_indices = np.argsort(decoded_quantized_v[:, 0] + decoded_quantized_v[:, 1] * 1e-6 + decoded_quantized_v[:, 2] * 1e-12)
    _decoded_quantized_v2= decoded_quantized_v[sorted_indices]
    
    # chech if the decoded quantized_v is the same as the original quantized_v
    print(anchor_np - _decoded_quantized_v2)
    
    
