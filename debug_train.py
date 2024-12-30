lmbda_list = [0.004]
scene_list = ['amsterdam']
dataset_path = '/home/ethan/Project/Python/HAC/data/bungeenerf'

enable_debug = False
iterations = 300
step_begin_quantization = 30
step_begin_RD_training = 100
CUDA = 0


import os
import argparse
import train 
import sys

def run_train():
    for lmbda in lmbda_list:
        for scene in scene_list:
            # 构造命令行参数列表
            sys.argv = [
                "train.py",
                "-s", f"{dataset_path}/{scene}",
                "--eval",
                "--lod", "30",
                "--voxel_size", "0",
                "--update_init_factor", "128",
                "-m", f"outputs/{os.path.basename(dataset_path)}/{scene}/{lmbda}_ESGS",
                "--lmbda", str(lmbda),
                "--step_begin_quantization", str(step_begin_quantization),
                "--step_begin_RD_training", str(step_begin_RD_training)
            ]
            if enable_debug:
                sys.argv += [
                    "--iterations", f"{iterations}",
                    "--test_iterations", f"{iterations}",
                    "--save_iterations", f"{iterations}",
                    "--start_stat", "160",
                    "--update_from", "160",
                    "--update_interval", "10",
                    "--update_until", "150"
                ]
            else:
                sys_argv_str = f"CUDA_VISIBLE_DEVICE={CUDA} python "
                for arg in sys.argv:
                    sys_argv_str += arg
                    sys_argv_str += " "
            
            if enable_debug:
                train.main()
            else:
                os.system(sys_argv_str)

if __name__ == "__main__":
    run_train()
