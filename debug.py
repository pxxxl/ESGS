import os
import argparse
import train 
import sys

def run_train():
    lmbda_list = [0.004]
    scene_list = ['amsterdam']
    dataset_path = '/home/ethan/Project/Python/HAC/data/bungeenerf'

    step_begin_quantization = 30
    step_begin_RD_training = 100
    step_begin_entropy_skipping = 200
    enable_es_in_final_estimation = 1
    es_ratio = 0.5

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
                "-m", f"outputs/{os.path.basename(dataset_path)}/{scene}/{lmbda}_{enable_es_in_final_estimation}_{es_ratio}",
                "--lmbda", str(lmbda),
                "--step_begin_quantization", str(step_begin_quantization),
                "--step_begin_RD_training", str(step_begin_RD_training),
                "--step_begin_entropy_skipping", str(step_begin_entropy_skipping),
                "--enable_es_in_final_estimation", str(enable_es_in_final_estimation),
                "--es_ratio", str(es_ratio),
                "--iterations", "300",
                "--test_iterations", "300",
                "--save_iterations", "300",
                "--start_stat", "50",
                "--update_from", "100",
                "--update_interval", "10",
                "--update_until", "150"
            ]
            
            # 调用 main 方法
            train.main()

if __name__ == "__main__":
    run_train()
