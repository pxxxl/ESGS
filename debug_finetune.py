lmbda_list = [0.004]
scene_list = ['amsterdam']
dataset_path = '/home/ethan/Project/Python/HAC/data/bungeenerf'
base_model_path = '/home/ethan/Project/ESGS/outputs/bungeenerf/amsterdam/0.004_ESGS'
finetune_model_path = '/home/ethan/Project/ESGS/outputs/bungeenerf/amsterdam/0.004_ESGS_finetune'


enable_debug = True
iterations = 100
step_begin_quantization = 30
step_begin_RD_training = 90


import os
import argparse
import train 
import finetune
import sys

def run_train():
    for lmbda in lmbda_list:
        for scene in scene_list:
            # 构造命令行参数列表
            sys.argv = [
                "finetune.py",
                "--base_model_path", "/home/ethan/Project/ESGS/outputs/bungeenerf/amsterdam/0.004_ESGS",
                "--base_source_path", f"{dataset_path}/{scene}",
                "--fine_tune_model_path", f"{finetune_model_path}"
            ]
            if enable_debug:
                sys.argv += [
                    "--iterations", f"{iterations}",
                    "--test_iterations", f"{iterations}",
                    "--save_iterations", f"{iterations}",
                    "--start_stat", "20",
                    "--update_from", "20",
                    "--update_interval", "10",
                    "--update_until", "80"
                ]
            
            # 调用 main 方法
            finetune.main()

if __name__ == "__main__":
    run_train()
