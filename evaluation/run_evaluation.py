#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import sys
import time
from datetime import datetime

def run_command(cmd, desc=None):
    """
    运行shell命令并显示输出
    
    Args:
        cmd: 要运行的命令
        desc: 命令描述
    
    Returns:
        int: 命令的返回码
    """
    if desc:
        print(f"\n{'='*80}")
        print(f"  {desc}")
        print(f"{'='*80}")
    
    print(f"执行命令: {cmd}")
    process = subprocess.Popen(
        cmd, 
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # 实时输出命令执行结果
    for line in process.stdout:
        print(line.strip())
    
    process.wait()
    return process.returncode

def create_directory(dir_path):
    """创建目录（如果不存在）"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"创建目录: {dir_path}")
    return dir_path

def main():
    parser = argparse.ArgumentParser(description="运行DNMSR和EVA-CLIP模型的暗网商品检索评估")
    
    # 基本配置
    parser.add_argument("--dnmsr_model_path", type=str, default=None, 
                        help="DNMSR模型路径")
    parser.add_argument("--samples_file", type=str, default=None,
                        help="采样商品文件路径")
    parser.add_argument("--candidates_file", type=str, default=None,
                        help="候选文档文件路径")
    
    # 输出目录
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="评估结果输出目录")
    
    # 运行阶段控制
    parser.add_argument("--skip_embedding", action="store_true",
                        help="跳过嵌入构建阶段")
    parser.add_argument("--skip_evaluation", action="store_true",
                        help="跳过检索评估阶段")
    
    args = parser.parse_args()
    
    # 自动检测路径配置
    possible_paths = [
        # 远程服务器路径
        {
            "dnmsr_model_path": "/home/qhy/MML/dnmsr/Visualized_m3.pth",
            "samples_file": "/home/qhy/MML/dnmsr/dnmsr_new/data_preprocessing/samples/sampled_products.json",
            "candidates_file": "/home/qhy/MML/dnmsr/dnmsr_new/data_preprocessing/samples/candidate_documents.json"
        },
        # 本地开发路径
        {
            "dnmsr_model_path": "./dnmsr/Visualized_m3.pth",
            "samples_file": "./samples/sampled_products.json",
            "candidates_file": "./samples/candidate_documents.json"
        }
    ]
    
    # 使用命令行参数或自动检测路径
    config = {
        "dnmsr_model_path": args.dnmsr_model_path,
        "samples_file": args.samples_file,
        "candidates_file": args.candidates_file
    }
    
    # 如果命令行没有指定路径，尝试自动检测
    if not all(config.values()):
        for path_config in possible_paths:
            if (os.path.exists(path_config["dnmsr_model_path"]) and 
                os.path.exists(path_config["samples_file"]) and 
                os.path.exists(path_config["candidates_file"])):
                
                # 使用自动检测到的有效路径
                for key, value in path_config.items():
                    if not config[key]:
                        config[key] = value
                break
    
    # 验证必要的路径是否有效
    if not all(config.values()):
        missing = [key for key, value in config.items() if not value]
        print(f"错误: 缺少必要的路径配置: {', '.join(missing)}")
        print("请提供有效的路径或确保文件存在")
        return 1
    
    # 输出完整配置信息
    print("\n检索评估配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"  output_dir: {args.output_dir}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = create_directory(os.path.join(args.output_dir, f"run_{timestamp}"))
    embeddings_dir = create_directory(os.path.join(output_dir, "embeddings"))
    results_dir = create_directory(os.path.join(output_dir, "results"))
    
    # 记录开始时间
    start_time = time.time()
    
    # 步骤1: 构建嵌入
    if not args.skip_embedding:
        embedding_cmd = (
            f"python -m dnmsr.dnmsr_new.evaluation.embedding_builder "
            f"--dnmsr_model_path {config['dnmsr_model_path']} "
            f"--samples_file {config['samples_file']} "
            f"--candidates_file {config['candidates_file']} "
            f"--output_dir {embeddings_dir}"
        )
        
        if run_command(embedding_cmd, "步骤1: 构建查询嵌入和候选文档库") != 0:
            print("错误: 嵌入构建失败")
            return 1
    else:
        print("\n跳过嵌入构建阶段")
    
    # 步骤2: 运行检索评估
    if not args.skip_evaluation:
        eval_cmd = (
            f"python -m dnmsr.dnmsr_new.evaluation.retrieval_evaluator "
            f"--embeddings_dir {embeddings_dir} "
            f"--output_dir {results_dir}"
        )
        
        if run_command(eval_cmd, "步骤2: 评估检索性能") != 0:
            print("错误: 检索评估失败")
            return 1
    else:
        print("\n跳过检索评估阶段")
    
    # 计算总运行时间
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n{'='*80}")
    print(f"  评估完成!")
    print(f"{'='*80}")
    print(f"总运行时间: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
    print(f"评估结果保存在: {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 