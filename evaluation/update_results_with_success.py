#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
更新评估结果 - 添加Success@k指标

此脚本读取已有的评估结果JSON文件，计算Success@k指标，并更新结果。
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union
import argparse

def compute_success_at_k(per_query_metrics, k_values=[1, 5, 10, 20, 50]):
    """
    计算每个查询的Success@k指标
    
    Args:
        per_query_metrics: 每个查询的详细指标
        k_values: 要计算的k值列表
        
    Returns:
        更新后的per_query_metrics和汇总指标
    """
    # 初始化汇总指标
    success_metrics = {f"success@{k}": 0.0 for k in k_values}
    
    # 计算每个查询的Success@k
    for query_metrics in per_query_metrics:
        for k in k_values:
            recall_key = f"recall@{k}"
            success_key = f"success@{k}"
            
            # 如果有召回率指标，基于它计算成功率
            if recall_key in query_metrics:
                recall = query_metrics[recall_key]
                # 只要有结果被召回（召回率>0），就算成功
                success = 1.0 if recall > 0 else 0.0
                query_metrics[success_key] = success
                success_metrics[success_key] += success
    
    # 计算平均成功率
    num_queries = len(per_query_metrics)
    if num_queries > 0:
        for k in k_values:
            success_metrics[f"success@{k}"] /= num_queries
    
    return per_query_metrics, success_metrics

def update_results_json(input_file, output_file=None):
    """
    更新评估结果JSON文件，添加Success@k指标
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径，如果为None则覆盖输入文件
    """
    if output_file is None:
        output_file = input_file
    
    # 读取JSON文件
    print(f"读取评估结果: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    k_values = [1, 5, 10, 20, 50]
    
    # 更新每个模型的结果
    for model, model_results in results.items():
        print(f"更新 {model} 模型的Success@k指标...")
        
        # 如果存在per_query字段，更新每个查询的Success@k
        if "per_query" in model_results:
            per_query_metrics = model_results["per_query"]
            updated_per_query, success_metrics = compute_success_at_k(per_query_metrics, k_values)
            
            # 更新结果
            model_results["per_query"] = updated_per_query
            
            # 将汇总指标添加到模型结果中
            for k, v in success_metrics.items():
                model_results[k] = v
    
    # 保存更新后的结果
    print(f"保存更新后的结果: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def update_summary_csv(json_file, csv_file):
    """
    根据更新后的JSON结果，更新汇总CSV文件
    
    Args:
        json_file: 更新后的JSON文件路径
        csv_file: 原始CSV文件路径
    """
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 生成汇总数据
    summary = {
        "model": [],
        "mAP": [],
    }
    
    # 添加指标
    k_values = [1, 5, 10, 20, 50]
    for k in k_values:
        summary[f"recall@{k}"] = []
        summary[f"precision@{k}"] = []
        summary[f"success@{k}"] = []
    
    # 添加各模型的结果
    for model, model_results in results.items():
        summary["model"].append(model)
        summary["mAP"].append(model_results.get("mAP", float('nan')))
        
        for k in k_values:
            summary[f"recall@{k}"].append(model_results.get(f"recall@{k}", float('nan')))
            summary[f"precision@{k}"].append(model_results.get(f"precision@{k}", float('nan')))
            summary[f"success@{k}"].append(model_results.get(f"success@{k}", float('nan')))
    
    # 创建DataFrame并保存
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(csv_file, index=False)
    print(f"汇总报告已更新: {csv_file}")

def create_success_plot(json_file, output_dir):
    """
    创建Success@k对比图
    
    Args:
        json_file: JSON结果文件路径
        output_dir: 输出目录
    """
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 绘制Success@k对比图
    plt.figure(figsize=(10, 6))
    
    k_values = [1, 5, 10, 20, 50]
    width = 0.35
    x = np.arange(len(k_values))
    
    for i, (model, model_results) in enumerate(results.items()):
        successes = [model_results.get(f"success@{k}", 0) for k in k_values]
        plt.bar(x + i*width, successes, width, label=model.upper())
    
    plt.xlabel('K值')
    plt.ylabel('成功率')
    plt.title('不同模型在各K值下的成功率对比 (Success@k)')
    plt.xticks(x + width/2, [f"@{k}" for k in k_values])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1.0)
    
    # 保存图表
    plt.tight_layout()
    success_plot_file = os.path.join(output_dir, "success_comparison.png")
    plt.savefig(success_plot_file, dpi=300)
    print(f"成功率对比图已保存: {success_plot_file}")

def main():
    parser = argparse.ArgumentParser(description="更新评估结果，添加Success@k指标")
    parser.add_argument("--results_dir", type=str, default="./results", help="结果目录")
    args = parser.parse_args()
    
    # 构建文件路径
    json_file = os.path.join(args.results_dir, "retrieval_results.json")
    csv_file = os.path.join(args.results_dir, "retrieval_summary.csv")
    
    # 更新JSON结果
    update_results_json(json_file)
    
    # 更新汇总CSV
    update_summary_csv(json_file, csv_file)
    
    # 创建Success@k对比图
    create_success_plot(json_file, args.results_dir)
    
    print(f"Success@k指标更新完成")

if __name__ == "__main__":
    main() 