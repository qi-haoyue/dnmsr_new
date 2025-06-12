#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
更新评估结果 - 添加Success@k和MRR@k指标

此脚本读取已有的评估结果JSON文件，计算Success@k和MRR@k指标，并更新结果。
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union
import argparse

def compute_metrics(per_query_metrics, k_values=[1, 5, 10, 20, 50]):
    """
    计算每个查询的Success@k和MRR@k指标
    
    Args:
        per_query_metrics: 每个查询的详细指标
        k_values: 要计算的k值列表
        
    Returns:
        更新后的per_query_metrics和汇总指标
    """
    # 初始化汇总指标
    success_metrics = {f"success@{k}": 0.0 for k in k_values}
    mrr_metrics = {f"mrr@{k}": 0.0 for k in k_values}
    
    # 计算每个查询的Success@k和MRR@k
    for query_metrics in per_query_metrics:
        for k in k_values:
            recall_key = f"recall@{k}"
            success_key = f"success@{k}"
            mrr_key = f"mrr@{k}"
            
            # 如果已有对应k值的召回率
            if recall_key in query_metrics:
                # 计算Success@k：只要有一个相关结果，值为1，否则为0
                query_metrics[success_key] = 1.0 if query_metrics[recall_key] > 0 else 0.0
                success_metrics[success_key] += query_metrics[success_key]
                
                # 为该查询计算MRR@k
                mrr_at_k = 0.0
                # 遍历前k个结果找到第一个相关的排名
                for i in range(1, k+1):
                    # 如果是精确度@i > 0，则在第i个位置找到了相关文档
                    precision_key = f"precision@{i}"
                    if precision_key in query_metrics and query_metrics[precision_key] > 0:
                        # MRR = 1/rank，其中rank是第一个相关文档的排名
                        mrr_at_k = 1.0 / i
                        break
                
                query_metrics[mrr_key] = mrr_at_k
                mrr_metrics[mrr_key] += mrr_at_k
    
    # 计算平均值
    if per_query_metrics:
        num_queries = len(per_query_metrics)
        for k in k_values:
            success_metrics[f"success@{k}"] /= num_queries
            mrr_metrics[f"mrr@{k}"] /= num_queries
    
    return per_query_metrics, success_metrics, mrr_metrics

def update_results_json(json_file, output_file=None):
    """
    读取评估结果JSON文件，计算并添加Success@k和MRR@k指标
    
    Args:
        json_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径，如果为None则覆盖输入文件
        
    Returns:
        bool: 操作是否成功
    """
    try:
        # 读取JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"加载评估结果文件: {json_file}")
        
        # 遍历每个模型的结果
        for model_name, model_results in results.items():
            print(f"处理模型: {model_name}")
            
            # 获取每个查询的指标
            if "per_query" not in model_results:
                print(f"  警告: 模型 {model_name} 没有per_query数据，跳过")
                continue
            
            per_query_metrics = model_results["per_query"]
            
            # 计算Success@k和MRR@k指标
            updated_per_query, success_metrics, mrr_metrics = compute_metrics(per_query_metrics)
            
            # 更新每个查询的指标
            model_results["per_query"] = updated_per_query
            
            # 更新汇总指标
            for k, value in success_metrics.items():
                model_results[k] = value
                print(f"  添加 {k}: {value:.4f}")
            
            for k, value in mrr_metrics.items():
                model_results[k] = value
                print(f"  添加 {k}: {value:.4f}")
        
        # 保存更新后的结果
        if output_file is None:
            output_file = json_file
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {output_file}")
        return True
    
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return False

def update_csv_summary(json_file, csv_file, output_csv=None):
    """
    根据更新后的JSON文件，更新CSV汇总文件
    
    Args:
        json_file: 更新后的JSON文件路径
        csv_file: 原始CSV汇总文件路径
        output_csv: 输出的CSV文件路径，如果为None则覆盖输入文件
    
    Returns:
        bool: 操作是否成功
    """
    try:
        # 读取JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 创建新的汇总数据
        summary = {
            "model": [],
            "mAP": [],
        }
        
        # 添加各类指标
        k_values = [1, 5, 10, 20, 50]
        for k in k_values:
            summary[f"recall@{k}"] = []
            summary[f"precision@{k}"] = []
            summary[f"success@{k}"] = []
            summary[f"mrr@{k}"] = []
        
        # 添加各模型的结果
        for model, model_results in results.items():
            summary["model"].append(model)
            summary["mAP"].append(model_results.get("mAP", float('nan')))
            
            for k in k_values:
                recall_key = f"recall@{k}"
                precision_key = f"precision@{k}"
                success_key = f"success@{k}"
                mrr_key = f"mrr@{k}"
                
                summary[recall_key].append(model_results.get(recall_key, float('nan')))
                summary[precision_key].append(model_results.get(precision_key, float('nan')))
                summary[success_key].append(model_results.get(success_key, float('nan')))
                summary[mrr_key].append(model_results.get(mrr_key, float('nan')))
        
        # 创建DataFrame并保存为CSV
        summary_df = pd.DataFrame(summary)
        
        if output_csv is None:
            output_csv = csv_file
        
        summary_df.to_csv(output_csv, index=False)
        print(f"CSV汇总已更新并保存到: {output_csv}")
        return True
    
    except Exception as e:
        print(f"更新CSV汇总时出错: {e}")
        return False

def setup_chinese_font():
    """配置matplotlib中文字体支持"""
    try:
        import matplotlib as mpl
        import platform
        
        system = platform.system()
        
        if system == 'Windows':
            # Windows字体
            mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        elif system == 'Darwin':  # macOS
            # macOS字体
            mpl.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
        else:  # Linux等
            # Linux字体
            mpl.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'Arial Unicode MS']
            
        # 通用设置，解决负号显示问题
        mpl.rcParams['axes.unicode_minus'] = False
        
        # 设置DPI，确保图像清晰
        mpl.rcParams['figure.dpi'] = 100
        
        print("中文字体配置完成")
        return True
    except Exception as e:
        print(f"配置中文字体时出错: {e}")
        return False

def plot_metrics(json_file, output_dir):
    """
    根据JSON文件绘制各种指标的对比图
    
    Args:
        json_file: 包含评估结果的JSON文件路径
        output_dir: 输出图表的目录
        
    Returns:
        bool: 操作是否成功
    """
    try:
        # 读取JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 配置中文字体支持
        # setup_chinese_font()  # 由于系统原因无法生效，改用英文
        
        # 绘制各种指标的对比图
        k_values = [1, 5, 10, 20, 50]
        width = 0.35
        x = np.arange(len(k_values))
        
        # 绘制Success@k对比图
        plt.figure(figsize=(10, 6))
        
        for i, (model, model_results) in enumerate(results.items()):
            successes = [model_results.get(f"success@{k}", 0) for k in k_values]
            plt.bar(x + i*width, successes, width, label=model.upper())
        
        plt.xlabel('K Value')
        plt.ylabel('Success Rate')
        plt.title('Success Rate Comparison at Different K Values (Success@k)')
        plt.xticks(x + width/2, [f"@{k}" for k in k_values])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 1.0)  # 成功率是0-1的值
        
        # 保存图表
        plt.tight_layout()
        success_plot_file = os.path.join(output_dir, "success_comparison.png")
        plt.savefig(success_plot_file, dpi=300)
        print(f"Success rate comparison chart saved to: {success_plot_file}")
        
        # 绘制MRR@k对比图
        plt.figure(figsize=(10, 6))
        
        for i, (model, model_results) in enumerate(results.items()):
            mrrs = [model_results.get(f"mrr@{k}", 0) for k in k_values]
            plt.bar(x + i*width, mrrs, width, label=model.upper())
        
        plt.xlabel('K Value')
        plt.ylabel('Mean Reciprocal Rank')
        plt.title('Mean Reciprocal Rank Comparison at Different K Values (MRR@k)')
        plt.xticks(x + width/2, [f"@{k}" for k in k_values])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 1.0)  # MRR是0-1的值
        
        # 保存图表
        plt.tight_layout()
        mrr_plot_file = os.path.join(output_dir, "mrr_comparison.png")
        plt.savefig(mrr_plot_file, dpi=300)
        print(f"MRR comparison chart saved to: {mrr_plot_file}")
        
        return True
    
    except Exception as e:
        print(f"Error generating metrics charts: {e}")
        return False

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='更新评估结果，添加Success@k和MRR@k指标')
    parser.add_argument('--input_json', type=str, required=True, help='输入的评估结果JSON文件路径')
    parser.add_argument('--input_csv', type=str, help='输入的CSV汇总文件路径')
    parser.add_argument('--output_json', type=str, help='输出的JSON文件路径，默认覆盖输入文件')
    parser.add_argument('--output_csv', type=str, help='输出的CSV文件路径，默认覆盖输入文件')
    parser.add_argument('--output_dir', type=str, help='输出图表的目录，默认使用JSON文件所在目录')
    
    args = parser.parse_args()
    
    # 设置默认输出路径
    if args.output_json is None:
        args.output_json = args.input_json
    
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.input_json)
    
    # 更新JSON文件
    success = update_results_json(args.input_json, args.output_json)
    if not success:
        return 1
    
    # 如果提供了CSV文件，也更新它
    if args.input_csv:
        if args.output_csv is None:
            args.output_csv = args.input_csv
        
        success = update_csv_summary(args.output_json, args.input_csv, args.output_csv)
        if not success:
            print("警告: CSV更新失败，但JSON更新成功")
    
    # 绘制指标对比图
    success = plot_metrics(args.output_json, args.output_dir)
    if not success:
        print("警告: 绘图失败，但指标更新成功")
    
    print("处理完成!")
    return 0

if __name__ == "__main__":
    exit(main()) 