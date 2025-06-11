#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import pandas as pd
import json
import argparse

class RetrievalEvaluator:
    """
    检索评估器
    
    用于评估DNMSR和EVA-CLIP模型在暗网商品检索任务上的性能
    """
    def __init__(
        self,
        embeddings_dir: str = "./embeddings",
        output_dir: str = "./results"
    ):
        """
        初始化检索评估器
        
        Args:
            embeddings_dir: 嵌入文件目录
            output_dir: 结果输出目录
        """
        self.embeddings_dir = embeddings_dir
        self.output_dir = output_dir
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 嵌入数据
        self.query_embeddings_dnmsr = None
        self.query_embeddings_clip = None
        self.query_product_ids = None
        
        self.gallery_embeddings_dnmsr = None
        self.gallery_product_ids_dnmsr = None
        self.gallery_modalities_dnmsr = None
        
        self.gallery_embeddings_clip = None
        self.gallery_product_ids_clip = None
        
        # 评估结果
        self.results = {}
    
    def load_embeddings(self) -> bool:
        """
        加载嵌入数据
        
        Returns:
            bool: 加载是否成功
        """
        try:
            # 加载查询嵌入
            query_file = os.path.join(self.embeddings_dir, "query_embeddings.npz")
            print(f"加载查询嵌入: {query_file}")
            
            if not os.path.exists(query_file):
                print(f"错误: 查询嵌入文件不存在: {query_file}")
                return False
            
            query_data = np.load(query_file, allow_pickle=True)
            self.query_embeddings_dnmsr = query_data["dnmsr_embeddings"]
            self.query_embeddings_clip = query_data["clip_embeddings"]
            self.query_product_ids = query_data["product_ids"]
            
            print(f"加载了 {len(self.query_product_ids)} 个查询嵌入")
            
            # 加载DNMSR候选文档嵌入
            dnmsr_gallery_file = os.path.join(self.embeddings_dir, "dnmsr_gallery_embeddings.npz")
            print(f"加载DNMSR候选文档嵌入: {dnmsr_gallery_file}")
            
            if not os.path.exists(dnmsr_gallery_file):
                print(f"错误: DNMSR候选文档嵌入文件不存在: {dnmsr_gallery_file}")
                return False
            
            dnmsr_gallery_data = np.load(dnmsr_gallery_file, allow_pickle=True)
            self.gallery_embeddings_dnmsr = dnmsr_gallery_data["embeddings"]
            self.gallery_product_ids_dnmsr = dnmsr_gallery_data["product_ids"]
            self.gallery_modalities_dnmsr = dnmsr_gallery_data["modalities"]
            
            print(f"加载了 {len(self.gallery_product_ids_dnmsr)} 个DNMSR候选文档嵌入")
            
            # 加载EVA-CLIP候选文档嵌入
            clip_gallery_file = os.path.join(self.embeddings_dir, "clip_gallery_embeddings.npz")
            print(f"加载EVA-CLIP候选文档嵌入: {clip_gallery_file}")
            
            if os.path.exists(clip_gallery_file):
                clip_gallery_data = np.load(clip_gallery_file, allow_pickle=True)
                self.gallery_embeddings_clip = clip_gallery_data["embeddings"]
                self.gallery_product_ids_clip = clip_gallery_data["product_ids"]
                
                print(f"加载了 {len(self.gallery_product_ids_clip)} 个EVA-CLIP候选文档嵌入")
            else:
                print(f"警告: EVA-CLIP候选文档嵌入文件不存在: {clip_gallery_file}")
                print(f"将跳过EVA-CLIP评估")
            
            return True
            
        except Exception as e:
            print(f"加载嵌入数据时出错: {e}")
            return False
    
    def normalize_embeddings(self) -> None:
        """标准化嵌入向量"""
        print(f"标准化嵌入向量...")
        
        # 标准化查询嵌入
        if isinstance(self.query_embeddings_dnmsr, np.ndarray) and len(self.query_embeddings_dnmsr) > 0:
            norms = np.linalg.norm(self.query_embeddings_dnmsr, axis=1, keepdims=True)
            self.query_embeddings_dnmsr = self.query_embeddings_dnmsr / norms
            
        if isinstance(self.query_embeddings_clip, np.ndarray) and len(self.query_embeddings_clip) > 0:
            norms = np.linalg.norm(self.query_embeddings_clip, axis=1, keepdims=True)
            self.query_embeddings_clip = self.query_embeddings_clip / norms
        
        # 标准化DNMSR候选文档嵌入
        if isinstance(self.gallery_embeddings_dnmsr, np.ndarray) and len(self.gallery_embeddings_dnmsr) > 0:
            norms = np.linalg.norm(self.gallery_embeddings_dnmsr, axis=1, keepdims=True)
            self.gallery_embeddings_dnmsr = self.gallery_embeddings_dnmsr / norms
        
        # 标准化EVA-CLIP候选文档嵌入
        if isinstance(self.gallery_embeddings_clip, np.ndarray) and len(self.gallery_embeddings_clip) > 0:
            norms = np.linalg.norm(self.gallery_embeddings_clip, axis=1, keepdims=True)
            self.gallery_embeddings_clip = self.gallery_embeddings_clip / norms
    
    def compute_similarity_matrix(self, query_embeddings, gallery_embeddings):
        """计算查询和候选文档之间的相似度矩阵"""
        print(f"查询嵌入形状: {query_embeddings.shape}")
        print(f"候选文档嵌入形状: {gallery_embeddings.shape}")
        
        # 对于EVA-CLIP嵌入的特殊处理
        if len(gallery_embeddings.shape) > 3:  # 如果是形如(273, 1, 257, 1024)的EVA-CLIP嵌入
            print("检测到EVA-CLIP格式的嵌入，执行特殊处理...")
            # 先取第一个维度的嵌入，通常为图像嵌入
            gallery_embeddings = gallery_embeddings[:, 0, 0, :]
            print(f"EVA-CLIP嵌入处理后形状: {gallery_embeddings.shape}")
        
        # 处理一般情况的嵌入维度
        if len(query_embeddings.shape) > 2:
            query_embeddings = query_embeddings.reshape(query_embeddings.shape[0], -1)
            print(f"调整后的查询嵌入形状: {query_embeddings.shape}")
            
        if len(gallery_embeddings.shape) > 2:
            # 对于一般的三维嵌入，保留最后一个维度
            if gallery_embeddings.shape[-1] == query_embeddings.shape[-1]:
                gallery_embeddings = gallery_embeddings.reshape(gallery_embeddings.shape[0], -1, gallery_embeddings.shape[-1])
                # 取第一个token的嵌入
                gallery_embeddings = gallery_embeddings[:, 0, :]
            else:
                # 如果最后维度不匹配，则尝试展平
                gallery_embeddings = gallery_embeddings.reshape(gallery_embeddings.shape[0], -1)
            print(f"调整后的候选文档嵌入形状: {gallery_embeddings.shape}")
        
        # 确保维度匹配
        if query_embeddings.shape[-1] != gallery_embeddings.shape[-1]:
            min_dim = min(query_embeddings.shape[-1], gallery_embeddings.shape[-1])
            query_embeddings = query_embeddings[..., :min_dim]
            gallery_embeddings = gallery_embeddings[..., :min_dim]
            print(f"截断后 - 查询形状: {query_embeddings.shape}, 候选形状: {gallery_embeddings.shape}")
        
        # 计算余弦相似度
        similarity_matrix = np.matmul(query_embeddings, gallery_embeddings.T)
        return similarity_matrix
    
    def evaluate_retrieval(
        self,
        query_embeddings: np.ndarray,
        query_product_ids: np.ndarray,
        gallery_embeddings: np.ndarray,
        gallery_product_ids: np.ndarray,
        gallery_modalities: np.ndarray = None,
        k_values: List[int] = [1, 5, 10, 20, 50],
        name: str = ""
    ) -> Dict[str, Any]:
        """
        评估检索性能
        
        Args:
            query_embeddings: 查询嵌入
            query_product_ids: 查询产品ID
            gallery_embeddings: 候选文档嵌入
            gallery_product_ids: 候选文档产品ID
            gallery_modalities: 候选文档模态
            k_values: 评估的k值列表
            name: 模型名称
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        print(f"评估{name}检索性能...")
        
        # 打印形状和类型信息以便调试
        print(f"查询嵌入形状: {query_embeddings.shape}, 类型: {query_embeddings.dtype}")
        print(f"候选文档嵌入形状: {gallery_embeddings.shape}, 类型: {gallery_embeddings.dtype}")
        
        # 处理形状不兼容的问题
        if len(query_embeddings.shape) > 2:
            query_embeddings = query_embeddings.squeeze()
        if len(gallery_embeddings.shape) > 2:
            gallery_embeddings = gallery_embeddings.squeeze()
        
        # 确保两个嵌入向量的最后一个维度匹配
        if query_embeddings.shape[-1] != gallery_embeddings.shape[-1]:
            print(f"警告: 嵌入维度不匹配! 查询: {query_embeddings.shape[-1]}, 候选: {gallery_embeddings.shape[-1]}")
            # 如果不匹配，我们可以选择舍弃一些维度或者填充
            min_dim = min(query_embeddings.shape[-1], gallery_embeddings.shape[-1])
            query_embeddings = query_embeddings[..., :min_dim]
            gallery_embeddings = gallery_embeddings[..., :min_dim]
            print(f"调整后 - 查询形状: {query_embeddings.shape}, 候选形状: {gallery_embeddings.shape}")
        
        # 计算相似度矩阵
        similarity_matrix = self.compute_similarity_matrix(query_embeddings, gallery_embeddings)
        
        # 初始化评估指标
        metrics = {
            f"recall@{k}": 0.0 for k in k_values
        }
        metrics.update({
            f"precision@{k}": 0.0 for k in k_values
        })
        # 添加Success@k指标
        metrics.update({
            f"success@{k}": 0.0 for k in k_values
        })
        metrics["mAP"] = 0.0
        metrics["num_queries"] = len(query_product_ids)
        
        # 记录每个查询的结果
        per_query_metrics = []
        
        # 对每个查询计算评估指标
        for i, query_id in enumerate(tqdm(query_product_ids, desc="评估查询")):
            # 获取当前查询的相似度
            similarities = similarity_matrix[i]
            
            # 获取相关候选文档的索引
            relevant_indices = np.where(gallery_product_ids == query_id)[0]
            
            if len(relevant_indices) == 0:
                print(f"警告: 查询ID {query_id} 没有相关候选文档")
                continue
            
            # 获取每个候选文档的相关性标签
            relevance = np.zeros(len(gallery_product_ids))
            relevance[relevant_indices] = 1
            
            # 按相似度降序排序
            sorted_indices = np.argsort(-similarities)
            
            # 计算每个k值的召回率和精确率
            query_metrics = {
                "query_id": query_id,
                "num_relevant": len(relevant_indices)
            }
            
            # 计算各个k值的指标
            for k in k_values:
                if k > len(sorted_indices):
                    continue
                
                # 取前k个结果
                top_k_indices = sorted_indices[:k]
                
                # 计算召回率 = 前k个中相关的数量 / 总相关数量
                recall_at_k = np.sum(relevance[top_k_indices]) / len(relevant_indices)
                metrics[f"recall@{k}"] += recall_at_k
                query_metrics[f"recall@{k}"] = recall_at_k
                
                # 计算精确率 = 前k个中相关的数量 / k
                precision_at_k = np.sum(relevance[top_k_indices]) / k
                metrics[f"precision@{k}"] += precision_at_k
                query_metrics[f"precision@{k}"] = precision_at_k
                
                # 计算Success@k = 前k个中是否至少有一个相关结果 (0或1)
                success_at_k = 1.0 if np.sum(relevance[top_k_indices]) > 0 else 0.0
                metrics[f"success@{k}"] += success_at_k
                query_metrics[f"success@{k}"] = success_at_k
            
            # 计算平均精度 (AP)
            # 获取相关候选的排名
            relevant_ranks = np.where(relevance[sorted_indices] == 1)[0] + 1
            if len(relevant_ranks) > 0:
                ap = np.sum(np.arange(1, len(relevant_ranks) + 1) / relevant_ranks) / len(relevant_indices)
                metrics["mAP"] += ap
                query_metrics["AP"] = ap
            
            # 如果有模态信息，记录每个模态的检索情况
            if gallery_modalities is not None:
                top_modalities = gallery_modalities[sorted_indices[:10]]
                modality_counts = {}
                for modality in top_modalities:
                    if modality not in modality_counts:
                        modality_counts[modality] = 0
                    modality_counts[modality] += 1
                query_metrics["top10_modalities"] = modality_counts
            
            per_query_metrics.append(query_metrics)
        
        # 计算平均指标
        num_valid_queries = len(per_query_metrics)
        if num_valid_queries > 0:
            for k in k_values:
                metrics[f"recall@{k}"] /= num_valid_queries
                metrics[f"precision@{k}"] /= num_valid_queries
                metrics[f"success@{k}"] /= num_valid_queries
            metrics["mAP"] /= num_valid_queries
        
        # 记录模态分布
        if gallery_modalities is not None:
            modality_distribution = {}
            for modality in gallery_modalities:
                if modality not in modality_distribution:
                    modality_distribution[modality] = 0
                modality_distribution[modality] += 1
            
            metrics["modality_distribution"] = {
                modality: count / len(gallery_modalities) for modality, count in modality_distribution.items()
            }
        
        # 记录详细的每个查询的结果
        metrics["per_query"] = per_query_metrics
        
        return metrics
    
    def evaluate_all(self) -> None:
        """评估所有检索任务"""
        # 标准化嵌入
        self.normalize_embeddings()
        
        # 初始化结果字典
        self.results = {}
        
        # 评估DNMSR模型
        if (self.query_embeddings_dnmsr is not None and 
            self.gallery_embeddings_dnmsr is not None):
            print(f"评估DNMSR模型的检索性能...")
            
            dnmsr_results = self.evaluate_retrieval(
                query_embeddings=self.query_embeddings_dnmsr,
                query_product_ids=self.query_product_ids,
                gallery_embeddings=self.gallery_embeddings_dnmsr,
                gallery_product_ids=self.gallery_product_ids_dnmsr,
                gallery_modalities=self.gallery_modalities_dnmsr,
                name="DNMSR"
            )
            
            self.results["dnmsr"] = dnmsr_results
            
            print(f"DNMSR检索评估完成")
            print(f"  mAP: {dnmsr_results['mAP']:.4f}")
            for k in [1, 5, 10]:
                print(f"  Recall@{k}: {dnmsr_results[f'recall@{k}']:.4f}")
                print(f"  Precision@{k}: {dnmsr_results[f'precision@{k}']:.4f}")
                print(f"  Success@{k}: {dnmsr_results[f'success@{k}']:.4f}")
            
            if "modality_distribution" in dnmsr_results:
                print(f"  模态分布:")
                for modality, ratio in dnmsr_results["modality_distribution"].items():
                    print(f"    {modality}: {ratio:.2%}")
        
        # 评估EVA-CLIP模型（仅图像检索）
        if (self.query_embeddings_clip is not None and 
            self.gallery_embeddings_clip is not None):
            print(f"评估EVA-CLIP模型的图像检索性能...")
            
            clip_results = self.evaluate_retrieval(
                query_embeddings=self.query_embeddings_clip,
                query_product_ids=self.query_product_ids,
                gallery_embeddings=self.gallery_embeddings_clip,
                gallery_product_ids=self.gallery_product_ids_clip,
                name="EVA-CLIP"
            )
            
            self.results["clip"] = clip_results
            
            print(f"EVA-CLIP检索评估完成")
            print(f"  mAP: {clip_results['mAP']:.4f}")
            for k in [1, 5, 10]:
                print(f"  Recall@{k}: {clip_results[f'recall@{k}']:.4f}")
                print(f"  Precision@{k}: {clip_results[f'precision@{k}']:.4f}")
                print(f"  Success@{k}: {clip_results[f'success@{k}']:.4f}")
    
    def save_results(self) -> None:
        """保存评估结果"""
        if not self.results:
            print(f"没有评估结果可保存")
            return
        
        print(f"保存评估结果到: {self.output_dir}")
        
        # 保存详细结果为JSON
        results_file = os.path.join(self.output_dir, "retrieval_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"详细结果已保存到: {results_file}")
        
        # 生成汇总报告
        summary = {
            "model": [],
            "mAP": [],
        }
        
        # 添加recall、precision和success指标
        k_values = [1, 5, 10, 20, 50]
        for k in k_values:
            summary[f"recall@{k}"] = []
            summary[f"precision@{k}"] = []
            summary[f"success@{k}"] = []
        
        # 添加各模型的结果
        for model, results in self.results.items():
            summary["model"].append(model)
            summary["mAP"].append(results["mAP"])
            
            for k in k_values:
                if f"recall@{k}" in results:
                    summary[f"recall@{k}"].append(results[f"recall@{k}"])
                else:
                    summary[f"recall@{k}"].append(float('nan'))
                    
                if f"precision@{k}" in results:
                    summary[f"precision@{k}"].append(results[f"precision@{k}"])
                else:
                    summary[f"precision@{k}"].append(float('nan'))
                    
                if f"success@{k}" in results:
                    summary[f"success@{k}"].append(results[f"success@{k}"])
                else:
                    summary[f"success@{k}"].append(float('nan'))
        
        # 创建DataFrame并保存为CSV
        summary_df = pd.DataFrame(summary)
        summary_file = os.path.join(self.output_dir, "retrieval_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        
        print(f"汇总报告已保存到: {summary_file}")
        
        # 绘制性能对比图
        self._plot_performance_comparison()
    
    def _plot_performance_comparison(self) -> None:
        """绘制性能对比图"""
        if not self.results:
            return
        
        # 绘制召回率对比图
        plt.figure(figsize=(10, 6))
        
        k_values = [1, 5, 10, 20, 50]
        width = 0.35
        x = np.arange(len(k_values))
        
        for i, (model, results) in enumerate(self.results.items()):
            recalls = [results.get(f"recall@{k}", 0) for k in k_values]
            plt.bar(x + i*width, recalls, width, label=model.upper())
        
        plt.xlabel('K值')
        plt.ylabel('召回率')
        plt.title('不同模型在各K值下的召回率对比')
        plt.xticks(x + width/2, [f"@{k}" for k in k_values])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 保存图表
        plt.tight_layout()
        recall_plot_file = os.path.join(self.output_dir, "recall_comparison.png")
        plt.savefig(recall_plot_file, dpi=300)
        
        # 绘制精确率对比图
        plt.figure(figsize=(10, 6))
        
        for i, (model, results) in enumerate(self.results.items()):
            precisions = [results.get(f"precision@{k}", 0) for k in k_values]
            plt.bar(x + i*width, precisions, width, label=model.upper())
        
        plt.xlabel('K值')
        plt.ylabel('精确率')
        plt.title('不同模型在各K值下的精确率对比')
        plt.xticks(x + width/2, [f"@{k}" for k in k_values])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 保存图表
        plt.tight_layout()
        precision_plot_file = os.path.join(self.output_dir, "precision_comparison.png")
        plt.savefig(precision_plot_file, dpi=300)
        
        # 绘制Success@k对比图
        plt.figure(figsize=(10, 6))
        
        for i, (model, results) in enumerate(self.results.items()):
            successes = [results.get(f"success@{k}", 0) for k in k_values]
            plt.bar(x + i*width, successes, width, label=model.upper())
        
        plt.xlabel('K值')
        plt.ylabel('成功率')
        plt.title('不同模型在各K值下的成功率对比 (Success@k)')
        plt.xticks(x + width/2, [f"@{k}" for k in k_values])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 1.0)  # 成功率是0-1的值
        
        # 保存图表
        plt.tight_layout()
        success_plot_file = os.path.join(self.output_dir, "success_comparison.png")
        plt.savefig(success_plot_file, dpi=300)
        
        # 绘制mAP对比图
        plt.figure(figsize=(8, 6))
        
        models = list(self.results.keys())
        maps = [results["mAP"] for results in self.results.values()]
        
        plt.bar(models, maps, color=['#1f77b4', '#ff7f0e'])
        plt.xlabel('模型')
        plt.ylabel('mAP')
        plt.title('不同模型的mAP对比')
        plt.ylim(0, 1.0)
        
        # 在柱状图上添加数值标签
        for i, v in enumerate(maps):
            plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 保存图表
        plt.tight_layout()
        map_plot_file = os.path.join(self.output_dir, "map_comparison.png")
        plt.savefig(map_plot_file, dpi=300)
        
        print(f"性能对比图已保存到: {self.output_dir}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="评估DNMSR和EVA-CLIP模型的检索性能")
    parser.add_argument("--embeddings_dir", type=str, default="./embeddings", help="嵌入文件目录")
    parser.add_argument("--output_dir", type=str, default="./results", help="结果输出目录")
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = RetrievalEvaluator(
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir
    )
    
    # 加载嵌入
    if not evaluator.load_embeddings():
        print(f"加载嵌入数据失败，退出评估")
        return
    
    # 评估检索性能
    evaluator.evaluate_all()
    
    # 保存结果
    evaluator.save_results()
    
    print(f"检索评估完成")

if __name__ == "__main__":
    main() 