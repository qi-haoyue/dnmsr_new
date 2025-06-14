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
import datetime

class RetrievalEvaluator:
    """
    检索评估器
    
    用于评估DNMSR和EVA-CLIP模型在暗网商品检索任务上的性能
    """
    def __init__(
        self,
        embeddings_dir: str = "./embeddings",
        output_dir: str = "./results",
        mode: str = "multimodal"  # 检索模式: multimodal, text_to_image, image_to_text
    ):
        """
        初始化检索评估器
        
        Args:
            embeddings_dir: 嵌入文件目录
            output_dir: 结果输出目录
            mode: 检索模式，可选值为 'multimodal'(多模态混合检索), 'text_to_image'(文搜图), 'image_to_text'(图搜文)
        """
        self.embeddings_dir = embeddings_dir
        self.output_dir = output_dir
        self.mode = mode
        
        # 根据模式设置评估名称
        self.mode_name_map = {
            "multimodal": "多模态混合检索",
            "text_to_image": "文搜图检索",
            "image_to_text": "图搜文检索"
        }
        self.mode_name = self.mode_name_map.get(mode, "未知模式")
        
        print(f"初始化检索评估器，模式: {self.mode_name}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 嵌入数据
        self.query_embeddings_dnmsr = None
        self.query_embeddings_clip = None
        self.query_product_ids = None
        self.query_modalities = None  # 查询模态
        
        self.gallery_embeddings_dnmsr = None
        self.gallery_product_ids_dnmsr = None
        self.gallery_modalities_dnmsr = None
        
        self.gallery_embeddings_clip = None
        self.gallery_product_ids_clip = None
        self.gallery_modalities_clip = None
        
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
            
            # 尝试加载查询模态信息（如果存在）
            if "modalities" in query_data:
                self.query_modalities = query_data["modalities"]
                print(f"加载了查询模态信息: {len(set(self.query_modalities))}种模态")
            
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
                if "modalities" in clip_gallery_data:
                    self.gallery_modalities_clip = clip_gallery_data["modalities"]
                
                print(f"加载了 {len(self.gallery_product_ids_clip)} 个EVA-CLIP候选文档嵌入")
            else:
                print(f"警告: EVA-CLIP候选文档嵌入文件不存在: {clip_gallery_file}")
                print(f"将跳过EVA-CLIP评估")
            
            # 对于特定检索模式，过滤查询和候选文档
            self._filter_data_by_mode()
            
            return True
            
        except Exception as e:
            print(f"加载嵌入数据时出错: {e}")
            return False
    
    def _filter_data_by_mode(self):
        """根据检索模式过滤数据"""
        print(f"根据检索模式 {self.mode_name} 过滤数据...")
        
        # 由于数据加载时可能同时包含不同模态的查询和候选，
        # 在此处根据检索模式进行过滤
        if self.mode == "text_to_image":
            # 文搜图模式：查询为文本，候选为图像
            # 过滤查询为文本模态
            if self.query_modalities is not None:
                text_indices = np.array([i for i, mod in enumerate(self.query_modalities) if mod == 'text'])
                if len(text_indices) > 0:
                    print(f"过滤查询：保留 {len(text_indices)}/{len(self.query_modalities)} 个文本查询")
                    self.query_embeddings_dnmsr = self.query_embeddings_dnmsr[text_indices]
                    self.query_embeddings_clip = self.query_embeddings_clip[text_indices]
                    self.query_product_ids = self.query_product_ids[text_indices]
                    self.query_modalities = self.query_modalities[text_indices]
                else:
                    print("警告：未找到文本模态的查询")
            
            # 过滤候选为image模态（仅保留image模态，严格排除text和multimodal模态）
            if self.gallery_modalities_dnmsr is not None:
                image_indices = np.array([i for i, mod in enumerate(self.gallery_modalities_dnmsr) if mod == 'image'])
                if len(image_indices) > 0:
                    print(f"过滤候选：保留 {len(image_indices)}/{len(self.gallery_modalities_dnmsr)} 个图像候选")
                    self.gallery_embeddings_dnmsr = self.gallery_embeddings_dnmsr[image_indices]
                    self.gallery_product_ids_dnmsr = self.gallery_product_ids_dnmsr[image_indices]
                    self.gallery_modalities_dnmsr = self.gallery_modalities_dnmsr[image_indices]
                else:
                    print("警告：未找到图像模态的候选")
                    
            # 对CLIP候选也执行相同操作
            if self.gallery_modalities_clip is not None and self.gallery_embeddings_clip is not None:
                image_indices = np.array([i for i, mod in enumerate(self.gallery_modalities_clip) if mod == 'image'])
                if len(image_indices) > 0:
                    print(f"过滤CLIP候选：保留 {len(image_indices)}/{len(self.gallery_modalities_clip)} 个图像候选")
                    self.gallery_embeddings_clip = self.gallery_embeddings_clip[image_indices]
                    self.gallery_product_ids_clip = self.gallery_product_ids_clip[image_indices]
                    self.gallery_modalities_clip = self.gallery_modalities_clip[image_indices]
                    
        elif self.mode == "image_to_text":
            # 图搜文模式：查询为图像，候选为文本
            # 过滤查询为图像模态
            if self.query_modalities is not None:
                image_indices = np.array([i for i, mod in enumerate(self.query_modalities) if mod == 'image'])
                if len(image_indices) > 0:
                    print(f"过滤查询：保留 {len(image_indices)}/{len(self.query_modalities)} 个图像查询")
                    self.query_embeddings_dnmsr = self.query_embeddings_dnmsr[image_indices]
                    self.query_embeddings_clip = self.query_embeddings_clip[image_indices]
                    self.query_product_ids = self.query_product_ids[image_indices]
                    self.query_modalities = self.query_modalities[image_indices]
                else:
                    print("警告：未找到图像模态的查询")
            
            # 过滤候选为文本模态（仅保留text模态，严格排除image和multimodal模态）
            if self.gallery_modalities_dnmsr is not None:
                text_indices = np.array([i for i, mod in enumerate(self.gallery_modalities_dnmsr) if mod == 'text'])
                if len(text_indices) > 0:
                    print(f"过滤候选：保留 {len(text_indices)}/{len(self.gallery_modalities_dnmsr)} 个文本候选")
                    self.gallery_embeddings_dnmsr = self.gallery_embeddings_dnmsr[text_indices]
                    self.gallery_product_ids_dnmsr = self.gallery_product_ids_dnmsr[text_indices]
                    self.gallery_modalities_dnmsr = self.gallery_modalities_dnmsr[text_indices]
                else:
                    print("警告：未找到文本模态的候选")
                    
            # 对CLIP候选也执行相同操作
            if self.gallery_modalities_clip is not None and self.gallery_embeddings_clip is not None:
                text_indices = np.array([i for i, mod in enumerate(self.gallery_modalities_clip) if mod == 'text'])
                if len(text_indices) > 0:
                    print(f"过滤CLIP候选：保留 {len(text_indices)}/{len(self.gallery_modalities_clip)} 个文本候选")
                    self.gallery_embeddings_clip = self.gallery_embeddings_clip[text_indices]
                    self.gallery_product_ids_clip = self.gallery_product_ids_clip[text_indices]
                    self.gallery_modalities_clip = self.gallery_modalities_clip[text_indices]
        
        # 多模态混合检索模式下不需要过滤，保持原始数据
    
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
        """
        计算查询和候选文档之间的相似度矩阵，按照visual_bge模型的实现方式
        
        Args:
            query_embeddings: 查询嵌入向量
            gallery_embeddings: 候选文档嵌入向量
            
        Returns:
            np.ndarray: 相似度矩阵
        """
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
        
        # 对嵌入向量进行归一化
        # 在visual_bge模型中，归一化是在encode方法中完成的
        # 因此这里需要确保向量已经归一化
        query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        gallery_norms = np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
        
        # 避免除以零
        query_norms = np.maximum(query_norms, 1e-8)
        gallery_norms = np.maximum(gallery_norms, 1e-8)
        
        # 标准化向量
        query_embeddings = query_embeddings / query_norms
        gallery_embeddings = gallery_embeddings / gallery_norms
        
        # 计算余弦相似度 - 与visual_bge.compute_similarity保持一致
        similarity_matrix = np.matmul(query_embeddings, gallery_embeddings.T)
        
        # 输出相似度范围（调试用）
        min_sim = np.min(similarity_matrix)
        max_sim = np.max(similarity_matrix)
        print(f"相似度值范围: [{min_sim:.4f}, {max_sim:.4f}]")
        
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
        
        # 再次检查模态过滤，确保只使用符合当前模式的候选文档
        if gallery_modalities is not None and self.mode != "multimodal":
            print(f"检查候选文档模态是否符合{self.mode_name}要求...")
            
            # 根据模式确定需要的模态
            target_modality = None
            if self.mode == "text_to_image":
                target_modality = "image"
            elif self.mode == "image_to_text":
                target_modality = "text"
                
            if target_modality:
                # 找出符合目标模态的索引
                valid_indices = np.array([i for i, mod in enumerate(gallery_modalities) if mod == target_modality])
                
                if len(valid_indices) > 0:
                    print(f"过滤候选：保留 {len(valid_indices)}/{len(gallery_modalities)} 个{target_modality}模态候选")
                    gallery_embeddings = gallery_embeddings[valid_indices]
                    gallery_product_ids = gallery_product_ids[valid_indices]
                    gallery_modalities = gallery_modalities[valid_indices]
                    print(f"过滤后 - 候选形状: {gallery_embeddings.shape}")
                else:
                    print(f"警告：没有找到{target_modality}模态的候选文档，无法进行{self.mode_name}评估")
                    # 返回空结果
                    return {
                        "num_queries": 0,
                        "error": f"没有找到{target_modality}模态的候选文档"
                    }
        
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
        # 添加MRR@k指标
        metrics.update({
            f"mrr@{k}": 0.0 for k in k_values
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
                
                # 计算MRR@k = 第一个相关结果的倒数排名，如果前k个没有相关结果则为0
                mrr_at_k = 0.0
                relevant_positions = np.where(relevance[top_k_indices] == 1)[0]
                if len(relevant_positions) > 0:
                    # 找到第一个相关结果的位置 (0-based) 加1转为排名 (1-based)
                    first_relevant_rank = relevant_positions[0] + 1
                    mrr_at_k = 1.0 / first_relevant_rank
                
                metrics[f"mrr@{k}"] += mrr_at_k
                query_metrics[f"mrr@{k}"] = mrr_at_k
            
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
                metrics[f"mrr@{k}"] /= num_valid_queries
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
        """评估所有模型的检索性能"""
        print(f"开始评估所有模型在{self.mode_name}上的检索性能...")
        
        # 首先标准化嵌入向量
        self.normalize_embeddings()
        
        # 评估DNMSR模型
        if self.query_embeddings_dnmsr is not None and self.gallery_embeddings_dnmsr is not None:
            print(f"评估DNMSR模型...")
            dnmsr_results = self.evaluate_retrieval(
                query_embeddings=self.query_embeddings_dnmsr,
                query_product_ids=self.query_product_ids,
                gallery_embeddings=self.gallery_embeddings_dnmsr,
                gallery_product_ids=self.gallery_product_ids_dnmsr,
                gallery_modalities=self.gallery_modalities_dnmsr,
                name="DNMSR"
            )
            self.results["dnmsr"] = dnmsr_results
            
        # 评估EVA-CLIP模型，如果有嵌入
        if (self.query_embeddings_clip is not None and 
            self.gallery_embeddings_clip is not None):
            print(f"评估EVA-CLIP模型...")
            clip_results = self.evaluate_retrieval(
                query_embeddings=self.query_embeddings_clip,
                query_product_ids=self.query_product_ids,
                gallery_embeddings=self.gallery_embeddings_clip,
                gallery_product_ids=self.gallery_product_ids_clip,
                gallery_modalities=self.gallery_modalities_clip,
                name="EVA-CLIP"
            )
            self.results["clip"] = clip_results
        
        # 保存评估结果到JSON文件
        self.save_results()
        
        # 生成比较图表
        self._plot_performance_comparison()
        
        print(f"{self.mode_name}评估完成，结果已保存到: {self.output_dir}")

    def generate_retrieval_examples(self, num_examples=10, top_k=10):
        """
        生成检索示例分析，展示指定数量的查询及其检索结果
        
        Args:
            num_examples: 要生成的示例数量
            top_k: 每个查询展示的检索结果数量
        """
        if not self.results:
            print(f"没有评估结果可分析")
            return
            
        print(f"生成检索示例分析 ({num_examples}个查询, 每个展示前{top_k}个结果)...")
        
        # 准备输出文件
        examples_file = os.path.join(self.output_dir, "retrieval_examples.txt")
        
        # 准备相似度矩阵 - 对每个模型，需要重新计算查询和候选文档之间的相似度
        similarity_matrices = {}
        
        # 计算DNMSR相似度矩阵
        if "dnmsr" in self.results and self.query_embeddings_dnmsr is not None and self.gallery_embeddings_dnmsr is not None:
            similarity_matrices["dnmsr"] = self.compute_similarity_matrix(
                self.query_embeddings_dnmsr, 
                self.gallery_embeddings_dnmsr
            )
            
        # 计算CLIP相似度矩阵
        if "clip" in self.results and self.query_embeddings_clip is not None and self.gallery_embeddings_clip is not None:
            similarity_matrices["clip"] = self.compute_similarity_matrix(
                self.query_embeddings_clip, 
                self.gallery_embeddings_clip
            )
        
        # 确保至少有一个模型的相似度矩阵
        if not similarity_matrices:
            print(f"无法生成检索示例分析，缺少相似度数据")
            return
            
        # 抽取随机查询ID作为示例
        query_indices = list(range(len(self.query_product_ids)))
        if len(query_indices) > num_examples:
            # 随机抽样，但固定随机种子以确保可重现性
            np.random.seed(42)
            selected_indices = np.random.choice(query_indices, num_examples, replace=False)
        else:
            selected_indices = query_indices
            
        # 为每个选中的查询生成分析
        with open(examples_file, 'w', encoding='utf-8') as f:
            f.write(f"# Retrieval Example Analysis\n\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total queries: {len(self.query_product_ids)}\n")
            f.write(f"Showing {len(selected_indices)} random examples with top {top_k} results each\n\n")
            
            for idx, query_idx in enumerate(selected_indices):
                query_id = self.query_product_ids[query_idx]
                
                f.write(f"## Example {idx + 1}: Query ID {query_id}\n\n")
                
                # 对每个模型展示检索结果
                for model_name, sim_matrix in similarity_matrices.items():
                    f.write(f"### {model_name.upper()} Model Results\n\n")
                    
                    # 获取该查询的相似度向量
                    similarities = sim_matrix[query_idx]
                    
                    # 对相似度进行排序，获取前k个结果
                    sorted_indices = np.argsort(-similarities)[:top_k]
                    
                    # 获取对应的候选文档ID
                    gallery_ids = None
                    gallery_modalities = None
                    
                    if model_name == "dnmsr":
                        gallery_ids = self.gallery_product_ids_dnmsr
                        gallery_modalities = self.gallery_modalities_dnmsr
                    elif model_name == "clip":
                        gallery_ids = self.gallery_product_ids_clip
                        gallery_modalities = self.gallery_modalities_clip
                        
                    if gallery_ids is None:
                        f.write(f"No gallery information available for {model_name}\n\n")
                        continue
                        
                    # 确定是否有相关文档
                    relevant_indices = np.where(gallery_ids == query_id)[0]
                    has_relevant = len(relevant_indices) > 0
                    
                    if has_relevant:
                        f.write(f"This query has {len(relevant_indices)} relevant documents in the gallery\n\n")
                    else:
                        f.write(f"This query has no relevant documents in the gallery\n\n")
                    
                    # 创建表格标题
                    if gallery_modalities is not None:
                        f.write(f"| Rank | Document ID | Similarity | Modality | Relevant |\n")
                        f.write(f"|------|------------|------------|----------|----------|\n")
                    else:
                        f.write(f"| Rank | Document ID | Similarity | Relevant |\n")
                        f.write(f"|------|------------|------------|----------|\n")
                    
                    # 展示前k个结果
                    for rank, idx in enumerate(sorted_indices):
                        doc_id = gallery_ids[idx]
                        similarity = similarities[idx]
                        is_relevant = doc_id == query_id
                        
                        if gallery_modalities is not None:
                            modality = gallery_modalities[idx]
                            f.write(f"| {rank+1} | {doc_id} | {similarity:.4f} | {modality} | {'✓' if is_relevant else '✗'} |\n")
                        else:
                            f.write(f"| {rank+1} | {doc_id} | {similarity:.4f} | {'✓' if is_relevant else '✗'} |\n")
                    
                    f.write("\n")
                
                # 添加查询之间的分隔线
                if idx < len(selected_indices) - 1:
                    f.write(f"---\n\n")
        
        print(f"检索示例分析已保存到: {examples_file}")
        
    def save_results(self) -> None:
        """保存评估结果到JSON文件"""
        # 将NumPy数组转换为Python列表以便JSON序列化
        for model, result in self.results.items():
            for k, v in result.items():
                if isinstance(v, np.ndarray):
                    self.results[model][k] = v.tolist()
                elif isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, np.ndarray):
                            self.results[model][k][sub_k] = sub_v.tolist()
        
        # 添加元数据
        metadata = {
            "date": datetime.datetime.now().isoformat(),
            "mode": self.mode,
            "mode_name": self.mode_name,
            "embeddings_dir": self.embeddings_dir
        }
        
        # 构建完整结果
        full_results = {
            "metadata": metadata,
            "results": self.results
        }
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存到JSON文件
        output_path = os.path.join(self.output_dir, "retrieval_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, ensure_ascii=False, indent=2)
            
        print(f"评估结果已保存到: {output_path}")
        
        # 生成汇总CSV
        summary_data = []
        for model_name, model_results in self.results.items():
            # 收集MRR和成功率
            mrr = model_results.get("mrr", 0)
            success_rate = model_results.get("success_rate", 0)
            
            # 收集不同k值的MAP和Recall
            for k in [1, 5, 10, 20, 50]:
                map_value = model_results.get(f"map@{k}", 0)
                recall = model_results.get(f"recall@{k}", 0)
                precision = model_results.get(f"precision@{k}", 0)
                
                summary_data.append({
                    "模型": model_name,
                    "指标": f"MAP@{k}",
                    "值": map_value
                })
                
                summary_data.append({
                    "模型": model_name,
                    "指标": f"召回率@{k}",
                    "值": recall
                })
                
                summary_data.append({
                    "模型": model_name,
                    "指标": f"精确率@{k}",
                    "值": precision
                })
                
            # 添加MRR和成功率
            summary_data.append({
                "模型": model_name,
                "指标": "MRR",
                "值": mrr
            })
            
            summary_data.append({
                "模型": model_name,
                "指标": "成功率",
                "值": success_rate
            })
        
        # 创建DataFrame并保存CSV
        if summary_data:
            df = pd.DataFrame(summary_data)
            
            # 按模型和指标排序
            df = df.sort_values(["模型", "指标"])
            
            csv_path = os.path.join(self.output_dir, "retrieval_summary.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"评估汇总已保存到: {csv_path}")

    def _plot_performance_comparison(self) -> None:
        """绘制性能对比图"""
        if not self.results:
            return
            
        # 配置中文字体支持 - 由于系统原因无法生效，改用英文
        # self._setup_chinese_font()
        
        # 绘制召回率对比图
        plt.figure(figsize=(10, 6))
        
        k_values = [1, 5, 10, 20, 50]
        width = 0.35
        x = np.arange(len(k_values))
        
        for i, (model, results) in enumerate(self.results.items()):
            recalls = [results.get(f"recall@{k}", 0) for k in k_values]
            plt.bar(x + i*width, recalls, width, label=model.upper())
        
        plt.xlabel('K Value')
        plt.ylabel('Recall')
        plt.title('Recall Comparison at Different K Values')
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
        
        plt.xlabel('K Value')
        plt.ylabel('Precision')
        plt.title('Precision Comparison at Different K Values')
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
        
        plt.xlabel('K Value')
        plt.ylabel('Success Rate')
        plt.title('Success Rate Comparison at Different K Values (Success@k)')
        plt.xticks(x + width/2, [f"@{k}" for k in k_values])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 1.0)  # 成功率是0-1的值
        
        # 保存图表
        plt.tight_layout()
        success_plot_file = os.path.join(self.output_dir, "success_comparison.png")
        plt.savefig(success_plot_file, dpi=300)
        
        # 绘制MRR@k对比图
        plt.figure(figsize=(10, 6))
        
        for i, (model, results) in enumerate(self.results.items()):
            mrrs = [results.get(f"mrr@{k}", 0) for k in k_values]
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
        mrr_plot_file = os.path.join(self.output_dir, "mrr_comparison.png")
        plt.savefig(mrr_plot_file, dpi=300)
        
        # 绘制mAP对比图
        plt.figure(figsize=(8, 6))
        
        models = list(self.results.keys())
        maps = [results["mAP"] for results in self.results.values()]
        
        plt.bar(models, maps, color=['#1f77b4', '#ff7f0e'])
        plt.xlabel('Model')
        plt.ylabel('mAP')
        plt.title('mAP Comparison Between Models')
        plt.ylim(0, 1.0)
        
        # 在柱状图上添加数值标签
        for i, v in enumerate(maps):
            plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 保存图表
        plt.tight_layout()
        map_plot_file = os.path.join(self.output_dir, "map_comparison.png")
        plt.savefig(map_plot_file, dpi=300)
        
        print(f"Performance comparison charts saved to: {self.output_dir}")
    
    def _setup_chinese_font(self):
        """配置中文字体支持"""
        import matplotlib as mpl
        # 检测操作系统
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

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="DNMSR检索评估工具")
    parser.add_argument('--embeddings_dir', type=str, required=True, help='嵌入向量目录')
    parser.add_argument('--output_dir', type=str, required=True, help='结果输出目录')
    parser.add_argument('--mode', type=str, default='multimodal', choices=['multimodal', 'text_to_image', 'image_to_text'], 
                       help='检索模式：multimodal(多模态混合检索), text_to_image(文搜图), image_to_text(图搜文)')
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = RetrievalEvaluator(
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        mode=args.mode
    )
    
    # 加载嵌入
    if not evaluator.load_embeddings():
        print("加载嵌入数据失败，退出评估")
        return
    
    # 评估所有模型的检索性能
    evaluator.evaluate_all()
    
    # 生成检索示例分析
    # evaluator.generate_retrieval_examples(num_examples=10, top_k=10)
    
    print("评估完成")

if __name__ == "__main__":
    main() 